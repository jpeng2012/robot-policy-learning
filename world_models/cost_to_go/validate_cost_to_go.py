
import argparse, json, csv
from pathlib import Path
import numpy as np

# Validate whether successful-trajectory retrieval provides a useful
# cost-to-go signal before using it for world-model action selection.
#
# For each query state from a successful episode:
#   1) exclude all states from the same episode to avoid trivial self-match,
#   2) retrieve k nearest states from OTHER successful trajectories,
#   3) estimate remaining steps-to-success from those neighbors,
#   4) compare that estimate with the true remaining steps.
#
# Main metrics:
#   - Spearman correlation:
#       measures whether estimated remaining steps rank trajectory progress
#       correctly. Higher is better; ranking matters more than exact value.
#   - Monotonic fraction:
#       fraction of transitions where estimated remaining steps decrease
#       as the successful trajectory progresses. > 0.5 is desirable.
#   - MAE steps:
#       absolute error in predicted remaining steps; secondary to ranking.
#   - Retrieval distance:
#       indicates how well the query state is covered by the successful
#       trajectory database; large distance means the estimate is less reliable.
#
# Goal:
#   confirm that "states similar to later parts of successful trajectories"
#   receive better cost-to-go estimates before connecting this scorer to the WM.

def rankdata(a):
    a=np.asarray(a); order=np.argsort(a,kind="mergesort"); r=np.empty(len(a),float)
    i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]: j+=1
        r[order[i:j]]=0.5*(i+j-1)+1.0; i=j
    return r

def spearman(x,y):
    if len(x)<2: return float("nan")
    rx,ry=rankdata(x),rankdata(y)
    if rx.std()<1e-12 or ry.std()<1e-12: return float("nan")
    return float(np.corrcoef(rx,ry)[0,1])

class LOOScorer:
    def __init__(self,path,k=20,temp=1.0):
        d=np.load(path,allow_pickle=False)
        self.X=d["descriptors_norm"].astype(np.float32)
        self.steps=d["steps_remaining"].astype(np.float32)
        self.progress=d["progress"].astype(np.float32)
        self.ep=d["episode_id"].astype(np.int32)
        self.t=d["timestep"].astype(np.int32)
        self.k=int(k); self.temp=float(temp)

    def score_index(self,qi):
        q=self.X[qi]; ep=self.ep[qi]
        mask=self.ep!=ep
        X=self.X[mask]; steps=self.steps[mask]; prog=self.progress[mask]
        d2=np.sum((X-q[None,:])**2,axis=1)
        k=min(self.k,len(d2))
        idx=np.argpartition(d2,k-1)[:k] if k<len(d2) else np.arange(len(d2))
        dist=np.sqrt(np.maximum(d2[idx],0))
        o=np.argsort(dist); idx=idx[o]; dist=dist[o]
        w=np.exp(-(dist**2)/max(2*self.temp*self.temp,1e-8))
        ws=max(float(w.sum()),1e-8)
        return {
            "expected_steps":float(np.sum(w*steps[idx])/ws),
            "expected_progress":float(np.sum(w*prog[idx])/ws),
            "mean_distance":float(np.sum(w*dist)/ws),
            "min_distance":float(dist[0]),
        }

def save_csv(path,rows):
    if not rows: return
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--database",default="data/success_cost_to_go_db.npz")
    ap.add_argument("--k",type=int,default=20)
    ap.add_argument("--stride",type=int,default=5)
    ap.add_argument("--max-episodes",type=int,default=None)
    ap.add_argument("--output-dir",default="results/cost_to_go_validation")
    a=ap.parse_args()

    s=LOOScorer(a.database,a.k)
    eps=np.unique(s.ep)
    if a.max_episodes is not None: eps=eps[:a.max_episodes]
    rows=[]; summaries=[]

    for n,ep in enumerate(eps,1):
        idx=np.flatnonzero(s.ep==ep)
        idx=idx[np.argsort(s.t[idx])]
        sel=idx[::a.stride]
        if sel[-1]!=idx[-1]: sel=np.concatenate([sel,idx[-1:]])

        actual=[]; pred=[]; dists=[]
        for qi in sel:
            r=s.score_index(int(qi))
            actual.append(float(s.steps[qi]))
            pred.append(r["expected_steps"])
            dists.append(r["mean_distance"])
            rows.append({
                "episode_id":int(ep),
                "timestep":int(s.t[qi]),
                "actual_steps":float(s.steps[qi]),
                **r,
            })

        actual=np.asarray(actual); pred=np.asarray(pred)
        summaries.append({
            "episode_id":int(ep),
            "queries":len(sel),
            "spearman":spearman(actual,pred),
            "mae_steps":float(np.mean(np.abs(actual-pred))),
            "monotonic_fraction":float(np.mean(np.diff(pred)<=0)) if len(pred)>1 else float("nan"),
            "mean_distance":float(np.mean(dists)),
        })
        if n%25==0 or n==len(eps): print(f"processed {n}/{len(eps)}")

    rho=np.array([x["spearman"] for x in summaries],float); rho=rho[np.isfinite(rho)]
    mono=np.array([x["monotonic_fraction"] for x in summaries],float); mono=mono[np.isfinite(mono)]
    mae=np.array([x["mae_steps"] for x in summaries],float)
    dist=np.array([x["mean_distance"] for x in summaries],float)

    summary={
        "episodes":len(summaries),
        "median_spearman_steps":float(np.median(rho)),
        "mean_spearman_steps":float(np.mean(rho)),
        "median_monotonic_fraction":float(np.median(mono)),
        "mean_monotonic_fraction":float(np.mean(mono)),
        "median_mae_steps":float(np.median(mae)),
        "mean_mae_steps":float(np.mean(mae)),
        "median_retrieval_distance":float(np.median(dist)),
        "mean_retrieval_distance":float(np.mean(dist)),
    }

    od=Path(a.output_dir); od.mkdir(parents=True,exist_ok=True)
    save_csv(od/"queries.csv",rows); save_csv(od/"episodes.csv",summaries)
    (od/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print("saved:",od)

if __name__=="__main__":
    main()
