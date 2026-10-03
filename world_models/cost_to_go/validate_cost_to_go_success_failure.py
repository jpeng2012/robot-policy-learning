
import argparse, csv, json
from pathlib import Path
import numpy as np

# Validate whether the success-trajectory retrieval score can distinguish
# successful rollouts from failed rollouts.
#
# The retrieval database contains ONLY states from successful trajectories.
#
# For each state x in a rollout, compute:
#
#   1) Estimated cost-to-go:
#
#        T_hat(x) =
#            weighted average of "steps remaining to success"
#            among the k nearest successful states.
#
#      Smaller is better.
#      Interpretation:
#        "When successful trajectories visited states similar to x,
#         how many steps were usually left until success?"
#
#   2) Distance to successful-state manifold:
#
#        D(x) =
#            weighted average distance to the retrieved successful states.
#
#      Smaller is better.
#      Interpretation:
#        "How similar is x to states actually observed on successful rollouts?"
#
#   3) Combined ranking score:
#
#        S(x) =
#            - T_hat(x) / T_scale
#            - lambda * D(x)
#
#      Higher is better.
#
#      The first term rewards progress toward later successful states.
#      The second term penalizes states far from demonstrated successful
#      behavior, where the cost-to-go estimate is less trustworthy.
#
# Validation compares the first quarter and last quarter of each episode:
#
#   Successful rollout should ideally show:
#       delta_steps    < 0   (estimated remaining work decreases)
#       delta_distance < 0   (moves closer to successful manifold)
#       delta_score    > 0   (overall score improves)
#
#   Failed rollout may initially make progress, but should typically show:
#       weaker reduction in estimated remaining steps,
#       increasing distance from successful behavior,
#       and/or lower final score.
#
# This test does NOT evaluate the world model yet.
# It only checks whether the trajectory-retrieval metric is useful enough
# to later rank WM-predicted action candidates.

def discover(roots):
    out=[]
    for r in roots:
        r=Path(r)
        if not r.exists(): continue
        direct=[p for p in r.iterdir() if p.is_dir() and (p/"metadata.json").exists()]
        out += direct if direct else [p for p in r.rglob("*") if p.is_dir() and (p/"metadata.json").exists()]
    return sorted(set(out))

def load_ep(p):
    with (p/"metadata.json").open() as f: meta=json.load(f)
    req=["task_state.npy","robot_config.npy","cube_pos.npy","target_pos.npy"]
    if any(not (p/x).exists() for x in req): return None
    return meta, np.load(p/"task_state.npy").astype(np.float32), np.load(p/"robot_config.npy").astype(np.float32), np.load(p/"cube_pos.npy").astype(np.float32), np.load(p/"target_pos.npy").astype(np.float32)

def desc(task, cfg, cube, target):
    eef=task[...,0:3]; grip=task[...,7:9]
    return np.concatenate([cube-eef,target-cube,cube[...,2:3],eef[...,2:3],grip,cfg],axis=-1).astype(np.float32)

class Scorer:
    def __init__(self,db,k=20,temp=1.0,penalty=0.15):
        d=np.load(db,allow_pickle=False)
        self.X=d["descriptors_norm"].astype(np.float32)
        self.steps=d["steps_remaining"].astype(np.float32)
        self.progress=d["progress"].astype(np.float32)
        self.mean=d["feature_mean"].astype(np.float32)
        self.std=d["feature_std"].astype(np.float32)
        self.k=k; self.temp=temp; self.penalty=penalty
        self.scale=max(float(np.percentile(self.steps,90)),1.0)

    
    def score(self,x):
        q=(x-self.mean)/self.std
        d2=np.sum((self.X-q[None,:])**2,axis=1)
        k=min(self.k,len(d2))
        idx=np.argpartition(d2,k-1)[:k] if k<len(d2) else np.arange(len(d2))
        dist=np.sqrt(np.maximum(d2[idx],0))
        o=np.argsort(dist); idx=idx[o]; dist=dist[o]
        w=np.exp(-(dist**2)/max(2*self.temp*self.temp,1e-8)); ws=max(float(w.sum()),1e-8)
        
        est=float(np.sum(w*self.steps[idx])/ws)
        md=float(np.sum(w*dist)/ws)
        
        prog=float(np.sum(w*self.progress[idx])/ws)

        # Candidate value combines:
        #   - progress: fewer retrieved steps remaining is better
        #   - support: staying close to known successful states is better
        #
        # Higher score = more promising and better supported by successful experience.

        score=-est/self.scale-self.penalty*md
        return est,prog,md,float(dist[0]),score

def save_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--database",default="data/success_cost_to_go_db.npz")
    ap.add_argument("--roots",nargs="+",required=True)
    ap.add_argument("--k",type=int,default=20)
    ap.add_argument("--stride",type=int,default=5)
    ap.add_argument("--output-dir",default="results/cost_to_go_success_failure")
    a=ap.parse_args()

    scorer=Scorer(a.database,a.k)
    eps=discover(a.roots)
    print("episodes discovered:",len(eps))
    states=[]; episodes=[]

    for eid,p in enumerate(eps):
        ep=load_ep(p)
        if ep is None: continue
        meta,task,cfg,cube,target=ep
        success=bool(meta.get("success",False))
        X=desc(task,cfg,cube,target)
        ids=np.arange(0,len(X),a.stride)
        if ids[-1]!=len(X)-1: ids=np.r_[ids,len(X)-1]
        local=[]
        for t in ids:
            est,prog,md,mind,score=scorer.score(X[t])
            row=dict(episode_id=eid,path=str(p),success=success,timestep=int(t),
                     fraction=float(t)/max(len(X)-1,1),expected_steps=est,
                     expected_progress=prog,mean_distance=md,min_distance=mind,score=score)
            states.append(row); local.append(row)
        q=max(1,len(local)//4)
        first=local[:q]; last=local[-q:]
        avg=lambda rr,key: float(np.mean([x[key] for x in rr]))
        episodes.append(dict(
            episode_id=eid,success=success,queries=len(local),
            first_steps=avg(first,"expected_steps"),last_steps=avg(last,"expected_steps"),
            delta_steps=avg(last,"expected_steps")-avg(first,"expected_steps"),
            first_distance=avg(first,"mean_distance"),last_distance=avg(last,"mean_distance"),
            delta_distance=avg(last,"mean_distance")-avg(first,"mean_distance"),
            first_score=avg(first,"score"),last_score=avg(last,"score"),
            delta_score=avg(last,"score")-avg(first,"score")))
        if (eid+1)%25==0: print(f"processed {eid+1}/{len(eps)}")

    def summarize(group):
        if not group:return {}
        keys=["last_steps","last_distance","last_score","delta_steps","delta_distance","delta_score"]
        return {k:{"mean":float(np.mean([x[k] for x in group])),
                   "median":float(np.median([x[k] for x in group]))} for k in keys}

    succ=[x for x in episodes if x["success"]]
    fail=[x for x in episodes if not x["success"]]
    summary={"episodes_total":len(episodes),"successful_episodes":len(succ),"failed_episodes":len(fail),
             "successful":summarize(succ),"failed":summarize(fail)}

    od=Path(a.output_dir); od.mkdir(parents=True,exist_ok=True)
    save_csv(od/"state_queries.csv",states); save_csv(od/"episode_summary.csv",episodes)
    (od/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print("\nDesired pattern:")
    print("  success: delta_steps < 0, delta_score > 0")
    print("  failure: weaker improvement and/or larger last_distance, lower last_score")

if __name__=="__main__":
    main()
