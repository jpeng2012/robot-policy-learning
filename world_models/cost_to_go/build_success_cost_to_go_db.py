from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

DEFAULT_ROOTS = [
    "data/policy_wm_features_vjepa21_fm",
    "data/policy_wm_features_vjepa21_df",
]

def discover_episode_dirs(roots):
    out=[]
    for root in roots:
        root=Path(root)
        if not root.exists():
            print(f"[skip missing root] {root}")
            continue
        direct=[p for p in root.iterdir() if p.is_dir() and (p/"metadata.json").exists()]
        out.extend(direct if direct else [p for p in root.rglob("*") if p.is_dir() and (p/"metadata.json").exists()])
    uniq=[]; seen=set()
    for p in sorted(out):
        rp=str(p.resolve())
        if rp not in seen:
            seen.add(rp); uniq.append(p)
    return uniq

def load_episode(path):
    path=Path(path)
    with (path/"metadata.json").open("r") as f:
        metadata=json.load(f)
    req=["task_state.npy","robot_config.npy","cube_pos.npy","target_pos.npy"]
    missing=[x for x in req if not (path/x).exists()]
    if missing:
        return None, missing
    task=np.load(path/"task_state.npy").astype(np.float32)
    config=np.load(path/"robot_config.npy").astype(np.float32)
    cube=np.load(path/"cube_pos.npy").astype(np.float32)
    target=np.load(path/"target_pos.npy").astype(np.float32)
    n=len(task)
    if not (len(config)==len(cube)==len(target)==n):
        raise ValueError(f"length mismatch in {path}")
    return dict(metadata=metadata,task=task,config=config,cube=cube,target=target), None

def make_descriptor(task, config, cube, target):
    eef=task[...,0:3]
    gripper=task[...,7:9]
    return np.concatenate([
        cube-eef,
        target-cube,
        cube[...,2:3],
        eef[...,2:3],
        gripper,
        config,
    ],axis=-1).astype(np.float32)

def build_database(episode_dirs, include_expert_without_success=False):
    X=[]; steps=[]; progress=[]; epids=[]; ts=[]
    used=0; skipped_fail=0; skipped_missing=0; missing_examples=[]
    for path in episode_dirs:
        ep,missing=load_episode(path)
        if ep is None:
            skipped_missing+=1
            if len(missing_examples)<8:
                missing_examples.append({"path":str(path),"missing":missing})
            continue
        success=ep["metadata"].get("success",None)
        if success is False or (success is None and not include_expert_without_success):
            skipped_fail+=1; continue
        T=len(ep["task"])-1
        if T<=0: continue
        desc=make_descriptor(ep["task"],ep["config"],ep["cube"],ep["target"])
        t=np.arange(T+1,dtype=np.int32)
        rem=(T-t).astype(np.float32)
        prog=(1.0-rem/max(T,1)).astype(np.float32)
        X.append(desc); steps.append(rem); progress.append(prog)
        epids.append(np.full(T+1,used,dtype=np.int32)); ts.append(t)
        used+=1
    if not X:
        raise RuntimeError("No usable successful episodes. Need metadata.json plus task_state.npy, robot_config.npy, cube_pos.npy, target_pos.npy.")
    X=np.concatenate(X); steps=np.concatenate(steps); progress=np.concatenate(progress)
    epids=np.concatenate(epids); ts=np.concatenate(ts)
    mean=X.mean(0).astype(np.float32)
    std=np.maximum(X.std(0).astype(np.float32),1e-5)
    Xn=((X-mean)/std).astype(np.float32)
    stats={
        "episodes_discovered":len(episode_dirs),
        "successful_episodes_used":used,
        "states":int(len(X)),
        "skipped_failure_or_unlabeled":skipped_fail,
        "skipped_missing_arrays":skipped_missing,
        "descriptor_dim":int(X.shape[1]),
        "missing_examples":missing_examples,
    }
    return X,Xn,steps,progress,epids,ts,mean,std,stats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--roots",nargs="+",default=DEFAULT_ROOTS)
    ap.add_argument("--output",default="data/success_cost_to_go_db.npz")
    ap.add_argument("--include-expert-without-success",action="store_true")
    args=ap.parse_args()
    eps=discover_episode_dirs(args.roots)
    print("episodes discovered:",len(eps))
    X,Xn,steps,progress,epids,ts,mean,std,stats=build_database(eps,args.include_expert_without_success)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,descriptors=X,descriptors_norm=Xn,steps_remaining=steps,progress=progress,episode_id=epids,timestep=ts,feature_mean=mean,feature_std=std)
    meta=out.with_suffix(".json")
    with meta.open("w") as f:
        json.dump({**stats,"roots":args.roots},f,indent=2)
    print("database:",out); print("metadata:",meta); print(json.dumps(stats,indent=2))
    if stats["skipped_missing_arrays"]:
        print("Some roots lack cube_pos.npy / target_pos.npy. Point --roots to raw rollout episode directories.")

if __name__=="__main__":
    main()
