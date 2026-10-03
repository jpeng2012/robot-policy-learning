from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

def make_descriptor(task, config, cube, target):
    task=np.asarray(task,dtype=np.float32)
    config=np.asarray(config,dtype=np.float32)
    cube=np.asarray(cube,dtype=np.float32)
    target=np.asarray(target,dtype=np.float32)
    eef=task[...,0:3]; gripper=task[...,7:9]
    return np.concatenate([
        cube-eef,
        target-cube,
        cube[...,2:3],
        eef[...,2:3],
        gripper,
        config,
    ],axis=-1).astype(np.float32)

@dataclass
class RetrievalScore:
    score: float
    expected_steps: float
    expected_progress: float
    mean_distance: float
    min_distance: float
    effective_neighbors: float

class SuccessCostToGoScorer:
    def __init__(self,database_path,k=20,distance_penalty=0.15,temperature=1.0):
        d=np.load(Path(database_path),allow_pickle=False)
        self.X=d["descriptors_norm"].astype(np.float32)
        self.steps=d["steps_remaining"].astype(np.float32)
        self.progress=d["progress"].astype(np.float32)
        self.mean=d["feature_mean"].astype(np.float32)
        self.std=d["feature_std"].astype(np.float32)
        self.k=min(int(k),len(self.X))
        self.distance_penalty=float(distance_penalty)
        self.temperature=float(temperature)
        self.step_scale=max(float(np.percentile(self.steps,90)),1.0)

    def score_descriptor(self,descriptor):
        x=(np.asarray(descriptor,dtype=np.float32)-self.mean)/self.std
        d2=np.sum((self.X-x[None,:])**2,axis=1)
        idx=np.argpartition(d2,self.k-1)[:self.k] if self.k<len(self.X) else np.arange(len(self.X))
        dist=np.sqrt(np.maximum(d2[idx],0.0))
        o=np.argsort(dist); idx=idx[o]; dist=dist[o]
        w=np.exp(-(dist*dist)/max(2*self.temperature*self.temperature,1e-8))
        ws=max(float(w.sum()),1e-8)
        exp_steps=float(np.sum(w*self.steps[idx])/ws)
        exp_prog=float(np.sum(w*self.progress[idx])/ws)
        mean_dist=float(np.sum(w*dist)/ws)
        eff=float((w.sum()**2)/max(np.sum(w**2),1e-8))
        score=-(exp_steps/self.step_scale)-self.distance_penalty*mean_dist
        return RetrievalScore(float(score),exp_steps,exp_prog,mean_dist,float(dist[0]),eff)

    def score_state(self,task,config,cube,target):
        x=make_descriptor(task,config,cube,target)
        if x.ndim!=1:
            raise ValueError(f"score_state expects one state, got {x.shape}")
        return self.score_descriptor(x)

    def score_batch(self,task,config,cube,target):
        X=make_descriptor(task,config,cube,target)
        if X.ndim==1: X=X[None,:]
        return [self.score_descriptor(x) for x in X]
