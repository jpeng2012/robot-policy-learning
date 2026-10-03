# Successful-trajectory cost-to-go baseline

Build database:
python build_success_cost_to_go_db.py --roots <FM_RAW_ROOT> <DIFFUSION_RAW_ROOT> --output data/success_cost_to_go_db.npz

Required per episode:
metadata.json
task_state.npy
robot_config.npy
cube_pos.npy
target_pos.npy

Descriptor (17D):
- cube - EEF: 3
- target - cube: 3
- cube z: 1
- EEF z: 1
- gripper qpos: 2
- joints: 7

Query:
from success_cost_to_go import SuccessCostToGoScorer
scorer=SuccessCostToGoScorer("data/success_cost_to_go_db.npz",k=20)
r=scorer.score_state(task_state,robot_config,cube_pos,target_pos)
print(r)

