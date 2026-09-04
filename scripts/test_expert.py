import numpy as np
import robosuite as suite
import os
import pickle
from robosuite.utils.placement_samplers import UniformRandomSampler
from experts.pick_place_expert import pick_place_expert_action



def sample_target(cube_pos):
    while True:
        target_xy = np.random.uniform(
            low=[-0.15, -0.15],
            high=[0.15, 0.15],
        )

        # Don't generate trivial pick-and-place tasks
        if np.linalg.norm(target_xy - cube_pos[:2]) > 0.12:
            break

    return np.array([
        target_xy[0],
        target_xy[1],
        cube_pos[2],
    ], dtype=np.float32)

placement_initializer = UniformRandomSampler(
    name="ObjectSampler",
    x_range=[-0.10, 0.10],
    y_range=[-0.10, 0.10],
    rotation=None,
    ensure_object_boundary_in_range=True,
    ensure_valid_placement=True,
    reference_pos=[0, 0, 0.8],
    z_offset=0.01,
)

env = suite.make(
    env_name="Lift",
    robots="Panda",
    placement_initializer=placement_initializer,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    camera_names="agentview",
    camera_heights=128,
    camera_widths=128,
    control_freq=10,
)

target_successes = 10
num_success = 0
attempts = 0
trajectory_lengths = []

data_folder = "data/level1"
os.makedirs(data_folder, exist_ok=True)

while attempts <2*target_successes and num_success < target_successes:
    attempts += 1
    obs = env.reset()
 
    trajectory = []
    success = False

    initial_xml = env.sim.model.get_xml()
    initial_state = env.sim.get_state().flatten().copy()

    target_cube_pos = sample_target(obs["cube_pos"])
    state = "APPROACH"
    ctx = {
        "grasped": False, 
        "grasp_offset": None,
        }


    initial_cube_z = obs["cube_pos"][2]

    for t in range(200):
        grasped = env._check_grasp(
            gripper=env.robots[0].gripper,
            object_geoms=env.cube.contact_geoms,
        )
        
        ctx["grasped"] = grasped
        state_before_action = state
        
        action, state = pick_place_expert_action(
            obs,
            state,
            target_cube_pos,
            ctx,
        )

        
        trajectory.append({
            "obs": {
                "joint_pos": obs["robot0_joint_pos"].copy(),
                "eef_pos": obs["robot0_eef_pos"].copy(),
                "eef_quat": obs["robot0_eef_quat"].copy(),
                "gripper_qpos": obs["robot0_gripper_qpos"].copy(),
                "cube_pos": obs["cube_pos"].copy(),

                # Task conditioning
                "target_pos": target_cube_pos.copy(),
            },
            # "image": img,
            "action": action.copy(),
            "expert_state": state_before_action,
        })

        obs, reward, done, info = env.step(action)
        # env.render()

        if t % 20 == 0:
            print(
                f"{t:03d}",
                state,
                "cube:",
                np.round(obs["cube_pos"], 3),
                "target:",
                np.round(target_cube_pos, 3),
                "grasp:",
                grasped,
            )
        
        xy_error = np.linalg.norm(
            obs["cube_pos"][:2] - target_cube_pos[:2]
        )
    
        z_error = abs(
            obs["cube_pos"][2] - target_cube_pos[2]
        )
    
        success = (
            state == "DONE"
            and xy_error < 0.025
            and z_error < 0.025
        )
    
        if success:
            print("SUCCESS at step", t)
            break

    if success:
        num_success += 1
    trajectory_lengths.append(len(trajectory))


    data = {
        "initial_xml": initial_xml,
        "initial_state": initial_state,
        "target_cube_pos": target_cube_pos.copy(),
        "trajectory": trajectory,
    }

    with open(f"{data_folder}/demo_{attempts-1:03d}.pkl", "wb") as f:
        pickle.dump(data, f)

    print(f"Episode {attempts}: steps={len(trajectory)}, success={success}")

print(f"\nSummary: {num_success}/{attempts} successful")
print(f"Trajectory lengths: min={min(trajectory_lengths)}, max={max(trajectory_lengths)}")


env.close()
