import numpy as np
import robosuite as suite
import os
import pickle
from robosuite.utils.placement_samplers import UniformRandomSampler
from experts.pick_place_expert import pick_place_expert_action
from envs.constrained_pick_place import ConstrainedPickPlace


def sample_target(cube_pos):
    """
    Target always lies on opposite side of barrier.
    """

    target_x = np.random.uniform(
        -0.10,
        0.10,
    )

    target_y = np.random.uniform(
        0.15,
        0.20,
    )

    return np.array([
        target_x,
        target_y,
        cube_pos[2],
    ], dtype=np.float32)

placement_initializer = UniformRandomSampler(
    name="ObjectSampler",
    x_range=[-0.10, 0.10],
    y_range=[-0.18, -0.10],
    rotation=None,
    ensure_object_boundary_in_range=True,
    ensure_valid_placement=True,
    reference_pos=[0, 0, 0.8],
    z_offset=0.01,
)

env = ConstrainedPickPlace(
    robots="Panda",
    placement_initializer=placement_initializer,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=10,
)

print("has_renderer:", env.has_renderer)
print("renderer:", env.renderer)
print("viewer:", env.viewer)

obs = env.reset()

# for t in range(2000):
#     action = np.zeros(env.action_dim, dtype=np.float32)

#     obs, reward, done, info = env.step(action)
#     env.render()

target_cube_pos = sample_target(
    obs["cube_pos"]
)

env.set_target(target_cube_pos)

# Get observations again because target changed
obs = env._get_observations()

print("cube:", obs["cube_pos"])
print("target:", obs["target_pos"])
print("barrier:", obs["barrier_pos"])
print(
    "barrier half size:",
    obs["barrier_half_size"],
)


target_successes = 2
num_success = 0
attempts = 0
trajectory_lengths = []

data_folder = "data/level1"
os.makedirs(data_folder, exist_ok=True)

while attempts <2*target_successes and num_success < target_successes:
    attempts += 1
    obs = env.reset()

    print("barrier body id:",
        env.sim.model.body_name2id("transport_barrier"))
    
    print("barrier geom id:",
        env.sim.model.geom_name2id("transport_barrier_geom"))

    print("target body id:",
        env.sim.model.body_name2id("target_marker"))

    print("target geom id:",
        env.sim.model.geom_name2id("target_marker_geom"))
    
    
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

    for t in range(2000):
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
