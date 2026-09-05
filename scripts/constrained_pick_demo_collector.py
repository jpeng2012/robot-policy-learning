import os
import pickle
import numpy as np
import robosuite as suite
from robosuite.utils.placement_samplers import UniformRandomSampler
from experts.pick_place_expert import pick_place_expert_action
from experts.constrained_pick_place_expert import constrained_pick_place_expert_action
from envs.constrained_pick_place import ConstrainedPickPlace

def sample_target(cube_pos):
    """
    Target always lies on opposite side of barrier.
    """

    target_x = np.random.uniform(
        -0.08,
        0.08,
    )

    target_y = np.random.uniform(
        0.27,
        0.32,
    )

    return np.array([
        target_x,
        target_y,
        cube_pos[2],
    ], dtype=np.float32)

placement_initializer = UniformRandomSampler(
    name="ObjectSampler",
    x_range=[-0.08, 0.08],
    y_range=[-0.05, 0.02],
    rotation=None,
    ensure_object_boundary_in_range=True,
    ensure_valid_placement=True,
    reference_pos=[0, 0, 0.8],
    z_offset=0.01,
)

env = ConstrainedPickPlace(
    robots="Panda",
    placement_initializer=placement_initializer,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=10,
)

target_per_route = 250

success_count = {
    "LEFT": 0,
    "RIGHT": 0,
}

attempts = 0
data_folder = "data/level2"
os.makedirs(data_folder, exist_ok=True)

# Proper footprint containment
target_geom_id = env.sim.model.geom_name2id(
    "target_marker_geom"
)
cube_geom_id = env.sim.model.geom_name2id(
    "cube_g0"
)

target_size = env.sim.model.geom_size[
    target_geom_id
].copy()

cube_size = env.sim.model.geom_size[
    cube_geom_id
]

while (
    success_count["LEFT"] < target_per_route
    or success_count["RIGHT"] < target_per_route
):
    attempts += 1

    # Pick whichever route still needs data
    available_routes = [
        r for r in ["LEFT", "RIGHT"]
        if success_count[r] < target_per_route
    ]

    route = np.random.choice(available_routes)

    obs = env.reset()

    target_cube_pos = sample_target(obs["cube_pos"])
    env.set_target(target_cube_pos)
    obs = env._get_observations()

    trajectory = []
    success = False
    failure_reason = None

    initial_xml = env.sim.model.get_xml()
    initial_state = env.sim.get_state().flatten().copy()

    state = "APPROACH"

    ctx = {
        "grasped": False,
        "grasp_offset": None,
        "route": route,
        "transport_stage": 0,
        "grasp_count": 0,
    }

    prev_action = np.zeros(7, dtype=np.float32)

    for t in range(500):

        grasped = env._check_grasp(
            gripper=env.robots[0].gripper,
            object_geoms=env.cube.contact_geoms,
        )

        ctx["grasped"] = grasped

        state_before_action = state

        action, state = constrained_pick_place_expert_action(
            obs,
            state,
            target_cube_pos,
            ctx,
        )

        trajectory.append({
            "obs": {
                "eef_pos": obs["robot0_eef_pos"].copy(),
                "gripper_qpos": obs["robot0_gripper_qpos"].copy(),
                "cube_pos": obs["cube_pos"].copy(),
                "target_pos": target_cube_pos.copy(),
                "barrier_pos": obs["barrier_pos"].copy(),
                "barrier_half_size": obs[
                    "barrier_half_size"
                ].copy(),
                "prev_action": prev_action.copy(),
            },

            "action": action.copy(),

            # Metadata only
            "expert_state": state_before_action,
            "expert_route": route,
        })

        obs, reward, done, info = env.step(action)

        prev_action = action.copy()

        cube_pos = obs["cube_pos"]

        # Catastrophic invalid state
        if (
            cube_pos[2] < 0.7
            or not np.all(np.isfinite(cube_pos))
        ):
            failure_reason = "invalid_cube_pose"
            break

        # Dropped object
        if (
            state in ["LIFT", "TRANSPORT", "LOWER"]
            and not env._check_grasp(
                gripper=env.robots[0].gripper,
                object_geoms=env.cube.contact_geoms,
            )
        ):
            failure_reason = "dropped_object"
            break

        
        dx = abs(
            cube_pos[0] - target_cube_pos[0]
        )

        dy = abs(
            cube_pos[1] - target_cube_pos[1]
        )

        inside_target = (
            dx <= target_size[0] - cube_size[0]
            and
            dy <= target_size[1] - cube_size[1]
        )

        released = not env._check_grasp(
            gripper=env.robots[0].gripper,
            object_geoms=env.cube.contact_geoms,
        )

        target_half_size = obs["target_half_size"]
        x_in_target = abs(obs["cube_pos"][0] - target_cube_pos[0]) < target_half_size[0]
        y_in_target = abs(obs["cube_pos"][1] - target_cube_pos[1]) < target_half_size[1]
        z_error = abs(obs["cube_pos"][2] - target_cube_pos[2])

        

        success = (
            state == "DONE"
            and released
            and x_in_target
            and y_in_target
            and z_error < 0.025
        )

        if success:
            break

        if done:
            failure_reason = "environment_done"
            break

    if not success:
        print(
            f"FAIL route={route}, "
            f"reason={failure_reason}, "
            f"steps={len(trajectory)}"
        )
        continue

    idx = success_count[route]
    success_count[route] += 1

    data = {
        "initial_xml": initial_xml,
        "initial_state": initial_state,
        "target_cube_pos": target_cube_pos.copy(),
        "route": route,
        "success": True,
        "trajectory": trajectory,
    }

    filename = (
        f"{data_folder}/"
        f"{route.lower()}_{idx:03d}.pkl"
    )

    with open(filename, "wb") as f:
        pickle.dump(data, f)

    print(
        f"SUCCESS route={route} "
        f"L={success_count['LEFT']} "
        f"R={success_count['RIGHT']} "
        f"steps={len(trajectory)}"
    )


print("Collection complete")
print(success_count)
print("attempts:", attempts)

env.close()