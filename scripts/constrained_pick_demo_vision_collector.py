import os
import shutil
import pickle
import numpy as np
import robosuite as suite

from PIL import Image
from robosuite.utils.placement_samplers import UniformRandomSampler

from experts.pick_place_expert import pick_place_expert_action
from experts.constrained_pick_place_expert import (
    constrained_pick_place_expert_action,
)
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
    has_offscreen_renderer=True,
    use_camera_obs=True,

    camera_names=[
        "agentview",
        "robot0_eye_in_hand",
    ],

    camera_heights=[224, 224],
    camera_widths=[224, 224],

    control_freq=10,
    horizon=500,
    ignore_done=False,
)


# ============================================================
# Collection config
# ============================================================

target_per_route = 250

success_count = {
    "LEFT": 0,
    "RIGHT": 0,
}

attempts = 0

data_folder = "data/level3"
os.makedirs(data_folder, exist_ok=True)


# ============================================================
# Collection
# ============================================================

while (
    success_count["LEFT"] < target_per_route
    or success_count["RIGHT"] < target_per_route
):

    attempts += 1

    # --------------------------------------------------------
    # Pick whichever route still needs data
    # --------------------------------------------------------

    available_routes = [
        r
        for r in ["LEFT", "RIGHT"]
        if success_count[r] < target_per_route
    ]

    route = np.random.choice(
        available_routes
    )

    # --------------------------------------------------------
    # Temporary image directory for this attempt
    #
    # We do NOT yet know whether this attempt will succeed.
    # --------------------------------------------------------

    tmp_dir = os.path.join(
        data_folder,
        f"_tmp_attempt_{attempts:05d}",
    )

    os.makedirs(
        tmp_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    obs = env.reset()

    if attempts == 1:
        print(
            "agent:",
            obs["agentview_image"].shape,
            obs["agentview_image"].dtype,
        )

        print(
            "wrist:",
            obs["robot0_eye_in_hand_image"].shape,
            obs["robot0_eye_in_hand_image"].dtype,
        )

    target_cube_pos = sample_target(
        obs["cube_pos"]
    )

    env.set_target(
        target_cube_pos
    )

    obs = env._get_observations()

    trajectory = []
    success = False
    failure_reason = None

    initial_xml = env.sim.model.get_xml()

    initial_state = (
        env.sim
        .get_state()
        .flatten()
        .copy()
    )

    state = "APPROACH"

    ctx = {
        "grasped": False,
        "grasp_offset": None,
        "route": route,
        "transport_stage": 0,
        "grasp_count": 0,
    }

    # ========================================================
    # Episode rollout
    # ========================================================

    for t in range(500):

        # ----------------------------------------------------
        # Current grasp state
        # ----------------------------------------------------

        grasped = env._check_grasp(
            gripper=env.robots[0].gripper,
            object_geoms=env.cube.contact_geoms,
        )

        ctx["grasped"] = grasped

        state_before_action = state

        # ----------------------------------------------------
        # Expert
        # ----------------------------------------------------

        action, state = constrained_pick_place_expert_action(
            obs,
            state,
            target_cube_pos,
            ctx,
        )

        # ----------------------------------------------------
        # Save current RGB frames as JPG
        # ----------------------------------------------------

        agent_img = obs[
            "agentview_image"
        ]

        wrist_img = obs[
            "robot0_eye_in_hand_image"
        ]

        # robosuite images are typically upside-down
        # relative to normal image convention.
        agent_img = np.flipud(
            agent_img
        )

        wrist_img = np.flipud(
            wrist_img
        )

        agent_filename = (
            f"agent_{t:04d}.jpg"
        )

        wrist_filename = (
            f"wrist_{t:04d}.jpg"
        )

        agent_path = os.path.join(
            tmp_dir,
            agent_filename,
        )

        wrist_path = os.path.join(
            tmp_dir,
            wrist_filename,
        )

        Image.fromarray(
            agent_img
        ).save(
            agent_path,
            quality=90,
        )

        Image.fromarray(
            wrist_img
        ).save(
            wrist_path,
            quality=90,
        )

        # ----------------------------------------------------
        # Save trajectory metadata
        #
        # For now store only filenames.
        # On successful completion we will prepend the final
        # episode directory name.
        # ----------------------------------------------------

        trajectory.append({
            "obs": {
                "joint_pos": obs[
                    "robot0_joint_pos"
                ].copy(),
                "eef_pos": obs["robot0_eef_pos"].copy(),
                "eef_quat": obs["robot0_eef_quat"].copy(),
                "gripper_qpos": obs[
                    "robot0_gripper_qpos"
                ].copy(),
            },

            "agent_image":
                agent_filename,

            "wrist_image":
                wrist_filename,

            "action":
                action.copy(),

            # Metadata only
            "expert_state":
                state_before_action,

            "expert_route":
                route,
        })

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        obs, reward, done, info = env.step(
            action
        )

        cube_pos = obs["cube_pos"]

        # ----------------------------------------------------
        # Catastrophic invalid state
        # ----------------------------------------------------

        if (
            cube_pos[2] < 0.7
            or not np.all(
                np.isfinite(cube_pos)
            )
        ):
            failure_reason = (
                "invalid_cube_pose"
            )
            break

        # ----------------------------------------------------
        # Dropped object
        # ----------------------------------------------------

        if (
            state
            in [
                "LIFT",
                "TRANSPORT",
                "LOWER",
            ]
            and not env._check_grasp(
                gripper=
                    env.robots[0].gripper,

                object_geoms=
                    env.cube.contact_geoms,
            )
        ):
            failure_reason = (
                "dropped_object"
            )
            break

        # ----------------------------------------------------
        # Released?
        # ----------------------------------------------------

        released = not env._check_grasp(
            gripper=env.robots[0].gripper,
            object_geoms=env.cube.contact_geoms,
        )

        # ----------------------------------------------------
        # Target containment
        # ----------------------------------------------------

        target_half_size = obs[
            "target_half_size"
        ]

        x_in_target = (
            abs(
                obs["cube_pos"][0]
                - target_cube_pos[0]
            )
            < target_half_size[0]
        )

        y_in_target = (
            abs(
                obs["cube_pos"][1]
                - target_cube_pos[1]
            )
            < target_half_size[1]
        )

        z_error = abs(
            obs["cube_pos"][2]
            - target_cube_pos[2]
        )

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

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
            failure_reason = (
                "environment_done"
            )
            break

    # ========================================================
    # Failed attempt
    #
    # Delete all images collected for this rollout.
    # ========================================================

    if not success:

        shutil.rmtree(
            tmp_dir,
            ignore_errors=True,
        )

        print(
            f"FAIL route={route}, "
            f"reason={failure_reason}, "
            f"steps={len(trajectory)}"
        )

        continue

    # ========================================================
    # Successful trajectory
    # ========================================================

    idx = success_count[route]

    demo_name = (
        f"{route.lower()}_{idx:03d}"
    )

    final_image_dir = os.path.join(
        data_folder,
        demo_name,
    )

    # --------------------------------------------------------
    # Rename temporary folder:
    #
    # _tmp_attempt_00017/
    #        ↓
    # left_004/
    # --------------------------------------------------------

    os.rename(
        tmp_dir,
        final_image_dir,
    )

    # --------------------------------------------------------
    # Convert filenames into dataset-relative paths
    #
    # agent_0012.jpg
    #       ↓
    # left_004/agent_0012.jpg
    # --------------------------------------------------------

    for step in trajectory:

        step["agent_image"] = os.path.join(
            demo_name,
            step["agent_image"],
        )

        step["wrist_image"] = os.path.join(
            demo_name,
            step["wrist_image"],
        )

    # --------------------------------------------------------
    # Increment count only AFTER successful finalization
    # --------------------------------------------------------

    success_count[route] += 1

    # --------------------------------------------------------
    # Save trajectory metadata
    # --------------------------------------------------------

    data = {
        "initial_xml":
            initial_xml,

        "initial_state":
            initial_state,

        "target_cube_pos":
            target_cube_pos.copy(),

        "route":
            route,

        "success":
            True,

        "trajectory":
            trajectory,
    }

    metadata_path = os.path.join(
        data_folder,
        f"{demo_name}.pkl",
    )

    with open(
        metadata_path,
        "wb",
    ) as f:
        pickle.dump(
            data,
            f,
        )

    print(
        f"SUCCESS route={route} "
        f"L={success_count['LEFT']} "
        f"R={success_count['RIGHT']} "
        f"steps={len(trajectory)}"
    )


# ============================================================
# Summary
# ============================================================

print("\nCollection complete")
print(success_count)
print("attempts:", attempts)

env.close()