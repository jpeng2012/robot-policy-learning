import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from robosuite.utils.placement_samplers import UniformRandomSampler

# ---------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.constrained_pick_place import ConstrainedPickPlace


# ============================================================
# Config
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint_path = "policies/bc/bc_level2_chunk.pth"

num_episodes = 10
max_steps = 500
history_len = 4

prediction_horizon = 16
execution_horizon = 4

has_renderer = True


# ============================================================
# BC chunk model
# ============================================================

class ChunkBCPolicy(nn.Module):
    def __init__(self, input_dim=68, horizon=16, action_dim=7):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, horizon * action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        y = self.net(x)

        return y.view(
            x.shape[0],
            self.horizon,
            self.action_dim,
        )


# ============================================================
# Feature construction
# Must exactly match build_dataset.py
# ============================================================

def make_feature(
    obs,
    target_cube_pos,
    # prev_action,
):
    return np.concatenate([
        obs["robot0_eef_pos"],        # 3
        obs["robot0_gripper_qpos"],   # 2
        obs["cube_pos"],              # 3
        target_cube_pos,              # 3
        obs["barrier_pos"],           # 3
        obs["barrier_half_size"],     # 3
        # prev_action,                  # 7
    ]).astype(np.float32)


# ============================================================
# Target sampling
# Same distribution as demo collection
# ============================================================

def sample_target(cube_pos):

    target_x = np.random.uniform(
        -0.08,
        0.08,
    )

    target_y = np.random.uniform(
        0.27,
        0.32,
    )

    return np.array(
        [
            target_x,
            target_y,
            cube_pos[2],
        ],
        dtype=np.float32,
    )


# ============================================================
# Environment
# ============================================================

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

    has_renderer=has_renderer,
    has_offscreen_renderer=False,
    use_camera_obs=False,

    control_freq=10,

    horizon=max_steps,
    ignore_done=False,
)


# ============================================================
# Load model
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)

x_mean = checkpoint["x_mean"].cpu().numpy()
x_std = checkpoint["x_std"].cpu().numpy()

input_dim = checkpoint["input_dim"]
action_dim = checkpoint["action_dim"]


policy = ChunkBCPolicy(
    input_dim=input_dim,
    action_dim=action_dim,
    horizon=prediction_horizon,
).to(device)

policy.load_state_dict(
    checkpoint["model_state_dict"]
)

policy.eval()

print("Loaded:", checkpoint_path)
print("device:", device)


# ============================================================
# Geometry for success test
# ============================================================

# These are fixed across episodes.

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
].copy()

print("target half size:", target_size)
print("cube half size:", cube_size)


# ============================================================
# Evaluation stats
# ============================================================

num_success = 0
num_drop = 0
num_invalid = 0
num_done = 0
num_timeout = 0

trajectory_lengths = []

# Crude route classification based on maximum lateral excursion.
left_like = 0
right_like = 0
straight_like = 0


# ============================================================
# Rollout
# ============================================================

for episode in range(num_episodes):

    obs = env.reset()

    target_cube_pos = sample_target(
        obs["cube_pos"]
    )

    env.set_target(
        target_cube_pos
    )

    obs = env._get_observations()

    prev_action = np.zeros(
        7,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Initialize history by repeating first frame
    # --------------------------------------------------------

    history = deque(
        maxlen=history_len
    )

    first_feature = make_feature(
        obs,
        target_cube_pos,
        # prev_action,
    )

    for _ in range(history_len):
        history.append(
            first_feature.copy()
        )

    success = False
    failure_reason = None

    initial_cube_z = obs["cube_pos"][2]

    # Track trajectory to determine which route BC chose
    cube_x_history = []
    cube_y_history = []

    # --------------------------------------------------------
    # Episode rollout
    # --------------------------------------------------------

    was_lifted = False

    for t in range(max_steps):

        # ----------------------------------------------------
        # Construct policy input
        # ----------------------------------------------------

        x = np.concatenate(
            list(history)
        ).astype(np.float32)

        # MUST use training normalization
        x = (
            x - x_mean
        ) / x_std

        x_tensor = (
            torch.from_numpy(x)
            .unsqueeze(0)
            .to(device)
        )

        # ----------------------------------------------------
        # Predict action
        # ----------------------------------------------------

        with torch.no_grad():
            action_chunk = (
                policy(x_tensor)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        action_chunk = np.clip(
            action_chunk,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # Step environment
        # ----------------------------------------------------

        for i in range(execution_horizon):
            action = action_chunk[i, :].squeeze(0).copy()
            obs, reward, done, info = env.step(
                action
            )

            if has_renderer:
                env.render()

            prev_action = action.copy()

            cube_pos = obs["cube_pos"].copy()

            cube_x_history.append(
                cube_pos[0]
            )

            cube_y_history.append(
                cube_pos[1]
            )

            # ----------------------------------------------------
            # Update history
            # ----------------------------------------------------

            feature = make_feature(
                obs,
                target_cube_pos,
                # prev_action,
            )

            history.append(
                feature
            )

        # ----------------------------------------------------
        # Catastrophic invalid state
        # ----------------------------------------------------

        if (
            cube_pos[2] < 0.7
            or not np.all(np.isfinite(cube_pos))
        ):
            failure_reason = "invalid_cube_pose"
            num_invalid += 1
            break

        # ----------------------------------------------------
        # Current grasp status
        # ----------------------------------------------------

        grasped = env._check_grasp(
            gripper=env.robots[0].gripper,
            object_geoms=env.cube.contact_geoms,
        )

        
        # once lifted, keep it true.
        if cube_pos[2] > initial_cube_z + 0.05:
            was_lifted = True

        # ----------------------------------------------------
        # Placement containment
        # ----------------------------------------------------

        dx = abs(
            cube_pos[0]
            - target_cube_pos[0]
        )

        dy = abs(
            cube_pos[1]
            - target_cube_pos[1]
        )

        inside_target = (
            dx
            <= target_size[0]
            - cube_size[0]
            and
            dy
            <= target_size[1]
            - cube_size[1]
        )

        # ----------------------------------------------------
        # Success
        #
        # BC has no expert state machine, so success is purely
        # physical:
        #
        # cube inside target + no longer grasped
        # ----------------------------------------------------

        success = (
            inside_target
            and not grasped
            and abs(
                cube_pos[2]
                - target_cube_pos[2]
            ) < 0.025
        )

        if success:
            num_success += 1

            trajectory_lengths.append(
                t + 1
            )

            break

        # ----------------------------------------------------
        # Detect drop after successful lift
        # ----------------------------------------------------

        if (
            was_lifted
            and not grasped
            and not inside_target
            and cube_pos[2] < initial_cube_z + 0.04
        ):
            failure_reason = "dropped_object"
            num_drop += 1
            break

        # ----------------------------------------------------
        # Environment termination
        # ----------------------------------------------------

        if done:
            failure_reason = "environment_done"
            num_done += 1
            break

    else:
        failure_reason = "timeout"
        num_timeout += 1

    # ========================================================
    # Route classification
    # ========================================================

    if len(cube_x_history) > 0:

        x_arr = np.asarray(
            cube_x_history
        )

        min_x = np.min(x_arr)
        max_x = np.max(x_arr)

        # Barrier ends at approximately +/- 0.10.
        #
        # These thresholds classify whether the BC policy made
        # a significant excursion around either side.
        if min_x < -0.15:
            left_like += 1

            route_label = "LEFT"

        elif max_x > 0.15:
            right_like += 1

            route_label = "RIGHT"

        else:
            straight_like += 1

            route_label = "STRAIGHT"

    else:
        route_label = "UNKNOWN"

    print(
        f"Episode {episode:03d} | "
        f"success={success} | "
        f"route={route_label} | "
        f"failure={failure_reason}"
    )


# ============================================================
# Summary
# ============================================================

print("\n==============================")
print("BC Evaluation")
print("==============================")

print(
    f"Success: "
    f"{num_success}/{num_episodes} "
    f"({100.0 * num_success / num_episodes:.1f}%)"
)

print()
print("Failures:")
print("  drop:", num_drop)
print("  invalid:", num_invalid)
print("  done:", num_done)
print("  timeout:", num_timeout)

print()
print("Route behavior:")
print("  LEFT-like:", left_like)
print("  RIGHT-like:", right_like)
print("  STRAIGHT-like:", straight_like)

if trajectory_lengths:

    print()
    print(
        "Successful trajectory length:"
    )

    print(
        f"  mean="
        f"{np.mean(trajectory_lengths):.1f}"
    )

    print(
        f"  min="
        f"{np.min(trajectory_lengths)}"
    )

    print(
        f"  max="
        f"{np.max(trajectory_lengths)}"
    )


env.close()