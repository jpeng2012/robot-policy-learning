import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

from robosuite.utils.placement_samplers import UniformRandomSampler


# ============================================================
# Project imports
# ============================================================

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

checkpoint_path = "policies/vision_bc/vision_bc_level3_hist_1_4.pth"

num_episodes = 10
max_steps = 700

has_renderer = True


# ============================================================
# Model
# ============================================================

class VisionBCModel(nn.Module):
    def __init__(
            self,
            vision_history_len=1,
            proprio_history_len=4,
            proprio_dim=12,
            action_dim=7,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        self.agent_encoder = resnet18(weights=weights)
        self.wrist_encoder = resnet18(weights=weights)

        self.agent_encoder.fc = nn.Identity()
        self.wrist_encoder.fc = nn.Identity()

        self.vision_history_len = vision_history_len
        self.proprio_history_len = proprio_history_len
        input_dim = 512 * 2 * vision_history_len + proprio_dim * proprio_history_len

        self.motion_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, action_dim-1),
            nn.Tanh(),
        )

        self.gripper_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, agent, wrist, proprio):
        # agent: (B, T, C, H, W)
        B, T, C, H, W = agent.shape
        agent = agent.reshape(-1, *agent.shape[2:])

        # wrist: (B, T, C, H, W)
        wrist = wrist.reshape(-1, *wrist.shape[2:])

        agent_feat = self.agent_encoder(agent).reshape(-1, T*512)
        wrist_feat = self.wrist_encoder(wrist).reshape(-1, T*512)

        proprio = proprio.reshape(B, -1)

        x = torch.cat([agent_feat, wrist_feat, proprio], dim=1)

        motion = self.motion_head(x)
        gripper = self.gripper_head(x)

        return torch.cat([motion, gripper], dim=1)


# ============================================================
# Target sampling
# Must match training distribution
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
    has_offscreen_renderer=True,
    use_camera_obs=True,

    camera_names=[
        "agentview",
        "robot0_eye_in_hand",
    ],

    camera_heights=[224, 224],
    camera_widths=[224, 224],

    control_freq=10,
    horizon=max_steps,
    ignore_done=False,
)


# ============================================================
# Image preprocessing
# ============================================================

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()


def process_image(image_np):
    """
    Convert robosuite image to normalized ResNet tensor.

    robosuite camera output was flipped before JPG storage,
    so during live rollout we apply the same flip here.
    """

    image_np = np.flipud(
        image_np
    ).copy()

    image = Image.fromarray(
        image_np
    ).convert("RGB")

    return transform(image)


# ============================================================
# Proprioception
# ============================================================

def make_proprio(obs):

    return np.concatenate([
        obs["robot0_joint_pos"],       # 7
        obs["robot0_eef_pos"],         # 3
        obs["robot0_gripper_qpos"],    # 2
    ]).astype(np.float32)


# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)

vision_history_len = int(
    checkpoint["vision_history_len"]
)

proprio_history_len = int(
    checkpoint["proprio_history_len"]
)

proprio_mean = np.asarray(
    checkpoint["proprio_mean"],
    dtype=np.float32,
)

proprio_std = np.asarray(
    checkpoint["proprio_std"],
    dtype=np.float32,
)


policy = VisionBCModel(
    vision_history_len=vision_history_len,
    proprio_history_len=proprio_history_len,
    proprio_dim=12,
    action_dim=7,
).to(device)


policy.load_state_dict(
    checkpoint["model_state_dict"]
)

policy.eval()


print("Loaded:", checkpoint_path)
print("device:", device)
print("vision_history_len:", vision_history_len)
print("proprio_history_len:", proprio_history_len)


# ============================================================
# Evaluation statistics
# ============================================================

num_success = 0
num_drop = 0
num_invalid = 0
num_done = 0
num_timeout = 0

left_like = 0
right_like = 0
straight_like = 0

trajectory_lengths = []


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
    print(obs.keys())

    initial_cube_z = obs[
        "cube_pos"
    ][2]

    was_lifted = False
    ever_grasped = False

    cube_x_history = []

    # --------------------------------------------------------
    # History buffers
    # --------------------------------------------------------

    agent_history = deque(
        maxlen=vision_history_len
    )

    wrist_history = deque(
        maxlen=vision_history_len
    )

    proprio_history = deque(
        maxlen=proprio_history_len
    )

    # First frame
    agent_img = process_image(
        obs["agentview_image"]
    )

    wrist_img = process_image(
        obs["robot0_eye_in_hand_image"]
    )

    proprio = make_proprio(
        obs
    )

    proprio = (
        proprio - proprio_mean
    ) / proprio_std

    # Repeat first frame if vision_history_len > 1
    for _ in range(vision_history_len):

        agent_history.append(
            agent_img.clone()
        )

        wrist_history.append(
            wrist_img.clone()
        )

    for _ in range(proprio_history_len):
        proprio_history.append(
            proprio.copy()
        )


    success = False
    failure_reason = None


    # ========================================================
    # Control loop
    # ========================================================

    for t in range(max_steps):

        # ----------------------------------------------------
        # Build model input
        # ----------------------------------------------------

        agent_tensor = torch.stack(
            list(agent_history),
            dim=0,
        ).unsqueeze(0).to(
            device,
            non_blocking=True,
        )

        wrist_tensor = torch.stack(
            list(wrist_history),
            dim=0,
        ).unsqueeze(0).to(
            device,
            non_blocking=True,
        )

        proprio_tensor = torch.from_numpy(
            np.stack(
                list(proprio_history),
                axis=0,
            )
        ).unsqueeze(0).to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Policy inference
        # ----------------------------------------------------

        with torch.no_grad():

            action = (
                policy(
                    agent_tensor,
                    wrist_tensor,
                    proprio_tensor,
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

        action[6] = 1.0 if action[6] > 0 else -1.0

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        obs, reward, done, info = env.step(
            action
        )

        if has_renderer:
            env.render()

        cube_pos = obs[
            "cube_pos"
        ].copy()

        cube_x_history.append(
            cube_pos[0]
        )

        # ----------------------------------------------------
        # Update history
        # ----------------------------------------------------

        agent_img = process_image(
            obs["agentview_image"]
        )

        wrist_img = process_image(
            obs["robot0_eye_in_hand_image"]
        )

        proprio = make_proprio(
            obs
        )

        proprio = (
            proprio - proprio_mean
        ) / proprio_std


        agent_history.append(
            agent_img
        )

        wrist_history.append(
            wrist_img
        )

        proprio_history.append(
            proprio
        )


        # ----------------------------------------------------
        # Invalid state
        # ----------------------------------------------------

        if (
            cube_pos[2] < 0.7
            or not np.all(
                np.isfinite(cube_pos)
            )
        ):
            failure_reason = "invalid_cube_pose"
            num_invalid += 1
            break


        # ----------------------------------------------------
        # Grasp status
        # ----------------------------------------------------

        grasped = env._check_grasp(
            gripper=env.robots[0].gripper,
            object_geoms=env.cube.contact_geoms,
        )

        if grasped:
            ever_grasped = True


        if (
            cube_pos[2]
            > initial_cube_z + 0.05
        ):
            was_lifted = True


        # ----------------------------------------------------
        # Success geometry
        # ----------------------------------------------------

        target_half_size = obs[
            "target_half_size"
        ]

        x_in_target = (
            abs(
                cube_pos[0]
                - target_cube_pos[0]
            )
            < target_half_size[0]
        )

        y_in_target = (
            abs(
                cube_pos[1]
                - target_cube_pos[1]
            )
            < target_half_size[1]
        )

        z_error = abs(
            cube_pos[2]
            - target_cube_pos[2]
        )


        success = (
            x_in_target
            and y_in_target
            and z_error < 0.025
            and not grasped
            and was_lifted
        )


        if success:

            num_success += 1

            trajectory_lengths.append(
                t + 1
            )

            break


        # ----------------------------------------------------
        # Drop detection
        # ----------------------------------------------------

        if (
            was_lifted
            and ever_grasped
            and not grasped
            and not (
                x_in_target
                and y_in_target
            )
            and cube_pos[2]
            < initial_cube_z + 0.04
        ):

            failure_reason = (
                "dropped_object"
            )

            num_drop += 1

            break


        # ----------------------------------------------------
        # Environment termination
        # ----------------------------------------------------

        if done:

            failure_reason = (
                "environment_done"
            )

            num_done += 1

            break


    else:

        failure_reason = "timeout"
        num_timeout += 1


    # ========================================================
    # Route classification
    # ========================================================

    if len(cube_x_history) > 0:

        min_x = np.min(
            cube_x_history
        )

        max_x = np.max(
            cube_x_history
        )

        if min_x < -0.15:

            route_label = "LEFT"
            left_like += 1

        elif max_x > 0.15:

            route_label = "RIGHT"
            right_like += 1

        else:

            route_label = "STRAIGHT"
            straight_like += 1

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
print("Vision BC Evaluation")
print("==============================")

print(
    f"Success: "
    f"{num_success}/{num_episodes} "
    f"({100.0 * num_success / num_episodes:.1f}%)"
)

print("\nFailures:")
print("  drop:", num_drop)
print("  invalid:", num_invalid)
print("  done:", num_done)
print("  timeout:", num_timeout)

print("\nRoute behavior:")
print("  LEFT-like:", left_like)
print("  RIGHT-like:", right_like)
print("  STRAIGHT-like:", straight_like)


if trajectory_lengths:

    print(
        "\nSuccessful trajectory length:"
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