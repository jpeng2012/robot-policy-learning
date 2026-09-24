import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

from robosuite.utils.placement_samplers import UniformRandomSampler

from policies.common.observation_encoder_spatial import ObservationEncoder
from policies.act.model_spatial import ACTPolicy

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

checkpoint_path = "policies/act/act_1_4_ep015.pth"

num_episodes = 20
max_steps = 500

has_renderer = True


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

action_horizon = int(
    checkpoint.get(
        "action_horizon",
        16,
    )
)

latent_dim = int(
    checkpoint.get(
        "latent_dim",
        32,
    )
)

condition_dim = int(
    checkpoint.get(
        "condition_dim",
        256,
    )
)

hidden_dim = int(
    checkpoint.get(
        "hidden_dim",
        256,
    )
)

proprio_dim = int(
    checkpoint.get(
        "proprio_dim",
        12,
    )
)

action_dim = int(
    checkpoint.get(
        "action_dim",
        7,
    )
)

dim_feedforward = int(
    checkpoint.get(
        "dim_feedforward",
        3200,
    )
)

num_decoder_layer = int(
    checkpoint.get(
        "num_decoder_layer",
        4,
    )
)

num_encoder_layer = int(
    checkpoint.get(
        "num_encoder_layer",
        4,
    )
)

num_heads = int(
    checkpoint.get(
        "num_heads",
        4,
    )
)

execution_horizon = 4

# ============================================================
# ACT latent-diversity diagnostics
#
# Normal rollout is unchanged: policy(...) still uses z = 0.
# Separately, at the first successful lift in each episode, we
# decode the SAME observation using many fixed z ~ N(0, I).
# ============================================================

run_latent_diagnostics = True
num_latent_samples = 50
latent_diag_seed = 20260922
route_action_dim = 0

_latent_gen = torch.Generator(device=device)
_latent_gen.manual_seed(latent_diag_seed)
fixed_latents = torch.randn(
    num_latent_samples,
    latent_dim,
    generator=_latent_gen,
    device=device,
)

latent_diag_records = []

agent_encoder = resnet18(
    weights=weights
)

wrist_encoder = resnet18(
    weights=weights
)

agent_encoder.fc = nn.Identity()
wrist_encoder.fc = nn.Identity()

obs_encoder = ObservationEncoder(
    agent_encoder=agent_encoder,
    wrist_encoder=wrist_encoder,

    proprio_dim=proprio_dim,

    vision_history_len=vision_history_len,
    proprio_history_len=proprio_history_len,

    condition_dim=condition_dim,
).to(device)

agent_feat_dim = 512 * vision_history_len
wrist_feat_dim = 512 * vision_history_len
proprio_feat_dim = proprio_dim * proprio_history_len

policy = ACTPolicy(
    observation_encoder=obs_encoder,
    conditon_dim=condition_dim,
    agent_feat_dim=agent_feat_dim,
    wrist_feat_dim=wrist_feat_dim,
    proprio_feat_dim=proprio_feat_dim,
    action_dim=action_dim,
    action_horizon=action_horizon,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    num_encoder_layer=num_encoder_layer,
    num_decoder_layer=num_decoder_layer,
    num_heads=num_heads,
    dim_feedforward=dim_feedforward,
).to(device)


policy.load_state_dict(
    checkpoint["policy_state_dict"]
)

policy.eval()


print("Loaded:", checkpoint_path)
print("device:", device)
print("vision_history_len:", vision_history_len)
print("proprio_history_len:", proprio_history_len)

# ============================================================
# ACT latent-diversity diagnostic helpers
# ============================================================

@torch.no_grad()
def sample_chunks_over_z(
    policy, agent_tensor, wrist_tensor, proprio_tensor, latent_bank,
):
    """Fix observation o and vary only z. Returns (N, H, D)."""
    (
        _condition,
        agent_feat,
        wrist_feat,
        proprio_feat,
    ) = policy.observation_encoder(
        agent_tensor, wrist_tensor, proprio_tensor, return_features=True,
    )

    n = latent_bank.shape[0]
    agent_feat = agent_feat.expand(n, *agent_feat.shape[1:])
    wrist_feat = wrist_feat.expand(n, *wrist_feat.shape[1:])
    proprio_feat = proprio_feat.expand(n, *proprio_feat.shape[1:])

    return policy.decoder(
        agent_feat, wrist_feat, proprio_feat, latent_bank,
    )


@torch.no_grad()
def compute_latent_diversity(
    policy, agent_tensor, wrist_tensor, proprio_tensor, latent_bank, route_dim=0,
):
    """Compute Var_z A(o,z), route-score spread, and chunk diversity."""
    chunks = sample_chunks_over_z(
        policy, agent_tensor, wrist_tensor, proprio_tensor, latent_bank,
    )
    motion = chunks[..., :6]

    action_var_hd = chunks.var(dim=0, unbiased=False)
    motion_var_hd = motion.var(dim=0, unbiased=False)

    route_score = motion[:, :, route_dim].sum(dim=1)
    q_levels = torch.tensor(
        [0.10, 0.25, 0.50, 0.75, 0.90],
        device=route_score.device,
        dtype=route_score.dtype,
    )
    q = torch.quantile(route_score, q_levels).cpu().numpy()

    flat = motion.flatten(1)
    pairwise = torch.cdist(flat, flat, p=2)
    n = flat.shape[0]
    upper = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=flat.device), diagonal=1
    )
    pairwise_chunk_dist = pairwise[upper].mean().item() if n > 1 else 0.0

    per_dim_var = motion_var_hd.mean(dim=0).cpu().numpy()

    return {
        "mean_action_var": action_var_hd.mean().item(),
        "motion_action_var": motion_var_hd.mean().item(),
        "route_score_mean": route_score.mean().item(),
        "route_score_std": route_score.std(unbiased=False).item(),
        "route_q10": float(q[0]),
        "route_q25": float(q[1]),
        "route_q50": float(q[2]),
        "route_q75": float(q[3]),
        "route_q90": float(q[4]),
        "pairwise_chunk_dist": pairwise_chunk_dist,
        "per_dim_var": per_dim_var,
    }


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

    prev_xy_error = None

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
    latent_diag_collected = False

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

    t = 0
    while t < max_steps:


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

            action_chunk, _, _ = policy(
                    agent_tensor,
                    wrist_tensor,
                    proprio_tensor,
                )
            action_chunk = action_chunk.squeeze(0).cpu().numpy()
            

        action_chunk[:, 6] = np.where(
            action_chunk[:, 6] > 0,
            1.0,
            -1.0,
        )

        assert action_chunk.shape == (
            action_horizon,
            7,
        )
        
        action_chunk = np.clip(
            action_chunk,
            -1.0,
            1.0,
        )

        # ========================================================
        # Execute only first execution_horizon actions
        # ========================================================

        should_break = False

        for action_idx in range(
            min(
                execution_horizon,
                max_steps - t,
            )
        ):

            action = action_chunk[
                action_idx
            ]

            obs, reward, done, info = env.step(
                action
            )

            t += 1

            if has_renderer:
                env.render()

            cube_pos = obs[
                "cube_pos"
            ].copy()

            cube_x_history.append(
                cube_pos[0]
            )

            # ====================================================
            # Update observation histories after every action
            # ====================================================

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

            # ====================================================
            # Invalid
            # ====================================================

            if (
                cube_pos[2] < 0.7
                or not np.all(
                    np.isfinite(cube_pos)
                )
            ):

                failure_reason = (
                    "invalid_cube_pose"
                )

                num_invalid += 1
                should_break = True
                break

            # ====================================================
            # Grasp status
            # ====================================================

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

            # ====================================================
            # ACT latent-diversity diagnostic
            # ====================================================
            if (
                run_latent_diagnostics
                and was_lifted
                and not latent_diag_collected
            ):
                diag_agent_tensor = torch.stack(
                    list(agent_history), dim=0
                ).unsqueeze(0).to(device, non_blocking=True)

                diag_wrist_tensor = torch.stack(
                    list(wrist_history), dim=0
                ).unsqueeze(0).to(device, non_blocking=True)

                diag_proprio_tensor = torch.from_numpy(
                    np.stack(list(proprio_history), axis=0)
                ).unsqueeze(0).to(device, non_blocking=True)

                diag = compute_latent_diversity(
                    policy=policy,
                    agent_tensor=diag_agent_tensor,
                    wrist_tensor=diag_wrist_tensor,
                    proprio_tensor=diag_proprio_tensor,
                    latent_bank=fixed_latents,
                    route_dim=route_action_dim,
                )
                diag["episode"] = episode
                diag["step"] = t
                latent_diag_records.append(diag)
                latent_diag_collected = True

                pdv = diag["per_dim_var"]
                print(
                    "\n[ACT latent diagnostic] "
                    f"episode={episode} step={t} N_z={num_latent_samples}"
                )
                print(
                    f"  mean_action_var={diag['mean_action_var']:.6f} "
                    f"motion_var={diag['motion_action_var']:.6f}"
                )
                print(
                    f"  route_score mean={diag['route_score_mean']:+.4f} "
                    f"std={diag['route_score_std']:.4f}"
                )
                print(
                    "  route_score quantiles "
                    f"q10={diag['route_q10']:+.3f} "
                    f"q25={diag['route_q25']:+.3f} "
                    f"q50={diag['route_q50']:+.3f} "
                    f"q75={diag['route_q75']:+.3f} "
                    f"q90={diag['route_q90']:+.3f}"
                )
                print(
                    f"  pairwise_chunk_dist={diag['pairwise_chunk_dist']:.4f}"
                )
                print(
                    "  motion per-dim Var_z "
                    f"x={pdv[0]:.6f} y={pdv[1]:.6f} z={pdv[2]:.6f} "
                    f"r0={pdv[3]:.6f} r1={pdv[4]:.6f} r2={pdv[5]:.6f}"
                )

            # Success geometry
            # ====================================================

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

            rel_target = target_cube_pos - cube_pos
            eef_rel_target = target_cube_pos - obs["robot0_eef_pos"]

            gripper_cube_dist = np.linalg.norm(
                obs["robot0_eef_pos"] - obs["cube_pos"]
            )

            print(
                f"t={t} "
                f"cube_rel=({rel_target[0]:+.3f}, {rel_target[1]:+.3f}, {rel_target[2]:+.3f}) "
                f"eef_rel=({eef_rel_target[0]:+.3f}, {eef_rel_target[1]:+.3f}, {eef_rel_target[2]:+.3f}) "
                f"action=({action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f})"
                f"gripper_cube_dist={gripper_cube_dist:.3f}"
            )

            xy_error = np.linalg.norm(
                target_cube_pos[:2] - cube_pos[:2]
            )

            if prev_xy_error is not None:
                delta_xy_error = xy_error - prev_xy_error

                print(
                    f"xy_error={xy_error:.4f} "
                    f"d_error={delta_xy_error:+.4f}"
                )

            prev_xy_error = xy_error

            if success:

                num_success += 1

                trajectory_lengths.append(
                    t
                )

                should_break = True
                break

            # ====================================================
            # Drop
            # ====================================================

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

                should_break = True
                break

            # ====================================================
            # Environment done
            # ====================================================

            if done:

                failure_reason = (
                    "environment_done"
                )

                num_done += 1

                should_break = True
                break


        # End execution_horizon

        if should_break:
            break


    # ============================================================
    # Timeout
    # ============================================================

    if (
        not success
        and failure_reason is None
        and t >= max_steps
    ):
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
        f"failure={failure_reason} | "
        f"total success {num_success}"
    )


# ============================================================
# Summary
# ============================================================

print("\n==============================")
print("ACT Evaluation")
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

# ============================================================
# ACT latent-diversity summary
# ============================================================
if run_latent_diagnostics:
    print("\n==============================")
    print("ACT Latent-Diversity Diagnostics")
    print("==============================")

    if latent_diag_records:
        def _diag_mean(key):
            return float(np.mean([r[key] for r in latent_diag_records]))

        print("Observations analyzed:", len(latent_diag_records))
        print("Fixed latent samples / observation:", num_latent_samples)
        print(f"Mean Var_z[A]: {_diag_mean('mean_action_var'):.6f}")
        print(f"Mean motion Var_z[A]: {_diag_mean('motion_action_var'):.6f}")
        print(f"Mean route-score std: {_diag_mean('route_score_std'):.4f}")
        print(
            f"Mean pairwise chunk distance: "
            f"{_diag_mean('pairwise_chunk_dist'):.4f}"
        )

        per_dim = np.stack([
            r["per_dim_var"] for r in latent_diag_records
        ]).mean(axis=0)
        print(
            "Mean per-dim motion Var_z: "
            f"x={per_dim[0]:.6f} y={per_dim[1]:.6f} z={per_dim[2]:.6f} "
            f"r0={per_dim[3]:.6f} r1={per_dim[4]:.6f} r2={per_dim[5]:.6f}"
        )

        print("\nPer-observation route-score spread:")
        for r in latent_diag_records:
            print(
                f"  ep={r['episode']:03d} t={r['step']:03d} "
                f"std={r['route_score_std']:.4f} "
                f"q10={r['route_q10']:+.3f} "
                f"q50={r['route_q50']:+.3f} "
                f"q90={r['route_q90']:+.3f}"
            )
    else:
        print(
            "No diagnostic observation was collected; "
            "no episode reached the first-lift condition."
        )


env.close()