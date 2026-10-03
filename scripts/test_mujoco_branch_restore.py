from __future__ import annotations

import argparse
import copy
import math
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
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.constrained_pick_place import ConstrainedPickPlace
from policies.common.observation_encoder_spatial import ObservationEncoder
from policies.flow_matching.model import FlowMatchingPolicy


DEFAULTS = {
    "fm_checkpoint": "policies/flow_matching/flowmatching_1_4_1.pth",
    "branch_step": 20,
    "max_steps": 180,
    "flow_num_steps": 10,
    "execution_horizon": 4,
    "seed": 2601002,
}


# ============================================================
# Environment
# ============================================================
def make_env(max_steps, has_renderer=False):
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

    return ConstrainedPickPlace(
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


def sample_target(cube_pos, rng):
    return np.array(
        [
            rng.uniform(-0.08, 0.08),
            rng.uniform(0.27, 0.32),
            cube_pos[2],
        ],
        dtype=np.float32,
    )


# ============================================================
# FM policy preprocessing
# ============================================================
WEIGHTS = ResNet18_Weights.DEFAULT
TRANSFORM = WEIGHTS.transforms()


def process_image(image_np):
    image_np = np.flipud(image_np).copy()
    image = Image.fromarray(image_np).convert("RGB")
    return TRANSFORM(image)


def make_proprio(obs):
    return np.concatenate(
        [
            obs["robot0_joint_pos"],       # 7
            obs["robot0_eef_pos"],         # 3
            obs["robot0_gripper_qpos"],    # 2
        ]
    ).astype(np.float32)


def load_fm_policy(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    vision_history_len = int(checkpoint["vision_history_len"])
    proprio_history_len = int(checkpoint["proprio_history_len"])

    proprio_mean = np.asarray(
        checkpoint["proprio_mean"],
        dtype=np.float32,
    )
    proprio_std = np.asarray(
        checkpoint["proprio_std"],
        dtype=np.float32,
    )

    action_horizon = int(checkpoint.get("action_horizon", 16))
    condition_dim = int(checkpoint.get("condition_dim", 256))
    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    proprio_dim = int(checkpoint.get("proprio_dim", 12))
    action_dim = int(checkpoint.get("action_dim", 7))
    dim_feedforward = int(checkpoint.get("dim_feedforward", 3200))
    num_decoder_layer = int(checkpoint.get("num_decoder_layer", 4))
    num_heads = int(checkpoint.get("num_heads", 4))

    agent_encoder = resnet18(weights=WEIGHTS)
    wrist_encoder = resnet18(weights=WEIGHTS)
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

    policy = FlowMatchingPolicy(
        observation_encoder=obs_encoder,
        agent_feat_dim=512 * vision_history_len,
        wrist_feat_dim=512 * vision_history_len,
        proprio_feat_dim=proprio_dim * proprio_history_len,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dim=hidden_dim,
        num_layers=num_decoder_layer,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
    ).to(device)

    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    return {
        "policy": policy,
        "vision_history_len": vision_history_len,
        "proprio_history_len": proprio_history_len,
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "action_horizon": action_horizon,
    }


def init_histories(obs, fm):
    agent = process_image(obs["agentview_image"])
    wrist = process_image(obs["robot0_eye_in_hand_image"])

    proprio = make_proprio(obs)
    proprio = (proprio - fm["proprio_mean"]) / fm["proprio_std"]

    agent_history = deque(maxlen=fm["vision_history_len"])
    wrist_history = deque(maxlen=fm["vision_history_len"])
    proprio_history = deque(maxlen=fm["proprio_history_len"])

    for _ in range(fm["vision_history_len"]):
        agent_history.append(agent.clone())
        wrist_history.append(wrist.clone())

    for _ in range(fm["proprio_history_len"]):
        proprio_history.append(proprio.copy())

    return agent_history, wrist_history, proprio_history


def update_histories(obs, fm, agent_history, wrist_history, proprio_history):
    agent_history.append(
        process_image(obs["agentview_image"])
    )
    wrist_history.append(
        process_image(obs["robot0_eye_in_hand_image"])
    )

    proprio = make_proprio(obs)
    proprio = (proprio - fm["proprio_mean"]) / fm["proprio_std"]
    proprio_history.append(proprio)


@torch.no_grad()
def sample_chunk(
    fm,
    agent_history,
    wrist_history,
    proprio_history,
    device,
    flow_num_steps,
):
    agent = torch.stack(
        list(agent_history),
        dim=0,
    ).unsqueeze(0).to(device)

    wrist = torch.stack(
        list(wrist_history),
        dim=0,
    ).unsqueeze(0).to(device)

    proprio = torch.from_numpy(
        np.stack(list(proprio_history), axis=0)
    ).unsqueeze(0).to(device)

    chunk = fm["policy"].sample_actions(
        agent,
        wrist,
        proprio,
        num_steps=flow_num_steps,
    )[0].cpu().numpy()

    # Match cleaned WM rollout convention.
    chunk[:, 3:6] = 0.0
    chunk[:, 6] = np.where(
        chunk[:, 6] > 0,
        1.0,
        -1.0,
    )
    chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32)

    return chunk


# ============================================================
# Simulator snapshot / restore
# ============================================================
def save_sim_state(env):
    state = env.sim.get_state()

    if hasattr(state, "flatten"):
        state = state.flatten()

    return np.asarray(state).copy()


def restore_sim_state(env, flat_state):
    if hasattr(env.sim, "set_state_from_flattened"):
        env.sim.set_state_from_flattened(flat_state)
    elif hasattr(env.sim, "set_state"):
        env.sim.set_state(flat_state)
    else:
        raise RuntimeError(
            "env.sim exposes neither set_state_from_flattened() "
            "nor set_state()."
        )

    env.sim.forward()
    return env._get_observations()


# ============================================================
# Diagnostics
# ============================================================
def snapshot_observation(obs):
    return {
        "eef_pos": np.asarray(obs["robot0_eef_pos"]).copy(),
        "eef_quat": np.asarray(obs["robot0_eef_quat"]).copy(),
        "joint_pos": np.asarray(obs["robot0_joint_pos"]).copy(),
        "gripper": np.asarray(obs["robot0_gripper_qpos"]).copy(),
        "cube_pos": np.asarray(obs["cube_pos"]).copy(),
    }


def max_abs(a, b):
    return float(np.max(np.abs(a - b)))


def quat_angle_deg(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    q1 /= max(np.linalg.norm(q1), 1e-12)
    q2 /= max(np.linalg.norm(q2), 1e-12)

    dot = np.clip(abs(float(np.dot(q1, q2))), 0.0, 1.0)
    return math.degrees(2.0 * math.acos(dot))


def execute_chunk(env, action_chunk):
    trajectory = []

    for action in action_chunk:
        obs, _, done, _ = env.step(action)

        trajectory.append(
            {
                "eef_pos": np.asarray(
                    obs["robot0_eef_pos"],
                    dtype=np.float64,
                ).copy(),
                "eef_quat": np.asarray(
                    obs["robot0_eef_quat"],
                    dtype=np.float64,
                ).copy(),
                "joint_pos": np.asarray(
                    obs["robot0_joint_pos"],
                    dtype=np.float64,
                ).copy(),
                "gripper": np.asarray(
                    obs["robot0_gripper_qpos"],
                    dtype=np.float64,
                ).copy(),
                "cube_pos": np.asarray(
                    obs["cube_pos"],
                    dtype=np.float64,
                ).copy(),
            }
        )

        if done:
            while len(trajectory) < len(action_chunk):
                trajectory.append(copy.deepcopy(trajectory[-1]))
            break

    return trajectory


def compare_trajectories(traj_a, traj_b, horizons=(1, 4, 8, 16)):
    print()
    print("=" * 100)
    print("NOMINAL REPLAY DISAGREEMENT")
    print("=" * 100)
    print(
        f"{'H':>3} "
        f"{'EEF cm':>10} "
        f"{'Ori deg':>10} "
        f"{'Joint RMSE':>12} "
        f"{'Grip MAE':>10} "
        f"{'Cube cm':>10}"
    )

    for h in horizons:
        if h > len(traj_a) or h > len(traj_b):
            continue

        a = traj_a[h - 1]
        b = traj_b[h - 1]

        eef_cm = (
            np.linalg.norm(
                a["eef_pos"] - b["eef_pos"]
            )
            * 100.0
        )

        ori_deg = quat_angle_deg(
            a["eef_quat"],
            b["eef_quat"],
        )

        joint_rmse = np.sqrt(
            np.mean(
                (
                    a["joint_pos"]
                    - b["joint_pos"]
                ) ** 2
            )
        )

        grip_mae = np.mean(
            np.abs(
                a["gripper"]
                - b["gripper"]
            )
        )

        cube_cm = (
            np.linalg.norm(
                a["cube_pos"] - b["cube_pos"]
            )
            * 100.0
        )

        print(
            f"{h:>3d} "
            f"{eef_cm:>10.6f} "
            f"{ori_deg:>10.6f} "
            f"{joint_rmse:>12.8f} "
            f"{grip_mae:>10.8f} "
            f"{cube_cm:>10.6f}"
        )


def print_restore_check(before, after):
    print()
    print("=" * 100)
    print("IMMEDIATE STATE RESTORE CHECK")
    print("=" * 100)

    eef_cm = (
        np.linalg.norm(
            before["eef_pos"] - after["eef_pos"]
        )
        * 100.0
    )

    joint_max = max_abs(
        before["joint_pos"],
        after["joint_pos"],
    )

    grip_max = max_abs(
        before["gripper"],
        after["gripper"],
    )

    cube_cm = (
        np.linalg.norm(
            before["cube_pos"] - after["cube_pos"]
        )
        * 100.0
    )

    ori_deg = quat_angle_deg(
        before["eef_quat"],
        after["eef_quat"],
    )

    print(f"EEF position difference : {eef_cm:.9f} cm")
    print(f"EEF orientation diff    : {ori_deg:.9f} deg")
    print(f"joint max abs diff      : {joint_max:.12f} rad")
    print(f"gripper max abs diff    : {grip_max:.12f}")
    print(f"cube position difference: {cube_cm:.9f} cm")


def inspect_controller(env):
    print()
    print("=" * 100)
    print("CONTROLLER STATE INSPECTION")
    print("=" * 100)

    robot = env.robots[0]

    candidates = []

    # Common robosuite layouts across versions.
    if hasattr(robot, "controller"):
        candidates.append(("robot.controller", robot.controller))

    if hasattr(robot, "part_controllers"):
        pc = robot.part_controllers
        if isinstance(pc, dict):
            for name, ctrl in pc.items():
                candidates.append(
                    (f"robot.part_controllers[{name!r}]", ctrl)
                )

    if hasattr(robot, "composite_controller"):
        candidates.append(
            ("robot.composite_controller", robot.composite_controller)
        )

    if not candidates:
        print("No common controller attribute found.")
        print("robot attrs containing 'control':")
        print(
            [
                x for x in dir(robot)
                if "control" in x.lower()
            ]
        )
        return

    interesting_names = [
        "goal_pos",
        "goal_ori",
        "goal_orientation",
        "origin_pos",
        "origin_ori",
        "ref_pos",
        "ref_ori_mat",
    ]

    for label, ctrl in candidates:
        print()
        print(label, "->", type(ctrl))

        found = False

        for attr in interesting_names:
            if hasattr(ctrl, attr):
                value = getattr(ctrl, attr)
                try:
                    arr = np.asarray(value)
                    print(
                        f"  {attr}: shape={arr.shape} "
                        f"value={np.array2string(arr, precision=5)}"
                    )
                except Exception:
                    print(f"  {attr}: {value!r}")

                found = True

        if not found:
            attrs = [
                x for x in dir(ctrl)
                if (
                    "goal" in x.lower()
                    or "origin" in x.lower()
                    or "ref_" in x.lower()
                )
                and not x.startswith("__")
            ]
            print("  possible state attrs:", attrs[:30])


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fm-checkpoint",
        default=DEFAULTS["fm_checkpoint"],
    )

    parser.add_argument(
        "--branch-step",
        type=int,
        default=DEFAULTS["branch_step"],
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULTS["max_steps"],
    )

    parser.add_argument(
        "--flow-num-steps",
        type=int,
        default=DEFAULTS["flow_num_steps"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"],
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    fm = load_fm_policy(
        args.fm_checkpoint,
        device,
    )

    env = make_env(
        max_steps=args.max_steps,
        has_renderer=False,
    )

    try:
        obs = env.reset()

        target = sample_target(
            obs["cube_pos"],
            rng,
        )

        env.set_target(target)
        obs = env._get_observations()

        (
            agent_history,
            wrist_history,
            proprio_history,
        ) = init_histories(obs, fm)

        # ----------------------------------------------------
        # Roll nominal policy until branch point.
        # ----------------------------------------------------
        t = 0

        while t < args.branch_step:
            chunk = sample_chunk(
                fm=fm,
                agent_history=agent_history,
                wrist_history=wrist_history,
                proprio_history=proprio_history,
                device=device,
                flow_num_steps=args.flow_num_steps,
            )

            n_exec = min(
                DEFAULTS["execution_horizon"],
                args.branch_step - t,
            )

            for i in range(n_exec):
                obs, _, done, _ = env.step(chunk[i])

                t += 1

                update_histories(
                    obs,
                    fm,
                    agent_history,
                    wrist_history,
                    proprio_history,
                )

                if done:
                    raise RuntimeError(
                        f"Episode ended before branch step {args.branch_step}"
                    )

        print()
        print(
            f"Reached branch step {t}: "
            f"eef={np.asarray(obs['robot0_eef_pos']).round(5)}, "
            f"cube={np.asarray(obs['cube_pos']).round(5)}"
        )

        # Generate ONCE. Both replay A and replay B use exactly
        # this same nominal chunk.
        action_chunk = sample_chunk(
            fm=fm,
            agent_history=agent_history,
            wrist_history=wrist_history,
            proprio_history=proprio_history,
            device=device,
            flow_num_steps=args.flow_num_steps,
        )

        print()
        print("First four nominal actions:")
        print(action_chunk[:4])

        # ----------------------------------------------------
        # Save exact state S.
        # ----------------------------------------------------
        flat_state = save_sim_state(env)
        state_obs = env._get_observations()
        before = snapshot_observation(state_obs)

        inspect_controller(env)

        # ----------------------------------------------------
        # Replay A.
        # ----------------------------------------------------
        traj_a = execute_chunk(
            env,
            action_chunk,
        )

        # ----------------------------------------------------
        # Restore S in SAME environment.
        # ----------------------------------------------------
        restored_obs = restore_sim_state(
            env,
            flat_state,
        )
        after = snapshot_observation(restored_obs)

        print_restore_check(
            before,
            after,
        )

        inspect_controller(env)

        # ----------------------------------------------------
        # Replay B -- exact same actions.
        # ----------------------------------------------------
        traj_b = execute_chunk(
            env,
            action_chunk,
        )

        compare_trajectories(
            traj_a,
            traj_b,
            horizons=(1, 4, 8, 16),
        )

        print()
        print("=" * 100)

        final_eef_cm = (
            np.linalg.norm(
                traj_a[-1]["eef_pos"]
                - traj_b[-1]["eef_pos"]
            )
            * 100.0
        )

        if final_eef_cm < 0.01:
            print(
                "RESULT: same-environment MuJoCo restoration is highly "
                "reproducible. We can safely use same-env branching."
            )
        elif final_eef_cm < 0.1:
            print(
                "RESULT: replay is close but not exact. It may be usable, "
                "but controller-state restoration should still be examined."
            )
        else:
            print(
                "RESULT: simulation state alone is NOT sufficient for "
                "reproducible branching. Controller state must also be "
                "restored or reinitialized consistently."
            )

        print("=" * 100)

    finally:
        env.close()


if __name__ == "__main__":
    main()
