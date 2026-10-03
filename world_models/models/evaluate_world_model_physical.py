from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from world_models.data import WorldModelFeatureWindowDataset
from world_models.models import LatentWorldModelVjepa


DEFAULTS = {
    "expert_root": "data/level3_wm_features_vjepa21_vitb384",
    "fm_root": "data/policy_wm_features_vjepa21_fm",
    "diffusion_root": "data/policy_wm_features_vjepa21_df",
    "expert_checkpoint": "checkpoints/world_model_vjepa/world_model_best.pth",
    "mixed_checkpoint": "checkpoints/world_model_vjepa_mixed/world_model_best.pth",
    "split_seed": 42,
    "expert_train_ratio": 0.9,
    "policy_val_ratio": 0.2,
    "batch_size": 128,
    "num_workers": 4,
    "visual_feature_dim": 768,
    "latent_dim": 384,
    "latent_grid_size": 4,
    "task_state_dim": 9,
    "robot_config_dim": 7,
    "action_dim": 7,
    "horizon": 16,
    "num_dynamics_layers": 6,
    "num_dynamics_heads": 6,
    "dynamics_ff_dim": 1536,
    "dropout": 0.1,
}

MODEL_INPUT_KEYS = {
    "agent_features",
    "wrist_features",
    "task_state",
    "robot_config",
    "actions",
    "future_agent_features",
    "future_wrist_features",
    "future_task_state",
    "future_robot_config",
}


def discover_episodes(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "metadata.json").exists()
    )
    if not dirs:
        raise RuntimeError(f"No episodes found in {root}")
    return dirs


def split_expert_episodes(trajectory_dirs, train_ratio=0.9, seed=42):
    rng = random.Random(seed)
    groups = {}
    for path in trajectory_dirs:
        with (path / "metadata.json").open("r") as f:
            metadata = json.load(f)
        route = metadata.get("route", "UNKNOWN")
        groups.setdefault(route, []).append(path)

    train_dirs, val_dirs = [], []
    for _, paths in sorted(groups.items()):
        paths = list(paths)
        rng.shuffle(paths)
        n = len(paths)
        if n == 1:
            split = 1
        else:
            split = int(train_ratio * n)
            split = max(1, min(split, n - 1))
        train_dirs.extend(paths[:split])
        val_dirs.extend(paths[split:])
    rng.shuffle(train_dirs)
    rng.shuffle(val_dirs)
    return train_dirs, val_dirs


def split_policy_episodes(trajectory_dirs, val_ratio=0.2, seed=42):
    groups = {"success": [], "failure": []}
    for path in trajectory_dirs:
        with (path / "metadata.json").open("r") as f:
            metadata = json.load(f)
        key = "success" if metadata["success"] else "failure"
        groups[key].append(path)

    rng = random.Random(seed)
    train_dirs, val_dirs = [], []
    for _, paths in groups.items():
        paths = sorted(paths)
        rng.shuffle(paths)
        if len(paths) < 2:
            train_dirs.extend(paths)
            continue
        n_val = max(1, round(len(paths) * val_ratio))
        n_val = min(n_val, len(paths) - 1)
        val_dirs.extend(paths[:n_val])
        train_dirs.extend(paths[n_val:])
    train_dirs.sort()
    val_dirs.sort()
    return train_dirs, val_dirs


def build_model(cfg, device):
    return LatentWorldModelVjepa(
        visual_feature_dim=cfg["visual_feature_dim"],
        latent_dim=cfg["latent_dim"],
        latent_grid_size=cfg["latent_grid_size"],
        task_state_dim=cfg["task_state_dim"],
        robot_config_dim=cfg["robot_config_dim"],
        action_dim=cfg["action_dim"],
        horizon=cfg["horizon"],
        num_dynamics_layers=cfg["num_dynamics_layers"],
        num_dynamics_heads=cfg["num_dynamics_heads"],
        dynamics_ff_dim=cfg["dynamics_ff_dim"],
        dropout=cfg["dropout"],
    ).to(device)


def load_model(checkpoint_path, cfg, device):
    model = build_model(cfg, device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded {checkpoint_path} (epoch={checkpoint.get('epoch', 'unknown')})")
    return model


def move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def denormalize_task(model, x):
    return x * model.task_std + model.task_mean


def denormalize_config(model, x):
    return x * model.config_std + model.config_mean


def quaternion_angle_deg(q_pred, q_true):
    q_pred = torch.nn.functional.normalize(q_pred, dim=-1)
    q_true = torch.nn.functional.normalize(q_true, dim=-1)
    dot = torch.sum(q_pred * q_true, dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot) * (180.0 / math.pi)


@torch.no_grad()
def evaluate_physical_metrics(model, loader, device, horizons=(1, 4, 8, 16)):
    horizons = [h for h in horizons if h <= model.horizon]
    acc = {
        h: {
            "n": 0,
            "eef_pos_cm_sum": 0.0,
            "eef_pos_cm_sq_sum": 0.0,
            "eef_ori_deg_sum": 0.0,
            "eef_ori_deg_sq_sum": 0.0,
            "gripper_mae_sum": 0.0,
            "joint_rmse_rad_sum": 0.0,
            "joint_mae_rad_sum": 0.0,
        }
        for h in horizons
    }

    for batch in tqdm(loader, desc="physical eval", leave=False):
        batch = move_batch(batch, device)
        model_batch = {k: v for k, v in batch.items() if k in MODEL_INPUT_KEYS}

        out = model.forward_features(**model_batch)
        pred = out["predicted_future"]
        target = out["target_future"]

        pred_task = denormalize_task(model, pred["task_state"])
        true_task = denormalize_task(model, target["task_state"])
        pred_cfg = denormalize_config(model, pred["robot_config"])
        true_cfg = denormalize_config(model, target["robot_config"])

        for h in horizons:
            i = h - 1
            p_task = pred_task[:, i]
            t_task = true_task[:, i]
            p_cfg = pred_cfg[:, i]
            t_cfg = true_cfg[:, i]

            pos_cm = torch.linalg.vector_norm(
                p_task[:, 0:3] - t_task[:, 0:3], dim=-1
            ) * 100.0

            ori_deg = quaternion_angle_deg(
                p_task[:, 3:7], t_task[:, 3:7]
            )

            gripper_mae = (
                p_task[:, 7:9] - t_task[:, 7:9]
            ).abs().mean(dim=-1)

            joint_delta = p_cfg - t_cfg
            joint_rmse = torch.sqrt(torch.mean(joint_delta ** 2, dim=-1))
            joint_mae = joint_delta.abs().mean(dim=-1)

            n = p_task.shape[0]
            a = acc[h]
            a["n"] += n
            a["eef_pos_cm_sum"] += pos_cm.sum().item()
            a["eef_pos_cm_sq_sum"] += (pos_cm ** 2).sum().item()
            a["eef_ori_deg_sum"] += ori_deg.sum().item()
            a["eef_ori_deg_sq_sum"] += (ori_deg ** 2).sum().item()
            a["gripper_mae_sum"] += gripper_mae.sum().item()
            a["joint_rmse_rad_sum"] += joint_rmse.sum().item()
            a["joint_mae_rad_sum"] += joint_mae.sum().item()

    results = {}
    for h, a in acc.items():
        n = max(a["n"], 1)
        results[h] = {
            "n": a["n"],
            "eef_pos_mean_cm": a["eef_pos_cm_sum"] / n,
            "eef_pos_rmse_cm": math.sqrt(a["eef_pos_cm_sq_sum"] / n),
            "eef_ori_mean_deg": a["eef_ori_deg_sum"] / n,
            "eef_ori_rmse_deg": math.sqrt(a["eef_ori_deg_sq_sum"] / n),
            "gripper_mae": a["gripper_mae_sum"] / n,
            "joint_rmse_rad": a["joint_rmse_rad_sum"] / n,
            "joint_mae_rad": a["joint_mae_rad_sum"] / n,
        }
    return results


def print_results(model_name, source_name, results):
    print()
    print("=" * 92)
    print(f"{model_name} | {source_name}")
    print("=" * 92)
    print(
        f"{'H':>3} {'EEF mean cm':>12} {'EEF RMSE cm':>12} "
        f"{'Ori mean deg':>13} {'Ori RMSE deg':>13} "
        f"{'Grip MAE':>10} {'Joint RMSE':>12} {'Joint MAE':>11}"
    )
    for h, r in results.items():
        print(
            f"{h:>3d} "
            f"{r['eef_pos_mean_cm']:>12.4f} "
            f"{r['eef_pos_rmse_cm']:>12.4f} "
            f"{r['eef_ori_mean_deg']:>13.4f} "
            f"{r['eef_ori_rmse_deg']:>13.4f} "
            f"{r['gripper_mae']:>10.6f} "
            f"{r['joint_rmse_rad']:>12.6f} "
            f"{r['joint_mae_rad']:>11.6f}"
        )


def make_val_loaders(cfg):
    expert_dirs = discover_episodes(cfg["expert_root"])
    fm_dirs = discover_episodes(cfg["fm_root"])
    diffusion_dirs = discover_episodes(cfg["diffusion_root"])

    _, expert_val = split_expert_episodes(
        expert_dirs,
        train_ratio=cfg["expert_train_ratio"],
        seed=cfg["split_seed"],
    )
    _, fm_val = split_policy_episodes(
        fm_dirs,
        val_ratio=cfg["policy_val_ratio"],
        seed=cfg["split_seed"],
    )
    _, diffusion_val = split_policy_episodes(
        diffusion_dirs,
        val_ratio=cfg["policy_val_ratio"],
        seed=cfg["split_seed"],
    )

    val_dirs = {
        "expert": expert_val,
        "fm": fm_val,
        "diffusion": diffusion_val,
    }

    print("validation episodes:", {k: len(v) for k, v in val_dirs.items()})

    loaders = {}
    for name, dirs in val_dirs.items():
        ds = WorldModelFeatureWindowDataset(
            trajectory_dirs=dirs,
            horizon=cfg["horizon"],
        )
        print(f"{name:10s}: {len(ds)} windows")
        loaders[name] = DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
    return loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-checkpoint", default=DEFAULTS["expert_checkpoint"])
    parser.add_argument("--mixed-checkpoint", default=DEFAULTS["mixed_checkpoint"])
    parser.add_argument("--expert-root", default=DEFAULTS["expert_root"])
    parser.add_argument("--fm-root", default=DEFAULTS["fm_root"])
    parser.add_argument("--diffusion-root", default=DEFAULTS["diffusion_root"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--mixed-only", action="store_true")
    parser.add_argument("--output-json", default="physical_eval_results.json")
    args = parser.parse_args()

    cfg = DEFAULTS.copy()
    cfg.update({
        "expert_root": args.expert_root,
        "fm_root": args.fm_root,
        "diffusion_root": args.diffusion_root,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    loaders = make_val_loaders(cfg)

    checkpoints = {"mixed": args.mixed_checkpoint}
    if not args.mixed_only:
        checkpoints = {
            "expert_only": args.expert_checkpoint,
            "mixed": args.mixed_checkpoint,
        }

    all_results = {}

    for model_name, checkpoint_path in checkpoints.items():
        print()
        print("#" * 92)
        print(f"Evaluating model: {model_name}")
        print("#" * 92)

        model = load_model(checkpoint_path, cfg, device)
        all_results[model_name] = {}

        for source_name, loader in loaders.items():
            results = evaluate_physical_metrics(
                model,
                loader,
                device,
                horizons=(1, 4, 8, 16),
            )
            all_results[model_name][source_name] = results
            print_results(model_name, source_name, results)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_ready = {
        model_name: {
            source_name: {
                str(h): metrics
                for h, metrics in source_results.items()
            }
            for source_name, source_results in model_results.items()
        }
        for model_name, model_results in all_results.items()
    }

    output_path = Path(args.output_json)
    with output_path.open("w") as f:
        json.dump(json_ready, f, indent=2)

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
