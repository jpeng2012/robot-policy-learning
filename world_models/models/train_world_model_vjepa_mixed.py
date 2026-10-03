from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import time
import numpy as np
import json
import random

from world_models.data import WorldModelFeatureWindowDataset
from world_models.models import LatentWorldModelVjepa

DEFAULT_CONFIG = {
    # Data
    "split_seed": 42,
    "expert_root": "data/level3_wm_features_vjepa21_vitb384",
    "fm_root": "data/policy_wm_features_vjepa21_fm",
    "diffusion_root": "data/policy_wm_features_vjepa21_df",
    "pretrained_checkpoint": "checkpoints/world_model_vjepa/world_model_best.pth",
    "policy_val_ratio": 0.2,
    "train_val_split": 0.9,

    # Training
    "batch_size": 120,
    "epochs": 10,
    "lr": 3e-5,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "num_workers": 4,

    # Model
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

    # Output
    "output_dir": "checkpoints/world_model_vjepa_mixed",

    # Wandb
    "wandb_project": "robot-world-model",
    "wandb_name": None,
    "no_wandb": False,
}


def load_config(config_path: str | None) -> dict:
    """Load config from YAML file, with defaults for missing values."""
    config = DEFAULT_CONFIG.copy()

    if config_path is not None:
        with open(config_path) as f:
            user_config = yaml.safe_load(f)

        if user_config:
            config.update(user_config)

    return config


def move_batch_to_device(batch, device):
    tensor_keys = [
        "agent_features",
        "wrist_features",
        "task_state",
        "robot_config",
        "actions",
        "future_agent_features",
        "future_wrist_features",
        "future_task_state",
        "future_robot_config",
    ]

    result = {
        key: batch[key].to(device, non_blocking=True)
        for key in tensor_keys
    }

    if "route" in batch:
        result["route"] = batch["route"]

    if "t" in batch:
        result["t"] = batch["t"]

    if "trajectory_id" in batch:
        result["trajectory_id"] = batch["trajectory_id"]

    return result


def split_trajectories(trajectory_dirs, train_ratio=0.9, seed=42):
    """
    Deterministic episode-level stratified split.

    Trajectories are grouped by route using metadata.json,
    then each route is independently shuffled and split.

    This keeps LEFT / RIGHT representation in both train
    and validation sets.
    """
    rng = random.Random(seed)
    groups = {}

    for path in trajectory_dirs:
        metadata_path = (path / "metadata.json")
        with metadata_path.open("r") as f:
            metadata = json.load(f)

        route = metadata.get("route", "UNKNOWN")
        groups.setdefault(route, []).append(path)

    train_dirs = []
    val_dirs = []

    for route, paths in sorted(groups.items()):
        paths = list(paths)
        rng.shuffle(paths)
        n = len(paths)

        if n == 1:
            # With only one trajectory in a group,
            # keep it in training.
            split = 1
        else:
            split = int(train_ratio * n)
            # Ensure both train and val receive at least
            # one trajectory when possible.
            split = max(1, min(split, n - 1))

        route_train = paths[:split]
        route_val = paths[split:]

        train_dirs.extend(route_train)
        val_dirs.extend(route_val)

        print(
            f"route {route}: "
            f"{len(route_train)} train, "
            f"{len(route_val)} val"
        )

    # Shuffle final lists too, while remaining deterministic.
    rng.shuffle(train_dirs)
    rng.shuffle(val_dirs)

    return train_dirs, val_dirs


def compute_horizon_losses(model, predicated_future, target_future):
    """
    Evaluate prediction quality at selected future horizons.

    Horizon 1 means:
        predicted z_{t+1}

    Horizon 16 means:
        predicted z_{t+16}
    """
    results = {}
    horizons = [1, 2, 4, 8, 16]

    for horizon in horizons:
        if horizon > model.horizon:
            continue

        index = horizon - 1
        pred_h = {
            key: value[:, index:index + 1]
            for key, value in predicated_future.items()
        }

        target_h = {
            key: value[:, index:index + 1]
            for key, value in target_future.items()
        }

        losses = model.compute_latent_loss(pred_h, target_h)
        results[horizon] = {
            key: value
            for key, value in losses.items()
        }

    return results


def compute_state_stats(trajectory_dirs):
    task_values = []
    config_values = []

    for path in trajectory_dirs:
        task = np.load(path / "task_state.npy", mmap_mode="r")
        config = np.load(path / "robot_config.npy", mmap_mode="r",)

        task_values.append(np.asarray(task))
        config_values.append(np.asarray(config))

    task_values = np.concatenate(task_values, axis=0)
    config_values = np.concatenate(config_values, axis=0)

    task_mean = torch.tensor(
        task_values.mean(axis=0),
        dtype=torch.float32,
    )

    task_std = torch.tensor(
        task_values.std(axis=0),
        dtype=torch.float32,
    )

    config_mean = torch.tensor(
        config_values.mean(axis=0),
        dtype=torch.float32,
    )

    config_std = torch.tensor(
        config_values.std(axis=0),
        dtype=torch.float32,
    )

    return (task_mean, task_std, config_mean, config_std)


def train_one_epoch(model, loader, optimizer, device, grad_clip_norm: float = 1.0):
    model.train()

    totals = {
        "loss": 0.0,
        "agent_loss": 0.0,
        "wrist_loss": 0.0,
        "task_loss": 0.0,
        "config_loss": 0.0,
    }
    num_batches = 0
    pbar = tqdm(loader, desc="Training", leave=False)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        model_batch = {
            key: value for key, value in batch.items()
            if key in {
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
        }

        optimizer.zero_grad(set_to_none=True)
        output = model.forward_features(**model_batch)
        losses = model.compute_latent_loss(output["predicted_future"], output["target_future"])
        loss = losses["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip_norm,
        )

        optimizer.step()

        for key in totals:
            totals[key] += losses[key].item()

        num_batches += 1
        running_loss = totals["loss"] / num_batches

        pbar.set_postfix(
            batch=f"{loss.item():.4f}",
            avg=f"{running_loss:.4f}",
        )

    return {key: value / max(num_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    metric_keys = [
        "loss",
        "agent_loss",
        "wrist_loss",
        "task_loss",
        "config_loss",
    ]

    # ========================================================
    # Normal world-model prediction
    # ========================================================
    totals = {key: 0.0 for key in metric_keys}

    # ========================================================
    # Copy-current-state baseline
    #
    # Assume:
    #
    #     future == current
    #
    # without running the dynamics model.
    # ========================================================
    copy_totals = {key: 0.0 for key in metric_keys}

    # ========================================================
    # Shuffled-action baseline
    #
    # Same current observation, but wrong action sequence.
    # ========================================================
    shuffle_totals = {key: 0.0 for key in metric_keys}

    num_batches = 0
    horizons = [1, 2, 4, 8, 16]
    horizons = [h for h in horizons if h <= model.horizon]

    horizon_totals = {
        horizon: {key: 0.0 for key in metric_keys}
        for horizon in horizons
    }
    horizon_counts = {key: 0 for key in horizons}

    route_totals = {}
    route_counts = {}

    pbar = tqdm(loader, desc="Validating", leave=False)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        model_batch = {
            key: value for key, value in batch.items()
            if key in {
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
        }

        # ====================================================
        # 1. Normal world-model prediction
        # ====================================================
        output = model.forward_features(**model_batch)
        predictions = output["predicted_future"]
        targets = output["target_future"]
        current_latent = output["current_latent"]
        losses = model.compute_latent_loss(predictions, targets)
        horizon_losses = compute_horizon_losses(model, predictions, targets)
        sample_losses = model.compute_loss_per_sample(predictions, targets)
        routes = batch.get("route")

        for key in metric_keys:
            totals[key] += losses[key].item()

        for horizon, h_losses in horizon_losses.items():
            for key in metric_keys:
                horizon_totals[horizon][key] += h_losses[key].item()

            horizon_counts[horizon] += 1

        if routes is not None:
            for i, route in enumerate(routes):
                route = str(route)
                if route not in route_totals:
                    route_totals[route] = {key: 0.0 for key in metric_keys}
                    route_counts[route] = 0
                for key in metric_keys:
                    route_totals[route][key] += sample_losses[key][i].item()
                route_counts[route] += 1

        # 3. COPY BASELINE
        #
        # Visual:
        #
        #     current frozen V-JEPA feature
        #     repeated H times
        #
        # Physical state:
        #
        #     current state repeated H times
        #
        # Targets use normalized task/config state, so current
        # physical state must be normalized the same way.
        # ====================================================
        H = model.horizon
        copy_agent = batch["agent_features"].unsqueeze(1).expand(-1, H, -1, -1)
        copy_wrist = batch["wrist_features"].unsqueeze(1).expand(-1, H, -1, -1)

        current_task_norm = (batch["task_state"] - model.task_mean) / model.task_std
        current_config_norm = (batch["robot_config"] - model.config_mean) / model.config_std

        copy_task = current_task_norm.unsqueeze(1).expand(-1, H, -1)
        copy_config = current_config_norm.unsqueeze(1).expand(-1, H, -1)

        copy_predictions = {
            "agent_features": copy_agent,
            "wrist_features": copy_wrist,
            "task_state": copy_task,
            "robot_config": copy_config,
        }

        copy_losses = model.compute_latent_loss(copy_predictions, targets)

        for key in metric_keys:
            copy_totals[key] += copy_losses[key].item()

        # ====================================================
        # 4. SHUFFLED-ACTION BASELINE
        #
        # Keep current state fixed.
        #
        # Replace its action chunk with one from another
        # example in the batch.
        # ====================================================
        batch_size = batch["actions"].shape[0]

        if batch_size > 1:
            perm = torch.randperm(batch_size, device=device)
            shuffled_actions = batch["actions"][perm]

            shuffled_predictions = model.predict_future(
                current_latent=current_latent,
                actions=shuffled_actions,
            )

            shuffle_losses = model.compute_latent_loss(shuffled_predictions, targets)

            for key in metric_keys:
                shuffle_totals[key] += shuffle_losses[key].item()
        else:
            # Extremely unlikely
            # but keep accounting consistent.
            for key in metric_keys:
                shuffle_totals[key] += losses[key].item()

        num_batches += 1
        running_loss = totals["loss"] / num_batches

        pbar.set_postfix(
            batch=f"{losses['loss'].item():.4f}",
            avg=f"{running_loss:.4f}",
        )

    mean_loss = {
        key: value / max(num_batches, 1) for key, value in totals.items()
    }

    mean_copy_loss = {
        key: value / max(num_batches, 1) for key, value in copy_totals.items()
    }

    mean_shuffle_loss = {
        key: value / max(num_batches, 1) for key, value in shuffle_totals.items()
    }

    mean_horizon_losses = {}
    for horizon in horizon_totals:
        if horizon_counts[horizon] > 0:
            mean_horizon_losses[horizon] = {
                key: horizon_totals[horizon][key] / horizon_counts[horizon]
                for key in metric_keys
            }

    route_metrics = {}
    for route in route_totals:
        count = route_counts[route]
        route_metrics[route] = {
            key: value / count
            for key, value in route_totals[route].items()
        }

    diagnostics = {
        "copy": mean_copy_loss,
        "shuffle": mean_shuffle_loss,
    }

    # ========================================================
    # Relative gain over shuffled actions
    # ========================================================
    copy_loss = mean_copy_loss["loss"]
    if copy_loss > 0:
        diagnostics["copy_improvement"] = (copy_loss - mean_loss["loss"]) / copy_loss
    else:
        diagnostics["copy_improvement"] = 0.0

    # ========================================================
    # Relative gain over shuffled actions
    # ========================================================
    shuffle_loss = mean_shuffle_loss["loss"]
    if shuffle_loss > 0:
        diagnostics["shuffle_improvement"] = (shuffle_loss - mean_loss["loss"]) / shuffle_loss
    else:
        diagnostics["shuffle_improvement"] = 0.0

    return mean_loss, mean_horizon_losses, diagnostics, route_metrics


def split_policy_episodes(trajectory_dirs, val_ratio=0.2, seed=42):
    """
    Deterministic episode-level split.

    Stratify by success/failure when possible.

    Never split individual windows from an episode.
    """
    groups = {
        "success": [],
        "failure": [],
    }

    for path in trajectory_dirs:
        with (path / "metadata.json").open("r") as f:
            metadata = json.load(f)

        group = (
            "success"
            if metadata["success"]
            else "failure"
        )

        groups[group].append(path)

    rng = random.Random(seed)
    train_dirs = []
    val_dirs = []

    for name, paths in groups.items():
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


@torch.no_grad()
def evaluate_fm_rollouts(model, feature_root, device, batch_size=64, num_workers=4):
    """
    Evaluate an already-loaded, expert-trained WM.

    Does not modify model weights or normalization.
    """
    feature_root = Path(feature_root)

    trajectory_dirs = sorted(
        p
        for p in feature_root.iterdir()
        if (p.is_dir() and (p / "metadata.json").exists())
    )

    train_dirs, val_dirs = split_policy_episodes(
        trajectory_dirs,
        val_ratio=0.2,
        seed=42,
    )

    print("FM train episodes:", len(train_dirs))
    print("FM validation episodes:", len(val_dirs))

    if not val_dirs:
        raise RuntimeError(
            "No FM validation episodes. "
            "Collect more rollouts before splitting."
        )

    # We are NOT training here.
    # Only the held-out episodes are loaded.
    dataset = WorldModelFeatureWindowDataset(
        trajectory_dirs=val_dirs,
        horizon=model.horizon,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.eval()

    (
        val_metrics,
        horizon_losses,
        diagnostics,
        route_metrics,
    ) = validate(
        model=model,
        loader=loader,
        device=device,
    )

    print()
    print("=== FM Zero-Shot Evaluation ===")
    print(f"val loss:     {val_metrics['loss']:.6f}")
    print(f"copy loss:    {diagnostics['copy']['loss']:.6f}")
    print(f"shuffle loss: {diagnostics['shuffle']['loss']:.6f}")
    print(f"improvement vs copy: {diagnostics['copy_improvement'] * 100:.1f}%")
    print(f"improvement vs shuffle: {diagnostics['shuffle_improvement'] * 100:.1f}%")

    for horizon, h in horizon_losses.items():
        print(
            f"horizon {horizon:2d}: "
            f"total={h['loss']:.6f} "
            f"agent={h['agent_loss']:.6f} "
            f"wrist={h['wrist_loss']:.6f} "
            f"task={h['task_loss']:.6f} "
            f"config={h['config_loss']:.6f}"
        )

    return {
        "metrics": val_metrics,
        "horizons": horizon_losses,
        "diagnostics": diagnostics,
        "train_dirs": train_dirs,
        "val_dirs": val_dirs,
    }


def discover_episodes(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Feature root missing: {root}")
    episodes = sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "metadata.json").is_file()
    )
    if not episodes:
        raise RuntimeError(f"No feature trajectories found in {root}")
    return episodes


def save_split_manifest(path, splits):
    manifest = {
        source: {
            partition: [str(p) for p in paths]
            for partition, paths in groups.items()
        }
        for source, groups in splits.items()
    }
    path.write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--pretrained-checkpoint", type=str, default=None,
        help="Override expert-only checkpoint path",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.no_wandb:
        config["no_wandb"] = True
    if args.pretrained_checkpoint:
        config["pretrained_checkpoint"] = args.pretrained_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("config:", config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the original exact episode splitting strategies:
    # expert: route-stratified 90/10; policies: success-stratified 80/20.
    expert_train, expert_val = split_trajectories(
        trajectory_dirs=discover_episodes(config["expert_root"]),
        train_ratio=config["train_val_split"],
        seed=config["split_seed"],
    )
    fm_train, fm_val = split_policy_episodes(
        trajectory_dirs=discover_episodes(config["fm_root"]),
        val_ratio=config["policy_val_ratio"],
        seed=config["split_seed"],
    )
    df_train, df_val = split_policy_episodes(
        trajectory_dirs=discover_episodes(config["diffusion_root"]),
        val_ratio=config["policy_val_ratio"],
        seed=config["split_seed"],
    )
    splits = {
        "expert": {"train": expert_train, "val": expert_val},
        "fm": {"train": fm_train, "val": fm_val},
        "diffusion": {"train": df_train, "val": df_val},
    }
    save_split_manifest(output_dir / "splits.json", splits)

    train_dirs = expert_train + fm_train + df_train
    if not all(splits[name]["val"] for name in splits):
        raise RuntimeError("Each source needs at least one held-out validation episode")
    train_dataset = WorldModelFeatureWindowDataset(
        trajectory_dirs=train_dirs, horizon=config["horizon"]
    )

    # Source-balanced sampling: 1/3 probability for each source.
    episode_sources = (
        ["expert"] * len(expert_train)
        + ["fm"] * len(fm_train)
        + ["diffusion"] * len(df_train)
    )
    window_sources = [episode_sources[int(trajectory_id)]
                      for trajectory_id, _ in train_dataset.index]
    counts = {name: window_sources.count(name) for name in splits}
    for name in splits:
        print(f"{name}: train episodes={len(splits[name]['train'])} "
              f"val episodes={len(splits[name]['val'])} "
              f"training windows={counts[name]}")
        if counts[name] == 0:
            raise RuntimeError(f"No usable training windows for {name}")

    sample_weights = torch.tensor(
        [1.0 / counts[name] for name in window_sources], dtype=torch.double
    )
    train_sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_dataset), replacement=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        sampler=train_sampler, num_workers=config["num_workers"],
        pin_memory=device.type == "cuda", drop_last=True,
    )
    val_datasets = {
        name: WorldModelFeatureWindowDataset(
            trajectory_dirs=partitions["val"], horizon=config["horizon"]
        ) for name, partitions in splits.items()
    }
    val_loaders = {
        name: DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=False,
            num_workers=config["num_workers"], pin_memory=device.type == "cuda",
        ) for name, dataset in val_datasets.items()
    }
    for name, ds in val_datasets.items():
        print(f"{name}: val windows={len(ds)}")
        if not len(ds):
            raise RuntimeError(f"No usable validation windows for {name}")

    model = LatentWorldModelVjepa(
        visual_feature_dim=config["visual_feature_dim"],
        latent_dim=config["latent_dim"],
        latent_grid_size=config["latent_grid_size"],
        task_state_dim=config["task_state_dim"],
        robot_config_dim=config["robot_config_dim"],
        action_dim=config["action_dim"],
        horizon=config["horizon"],
        num_dynamics_layers=config["num_dynamics_layers"],
        num_dynamics_heads=config["num_dynamics_heads"],
        dynamics_ff_dim=config["dynamics_ff_dim"],
        dropout=config["dropout"],
    ).to(device)

    checkpoint_path = Path(config["pretrained_checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Expert-only checkpoint not found: {checkpoint_path}")
    if checkpoint_path.resolve().parent == output_dir.resolve():
        raise RuntimeError("Expert checkpoint and output directory must differ")
    pretrained = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # strict=True restores both trainable weights and original normalization buffers.
    model.load_state_dict(pretrained["model_state_dict"], strict=True)
    print("Loaded expert model from:", checkpoint_path,
          "epoch:", pretrained.get("epoch"))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"], weight_decay=config["weight_decay"],
    )
    best_val_loss = float("inf")
    run = None
    if not config["no_wandb"]:
        wandb_config = dict(config)
        wandb_config["train_episodes"] = len(train_dirs)
        wandb_config["train_windows"] = len(train_dataset)
        wandb_config.update({f"val_{name}_windows": len(ds)
                             for name, ds in val_datasets.items()})
        run = wandb.init(project=config["wandb_project"],
                         name=config["wandb_name"], config=wandb_config)

    for epoch in range(config["epochs"]):
        start = time.time()
        train_metrics = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            device=device, grad_clip_norm=config["grad_clip_norm"],
        )
        validation_results = {}
        for name, loader in val_loaders.items():
            val_metrics, horizon_losses, diagnostics, route_metrics = validate(
                model=model, loader=loader, device=device,
            )
            validation_results[name] = {
                "metrics": val_metrics,
                "horizons": horizon_losses,
                "diagnostics": diagnostics,
                "routes": route_metrics,
            }
        elapsed = time.time() - start
        print(f"\nepoch {epoch:03d} train loss={train_metrics['loss']:.6f} "
              f"seconds={elapsed:.1f}")
        for name, result in validation_results.items():
            m = result["metrics"]
            d = result["diagnostics"]
            print(f"  {name:10s} loss={m['loss']:.6f} "
                  f"agent={m['agent_loss']:.6f} wrist={m['wrist_loss']:.6f} "
                  f"task={m['task_loss']:.6f} config={m['config_loss']:.6f} "
                  f"copy={d['copy']['loss']:.6f} "
                  f"shuffle={d['shuffle']['loss']:.6f}")
            for horizon, metrics in result["horizons"].items():
                print(f"    h{horizon:2d}: {metrics['loss']:.6f}")

        selection_loss = sum(validation_results[name]["metrics"]["loss"]
                             for name in splits) / len(splits)
        print(f"  selection loss: {selection_loss:.6f}")

        if run is not None:
            log_metrics = {"train/loss": train_metrics["loss"],
                           "val/selection_loss": selection_loss,
                           "system/epoch_seconds": elapsed}
            for key, value in train_metrics.items():
                log_metrics[f"train/{key}"] = value
            for name, result in validation_results.items():
                for key, value in result["metrics"].items():
                    log_metrics[f"val/{name}/{key}"] = value
                for key, value in result["diagnostics"].items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            log_metrics[f"val/{name}/{key}/{subkey}"] = subvalue
                    else:
                        log_metrics[f"val/{name}/{key}"] = value
                for horizon, h in result["horizons"].items():
                    for key, value in h.items():
                        log_metrics[f"val/{name}/horizon_{horizon}/{key}"] = value
            run.log(log_metrics, step=epoch)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_metrics["loss"],
            "val_loss": selection_loss,
            "validation_results": validation_results,
            "horizon": config["horizon"],
            "latent_dim": model.latent_dim,
            "latent_grid_size": model.latent_grid_size,
            "source_checkpoint": str(checkpoint_path),
        }
        torch.save(ckpt, output_dir / f"world_model_ep{epoch:03d}.pth")
        if selection_loss < best_val_loss:
            best_val_loss = selection_loss
            torch.save(ckpt, output_dir / "world_model_best.pth")
            print("  saved new best mixed checkpoint")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
