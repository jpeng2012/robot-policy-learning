from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
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
    "data_root": "data/level3_wm_features_vjepa21_vitb384",
    "train_val_split": 0.9,

    # Training
    "batch_size": 120,
    "epochs": 20,
    "lr": 3e-4,
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
    "output_dir": "checkpoints/world_model_vjepa",

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
    keys = [
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

    return {
        key:
            batch[key].to(
                device,
                non_blocking=True,
            )
        for key in keys
    }


def split_trajectories(
        trajectory_dirs,
        train_ratio=0.9,
        seed=42,
):
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


def compute_horizon_losses(
        model,
        predicated_future,
        target_future,
):
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

    task_values = np.concatenate(
        task_values,
        axis=0,
    )

    config_values = np.concatenate(
        config_values,
        axis=0,
    )

    task_mean = torch.tensor(
        task_values.mean(
            axis=0
        ),
        dtype=torch.float32,
    )

    task_std = torch.tensor(
        task_values.std(
            axis=0
        ),
        dtype=torch.float32,
    )

    config_mean = torch.tensor(
        config_values.mean(
            axis=0
        ),
        dtype=torch.float32,
    )

    config_std = torch.tensor(
        config_values.std(
            axis=0
        ),
        dtype=torch.float32,
    )

    return (
        task_mean,
        task_std,
        config_mean,
        config_std,
    )


def train_one_epoch(
        model,
        loader,
        optimizer,
        device,
        grad_clip_norm: float = 1.0,
):
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

        optimizer.zero_grad(set_to_none=True)

        output = model.forward_features(**batch)

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

    return {
        key: value / max(num_batches, 1) for key, value in totals.items() 
    }


@torch.no_grad()
def validate(
        model,
        loader,
        device,
):
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

    totals = { key: 0.0 for key in metric_keys }

    # ========================================================
    # Copy-current-state baseline
    #
    # Assume:
    #
    #     future == current
    #
    # without running the dynamics model.
    # ========================================================

    copy_totals = { key: 0.0 for key in metric_keys }

    # ========================================================
    # Shuffled-action baseline
    #
    # Same current observation, but wrong action sequence.
    # ========================================================

    shuffle_totals = { key: 0.0 for key in metric_keys }

    num_batches = 0

    horizons = [1, 2, 4, 8, 16]
   
    horizons = [ h for h in horizons if h <= model.horizon ]

    horizon_totals = {
        horizon: {key: 0.0 for key in metric_keys}
        for horizon in horizons
    }

    horizon_counts = { key: 0 for key in horizons }

    pbar = tqdm(loader, desc="Validating", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        # ====================================================
        # 1. Normal world-model prediction
        # ====================================================

        output = model.forward_features(**batch)

        predictions = output["predicted_future"]
        targets = output["target_future"]
        current_latent = output["current_latent"]

        losses = model.compute_latent_loss(predictions, targets)
        horizon_losses = compute_horizon_losses(model, predictions, targets)

        for key in metric_keys:
            totals[key] += losses[key].item()

        for horizon, h_losses in horizon_losses.items():
            for key in metric_keys:
                horizon_totals[horizon][key] += h_losses[key].item()

            horizon_counts[horizon] += 1

        
        # ====================================================
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
            "agent_features":
                copy_agent,
            "wrist_features":
                copy_wrist,
            "task_state":
                copy_task,
            "robot_config":
                copy_config,
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
    
    return mean_loss, mean_horizon_losses, diagnostics


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.no_wandb:
        config["no_wandb"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("config:", config)

    # ========================================================
    # Find trajectories
    # ========================================================

    trajectory_dirs = sorted(
        [path for path in Path(config["data_root"]).iterdir() if path.is_dir()]
    )

    if len(trajectory_dirs) < 2:
        raise RuntimeError(
            "Need at least two feature trajectories"
        )

    # ========================================================
    # Episode-level train / validation split
    # ========================================================
    (
        train_dirs,
        val_dirs,
    ) = split_trajectories(
        trajectory_dirs=trajectory_dirs,
        train_ratio=config["train_val_split"],
        seed=config["split_seed"],
    )

    print(
        "train trajectories:",
        len(train_dirs),
    )

    print(
        "val trajectories:",
        len(val_dirs),
    )


    # ========================================================
    # Datasets
    # ========================================================

    train_dataset = WorldModelFeatureWindowDataset(
            trajectory_dirs=train_dirs,
            horizon=config["horizon"],
    )

    val_dataset = WorldModelFeatureWindowDataset(
            trajectory_dirs=val_dirs,
            horizon=config["horizon"],
    )

    print(
        "train windows:",
        len(train_dataset),
    )

    print(
        "val windows:",
        len(val_dataset),
    )

    # ========================================================
    # DataLoaders
    # ========================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # ========================================================
    # Model
    # ========================================================

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

    (
        task_mean,
        task_std,
        config_mean,
        config_std,
    ) = compute_state_stats(train_dirs)

    model.set_state_normalization(
        task_mean=task_mean.to(device),
        task_std=task_std.to(device),
        config_mean=config_mean.to(device),
        config_std=config_std.to(device),
    )

    # ========================================================
    # Optimizer
    # ========================================================

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # ========================================================
    # Output directory
    # ========================================================

    output_dir = Path(config["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    run = None

    if not config["no_wandb"]:
        wandb_config = config.copy()
        wandb_config["train_trajectories"] = len(train_dirs)
        wandb_config["val_trajectories"] = len(val_dirs)
        wandb_config["train_windows"] = len(train_dataset)
        wandb_config["val_windows"] = len(val_dataset)

        run = wandb.init(
            project=config["wandb_project"],
            name=config["wandb_name"],
            config=wandb_config,
        )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=config["grad_clip_norm"],
        )

        (
            val_metrics,
            horizon_losses,
            diagnostics,
        ) = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        epoch_seconds = time.time() - epoch_start

        print()
        print(f"epoch {epoch:03d}")

        print(
            f"  train loss: "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"  val loss:   "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"  copy loss:     "
            f"{diagnostics['copy']['loss']:.6f}"
        )

        print(
            f"  shuffle loss:  "
            f"{diagnostics['shuffle']['loss']:.6f}"
        )

        print(
            f"  improvement vs copy: "
            f"{diagnostics['copy_improvement'] * 100:.1f}%"
        )

        print(
            f"  improvement vs shuffle: "
            f"{diagnostics['shuffle_improvement'] * 100:.1f}%"
        )

        for horizon in sorted(horizon_losses):
            h = horizon_losses[horizon]

            print(
                f"  horizon {horizon:2d}: "
                f"total={h['loss']:.6f} "
                f"agent={h['agent_loss']:.6f} "
                f"wrist={h['wrist_loss']:.6f} "
                f"task={h['task_loss']:.6f} "
                f"config={h['config_loss']:.6f}"
            )

        if run is not None:

            metrics = {
                "epoch":
                    epoch,

                "train/loss":
                    train_metrics["loss"],

                "train/agent":
                    train_metrics["agent_loss"],

                "train/wrist":
                    train_metrics["wrist_loss"],

                "train/task_state":
                    train_metrics["task_loss"],

                "train/robot_config":
                    train_metrics["config_loss"],

                "val/loss":
                    val_metrics["loss"],

                "val/agent":
                    val_metrics["agent_loss"],

                "val/wrist":
                    val_metrics["wrist_loss"],

                "val/task_state":
                    val_metrics["task_loss"],

                "val/robot_config":
                    val_metrics["config_loss"],

                "diagnostic/copy_loss":
                    diagnostics["copy"]["loss"],

                "diagnostic/shuffle_loss":
                    diagnostics["shuffle"]["loss"],

                "diagnostic/copy_improvement":
                    diagnostics["copy_improvement"],

                "diagnostic/shuffle_improvement":
                    diagnostics["shuffle_improvement"],

                "system/epoch_seconds":
                    epoch_seconds,
            }

            for name in [
                "agent_loss",
                "wrist_loss",
                "task_loss",
                "config_loss",
            ]:
                metrics[f"diagnostic/copy/{name}"] = diagnostics["copy"][name]
                metrics[f"diagnostic/shuffle/{name}"] = diagnostics["shuffle"][name]

            for horizon, h_losses in horizon_losses.items():
                metrics[f"val/horizon_{horizon}"] = h_losses["loss"]
                metrics[f"val/horizon_{horizon}/agent"] = h_losses["agent_loss"]
                metrics[f"val/horizon_{horizon}/wrist"] = h_losses["wrist_loss"]
                metrics[f"val/horizon_{horizon}/task"] = h_losses["task_loss"]
                metrics[f"val/horizon_{horizon}/config"] = h_losses["config_loss"]

            run.log(metrics, step=epoch,)

        # ====================================================
        # Save current checkpoint
        # ====================================================

        checkpoint = {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_metrics["loss"],

            "val_loss":
                val_metrics["loss"],

            "horizon_losses":
                horizon_losses,

            "horizon":
                config["horizon"],

            "latent_dim":
                model.latent_dim,

            "latent_grid_size":
                model.latent_grid_size,
        }

        torch.save(
            checkpoint,
            output_dir
            / f"world_model_ep{epoch:03d}.pth",
        )

        # ====================================================
        # Save best validation checkpoint
        # ====================================================

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]

            torch.save(
                checkpoint,
                output_dir
                / "world_model_best.pth",
            )

            print(
                "  saved new best checkpoint"
            )

    if run is not None:
        run.finish()

if __name__ == "__main__":
    main()