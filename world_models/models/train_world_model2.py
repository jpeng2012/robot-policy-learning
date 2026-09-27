from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import time

from world_models.data import WorldModelFeatureWindowDataset
from world_models.models import LatentWorldModel


DEFAULT_CONFIG = {
    # Data
    "data_root": "data/level3_wm_features",
    "train_val_split": 0.9,

    # Training
    "batch_size": 32,
    "epochs": 20,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "num_workers": 4,

    # Model
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
    "ema_momentum": 0.996,
    "freeze_vision": True,

    # Output
    "output_dir": "checkpoints/world_model",

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

        pred = predicated_future[:, index:index+1]

        target = target_future[:, index:index+1]

        losses = model.compute_latent_loss(pred, target)

        results[horizon] = losses["loss"]

    return results


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

        # ----------------------------------------------------
        # Update EMA target encoder AFTER optimizer step.
        # ----------------------------------------------------

        model.update_target_encoder()

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

    totals = {
        "loss": 0.0,
        "agent_loss": 0.0,
        "wrist_loss": 0.0,
        "task_loss": 0.0,
        "config_loss": 0.0,
    }
    num_batches = 0

    horizon_totals = {
        1: 0.0,
        2: 0.0,
        4: 0.0,
        8: 0.0,
        16: 0.0,
    }

    horizon_counts = {
        key: 0
        for key in horizon_totals
    }

    pbar = tqdm(loader, desc="Validating", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        output = model.forward_features(**batch)

        losses = model.compute_latent_loss(output["predicted_future"], output["target_future"])
        horizon_losses = compute_horizon_losses(model, output["predicted_future"], output["target_future"])

        for key in totals:
            totals[key] += losses[key].item()

        num_batches += 1

        running_loss = totals["loss"] / num_batches

        pbar.set_postfix(
            batch=f"{losses['loss'].item():.4f}",
            avg=f"{running_loss:.4f}",
        )

        for horizon, loss in horizon_losses.items():
            horizon_totals[horizon] += loss.item()
            horizon_counts[horizon] += 1
    
    mean_loss = {
        key: value / max(num_batches, 1) for key, value in totals.items() 
    }

    mean_horizon_losses = {}

    for horizon in horizon_totals:
        if horizon_counts[horizon]> 0:
            mean_horizon_losses[horizon] = horizon_totals[horizon] / horizon_counts[horizon]
    
    return mean_loss, mean_horizon_losses


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

    split = int(config["train_val_split"] * len(trajectory_dirs))

    train_dirs = trajectory_dirs[:split]

    val_dirs = trajectory_dirs[split:]

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

    model = LatentWorldModel(
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
        ema_momentum=config["ema_momentum"],
        freeze_vision=config["freeze_vision"],
    ).to(device)

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

        for horizon in sorted(
            horizon_losses
        ):
            print(
                f"  horizon {horizon:2d}: "
                f"{horizon_losses[horizon]:.6f}"
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

                "system/epoch_seconds":
                    epoch_seconds,
            }

            for horizon, loss in horizon_losses.items():
                metrics[f"val/horizon_{horizon}"] = loss

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