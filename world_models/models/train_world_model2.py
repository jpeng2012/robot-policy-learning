from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
# from torchvision.models import ViT_B_16_Weights
from tqdm import tqdm
import wandb

from world_models.data import WorldModelFeatureWindowDataset
from world_models.models import LatentWorldModel


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

    horzions = [1, 2, 4, 8, 16]

    for horzion in horzions:
        if horzion > model.horzion:
            continue

        index = horzion + 1

        pred = predicated_future[:, index:index+1]

        target = target_future[:, index, index+1]

        losses = model.compute_latent_loss(pred, target)

        results[
            horzion
        ] = losses["loss"]

    return results


def train_one_epoch(
        model,
        loader,
        optimizer,
        device,
):
    model.train()

    total_loss = 0.0
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
            max_norm=1.0,
        )

        optimizer.step()

        # ----------------------------------------------------
        # Update EMA target encoder AFTER optimizer step.
        # ----------------------------------------------------

        model.update_target_encoder()

        total_loss += loss.item()

        num_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
        model,
        loader,
        device,
):
    model.eval()

    total_loss = 0.0
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
        horizon_losses = compute_horizon_losses(output["predicted_future"], output["target_future"])

        total_loss += losses['loss'].item()

        num_batches += 1

        pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")

        for horizon, loss in horizon_losses.items():
            horizon_totals[horizon] += loss.item()
            horizon_counts[horizon] += 1


    
    
    mean_loss = total_loss / max(num_batches, 1)

    mean_horizon_losses = {}

    for horizon in horizon_totals:

        if horizon_counts[horizon]> 0:
            mean_horizon_losses[horizon] = horizon_totals[horizon] / horizon_counts[horizon]
    

    return mean_loss, mean_horizon_losses


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/level3_wm_features",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/world_model",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="robot-world-model",
    )

    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
    )

    args = parser.parse_args()

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

    print( "device: ", device)

    # ========================================================
    # Find trajectories
    # ========================================================

    trajectory_dirs = sorted(
        [ path for path in Path(args.data_root).iterdir() if path.is_dir() ]
    )

    if len(trajectory_dirs) < 2:
        raise RuntimeError(
            "Need at least two feature trajectories"
        )

    # ========================================================
    # Episode-level train / validation split
    # ========================================================

    split = int(0.9*len(trajectory_dirs))

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
            horizon=args.horizon,
    )

    val_dataset = WorldModelFeatureWindowDataset(
            trajectory_dirs=val_dirs,
            horizon=args.horizon,
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ========================================================
    # Model
    # ========================================================

    model = LatentWorldModel(
        latent_dim=384,
        latent_grid_size=4,
        task_state_dim=9,
        robot_config_dim=7,
        action_dim=7,
         horizon=args.horizon,
        num_dynamics_layers=6,
        num_dynamics_heads=6,
        dynamics_ff_dim=1536,
        dropout=0.1,
        ema_momentum=0.996,
        freeze_vision=True,
    ).to(device)

    # ========================================================
    # Optimizer
    # ========================================================

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=1e-4
    )

    # ========================================================
    # Output directory
    # ========================================================

    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    run = None
    
    if not args.no_wandb:

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "batch_size":
                    args.batch_size,

                "epochs":
                    args.epochs,

                "learning_rate":
                    args.lr,

                "horizon":
                    args.horizon,

                "latent_dim":
                    384,

                "latent_grid_size":
                    4,

                "num_state_tokens":
                    34,

                "dynamics_layers":
                    6,

                "dynamics_heads":
                    6,

                "dynamics_ff_dim":
                    1536,

                "ema_momentum":
                    0.996,

                "freeze_vision":
                    True,

                "feature_cache":
                    True,

                "train_trajectories":
                    len(train_dirs),

                "val_trajectories":
                    len(val_dirs),

                "train_windows":
                    len(train_dataset),

                "val_windows":
                    len(val_dataset),
            },
        )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        (
            val_loss,
            horizon_losses,
        ) = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        print()
        print(f"epoch {epoch:03d}")

        print(
            f"  train loss: "
            f"{train_loss:.6f}"
        )

        print(
            f"  val loss:   "
            f"{val_loss:.6f}"
        )

        for horizon in sorted(
            horizon_losses
        ):
            print(
                f"  horizon {horizon:2d}: "
                f"{horizon_losses[horizon]:.6f}"
            )

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
                train_loss,

            "val_loss":
                val_loss,

            "horizon_losses":
                horizon_losses,

            "horizon":
                args.horizon,

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

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                checkpoint,
                output_dir
                / "world_model_best.pth",
            )

            print(
                "  saved new best checkpoint"
            )


if __name__ == "__main__":
    main()