from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from world_models.models import LatentWorldModelVjepa

from world_models.models.cube_head import CubePositionHead
from world_models.models.train_cube_head import (
    CubeDataset,
    discover,
    split_episode_dirs,
    move,
    raw_future_latent,
)


def build_world_model(
    checkpoint_path,
    device,
):
    wm = LatentWorldModelVjepa(
        visual_feature_dim=768,
        latent_dim=384,
        latent_grid_size=4,
        task_state_dim=9,
        robot_config_dim=7,
        action_dim=7,
        horizon=16,
        num_dynamics_layers=6,
        num_dynamics_heads=6,
        dynamics_ff_dim=1536,
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    wm.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    wm.eval()

    return wm


@torch.no_grad()
def evaluate_cube_prediction(
    wm,
    cube_head,
    loader,
    cube_mean,
    cube_std,
    device,
    horizons=(1, 4, 8, 16),
):
    """
    Evaluate future cube xyz prediction.

    For each horizon h:

        error_h =
            || predicted_cube[t+h] - true_cube[t+h] ||_2

    Results are reported in centimeters.

    mean_cm:
        average Euclidean cube-position error.

    rmse_cm:
        emphasizes occasional larger prediction errors.

    H16 matters most initially because the reranker will score
    the predicted terminal state of each 16-step candidate chunk.
    """

    stats = {
        h: {
            "n": 0,
            "sum_error_cm": 0.0,
            "sum_squared_error_cm": 0.0,
        }
        for h in horizons
    }

    cube_head.eval()
    wm.eval()

    for batch in tqdm(
        loader,
        desc="cube eval",
        leave=False,
    ):
        batch = move(
            batch,
            device,
        )

        # ----------------------------------------------------
        # Frozen WM predicts future latent trajectory.
        #
        # Shape:
        #     [B, H, N, D]
        # ----------------------------------------------------

        future_latent = raw_future_latent(
            wm,
            batch,
        )

        # ----------------------------------------------------
        # Cube head predicts normalized xyz.
        #
        # Denormalize back to simulator coordinates (meters).
        # ----------------------------------------------------

        pred_cube = (
            cube_head(future_latent)
            * cube_std
            + cube_mean
        )

        true_cube = batch[
            "future_cube_pos"
        ]

        for h in horizons:
            i = h - 1

            # Euclidean xyz error in meters -> centimeters.
            error_cm = (
                torch.linalg.vector_norm(
                    pred_cube[:, i]
                    - true_cube[:, i],
                    dim=-1,
                )
                * 100.0
            )

            stats[h]["n"] += (
                error_cm.numel()
            )

            stats[h]["sum_error_cm"] += (
                error_cm.sum().item()
            )

            stats[h][
                "sum_squared_error_cm"
            ] += (
                (error_cm ** 2)
                .sum()
                .item()
            )

    results = {}

    for h in horizons:
        s = stats[h]

        n = s["n"]

        mean_cm = (
            s["sum_error_cm"]
            / n
        )

        rmse_cm = (
            s[
                "sum_squared_error_cm"
            ]
            / n
        ) ** 0.5

        results[h] = {
            "mean_cm": mean_cm,
            "rmse_cm": rmse_cm,
        }

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fm-root",
        default=(
            "data/"
            "policy_wm_features_vjepa21_fm"
        ),
    )

    parser.add_argument(
        "--df-root",
        default=(
            "data/"
            "policy_wm_features_vjepa21_df"
        ),
    )

    parser.add_argument(
        "--cube-head",
        default=(
            "checkpoints/"
            "world_model_cube_head/"
            "cube_head_best.pth"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device:",
        device,
    )

    # ========================================================
    # Load cube-head checkpoint.
    #
    # This checkpoint also remembers:
    #   - which WM checkpoint produced the latent
    #   - cube mean/std used during training
    # ========================================================

    cube_checkpoint = torch.load(
        args.cube_head,
        map_location=device,
        weights_only=False,
    )

    print(
        "cube head epoch:",
        cube_checkpoint["epoch"],
    )

    print(
        "cube head val loss:",
        cube_checkpoint["val_loss"],
    )

    wm_checkpoint = (
        cube_checkpoint[
            "wm_checkpoint"
        ]
    )

    wm = build_world_model(
        wm_checkpoint,
        device,
    )

    cube_head = CubePositionHead(
        latent_dim=384,
        hidden_dim=512,
        visual_tokens_per_camera=16,
    ).to(device)

    cube_head.load_state_dict(
        cube_checkpoint[
            "cube_head_state_dict"
        ]
    )

    cube_head.eval()

    cube_mean = (
        cube_checkpoint[
            "cube_mean"
        ]
        .to(device)
    )

    cube_std = (
        cube_checkpoint[
            "cube_std"
        ]
        .to(device)
    )

    # ========================================================
    # Evaluate FM and Diffusion separately.
    #
    # Use the same episode-level split logic / seed as training
    # so evaluation is on the held-out policy episodes.
    # ========================================================

    roots = {
        "FM": args.fm_root,
        "Diffusion": args.df_root,
    }

    for source_index, (
        source_name,
        root,
    ) in enumerate(
        roots.items()
    ):
        episode_dirs = discover(
            root
        )

        _, val_dirs = (
            split_episode_dirs(
                episode_dirs,
                val_ratio=0.2,
                seed=(
                    args.seed
                    + source_index
                ),
            )
        )

        dataset = CubeDataset(
            val_dirs,
            horizon=16,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        results = (
            evaluate_cube_prediction(
                wm=wm,
                cube_head=cube_head,
                loader=loader,
                cube_mean=cube_mean,
                cube_std=cube_std,
                device=device,
            )
        )

        print()
        print("=" * 60)
        print(source_name)
        print("=" * 60)

        for h in [
            1,
            4,
            8,
            16,
        ]:
            r = results[h]

            print(
                f"H={h:2d}  "
                f"mean={r['mean_cm']:.4f} cm  "
                f"rmse={r['rmse_cm']:.4f} cm"
            )


if __name__ == "__main__":
    main()