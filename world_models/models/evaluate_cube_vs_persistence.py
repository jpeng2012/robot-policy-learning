
from __future__ import annotations

"""
Compare the learned cube predictor against a trivial persistence baseline.

Persistence baseline:
    predict cube[t+h] = cube[t]

Why:
    If many windows occur before grasp / transport, the cube barely moves.
    Then a trivial "cube stays fixed" predictor can look deceptively good.

We therefore report:
    1) all held-out windows
    2) moving windows only

Moving window:
    ||cube[t+16] - cube[t]|| > moving_threshold_cm

Key result:
    On moving windows, WM cube prediction should beat persistence clearly.
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from world_models.data import WorldModelFeatureWindowDataset
from world_models.models import LatentWorldModelVjepa

from world_models.models.cube_head import CubePositionHead


class CubeEvalDataset(Dataset):
    """
    Wrap the existing feature-window dataset and add:

        current_cube_pos: [3]
        future_cube_pos:  [H, 3]
    """

    def __init__(self, trajectory_dirs, horizon=16):
        self.base = WorldModelFeatureWindowDataset(
            trajectory_dirs=trajectory_dirs,
            horizon=horizon,
        )
        self.horizon = horizon

        self.cubes = []

        for p in trajectory_dirs:
            cube_path = Path(p) / "cube_pos.npy"

            if not cube_path.exists():
                raise FileNotFoundError(cube_path)

            self.cubes.append(
                np.load(
                    cube_path,
                    mmap_mode="r",
                )
            )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        trajectory_id = int(
            sample["trajectory_id"]
        )

        t = int(
            sample["t"]
        )

        cube = self.cubes[
            trajectory_id
        ]

        current_cube = np.asarray(
            cube[t]
        ).copy()

        future_cube = np.asarray(
            cube[
                t + 1:
                t + self.horizon + 1
            ]
        ).copy()

        sample["current_cube_pos"] = (
            torch.from_numpy(
                current_cube
            ).float()
        )

        sample["future_cube_pos"] = (
            torch.from_numpy(
                future_cube
            ).float()
        )

        return sample


def discover(root):
    root = Path(root)

    return sorted(
        p for p in root.iterdir()
        if p.is_dir()
        and (p / "metadata.json").exists()
    )


def split_episode_dirs(
    dirs,
    val_ratio=0.2,
    seed=42,
):
    dirs = list(dirs)

    rng = random.Random(seed)
    rng.shuffle(dirs)

    n_val = max(
        1,
        round(len(dirs) * val_ratio),
    )

    n_val = min(
        n_val,
        len(dirs) - 1,
    )

    return (
        dirs[n_val:],
        dirs[:n_val],
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


def move_batch(
    batch,
    device,
):
    tensor_keys = [
        "agent_features",
        "wrist_features",
        "task_state",
        "robot_config",
        "actions",
        "current_cube_pos",
        "future_cube_pos",
    ]

    return {
        k: batch[k].to(
            device,
            non_blocking=True,
        )
        for k in tensor_keys
    }


def raw_future_latent(
    wm,
    batch,
):
    """
    Obtain the raw action-conditioned future latent from the frozen WM.
    """

    current_latent = (
        wm.encode_current_features(
            agent_features=
                batch["agent_features"],
            wrist_features=
                batch["wrist_features"],
            task_state=
                batch["task_state"],
            robot_config=
                batch["robot_config"],
        )
    )

    future_latent = wm.dynamics(
        state_tokens=current_latent,
        actions=batch["actions"],
    )

    return future_latent


@torch.no_grad()
def evaluate(
    wm,
    cube_head,
    loader,
    cube_mean,
    cube_std,
    device,
    moving_threshold_cm=1.0,
    horizons=(1, 4, 8, 16),
):
    """
    For each horizon h, compare:

        WM:
            ||pred_cube[t+h] - true_cube[t+h]||

        persistence:
            ||cube[t] - true_cube[t+h]||

    Both errors are reported in cm.
    """

    stats = {}

    for group in [
        "all",
        "moving",
    ]:
        stats[group] = {}

        for h in horizons:
            stats[group][h] = {
                "n": 0,
                "wm_sum": 0.0,
                "wm_sq_sum": 0.0,
                "persist_sum": 0.0,
                "persist_sq_sum": 0.0,
            }

    total_windows = 0
    moving_windows = 0

    for batch in tqdm(
        loader,
        desc="cube vs persistence",
        leave=False,
    ):
        batch = move_batch(
            batch,
            device,
        )

        future_latent = (
            raw_future_latent(
                wm,
                batch,
            )
        )

        # Cube head predicts normalized xyz.
        # Convert back to meters.
        pred_cube = (
            cube_head(
                future_latent
            )
            * cube_std
            + cube_mean
        )

        current_cube = (
            batch[
                "current_cube_pos"
            ]
        )

        true_future = (
            batch[
                "future_cube_pos"
            ]
        )

        # ----------------------------------------------------
        # Define moving windows using TRUE H16 displacement.
        # ----------------------------------------------------

        true_h16 = (
            true_future[
                :,
                15,
                :,
            ]
        )

        h16_motion_cm = (
            torch.linalg.vector_norm(
                true_h16
                - current_cube,
                dim=-1,
            )
            * 100.0
        )

        moving_mask = (
            h16_motion_cm
            > moving_threshold_cm
        )

        total_windows += (
            h16_motion_cm.numel()
        )

        moving_windows += int(
            moving_mask.sum().item()
        )

        for h in horizons:
            i = h - 1

            true_h = (
                true_future[
                    :,
                    i,
                    :,
                ]
            )

            pred_h = (
                pred_cube[
                    :,
                    i,
                    :,
                ]
            )

            # WM prediction error.
            wm_error_cm = (
                torch.linalg.vector_norm(
                    pred_h
                    - true_h,
                    dim=-1,
                )
                * 100.0
            )

            # Persistence baseline:
            # assume cube stays at its current location.
            persist_error_cm = (
                torch.linalg.vector_norm(
                    current_cube
                    - true_h,
                    dim=-1,
                )
                * 100.0
            )

            # -------------------------------
            # All windows
            # -------------------------------

            a = stats[
                "all"
            ][h]

            a["n"] += (
                wm_error_cm.numel()
            )

            a["wm_sum"] += (
                wm_error_cm.sum().item()
            )

            a["wm_sq_sum"] += (
                (wm_error_cm ** 2)
                .sum()
                .item()
            )

            a["persist_sum"] += (
                persist_error_cm
                .sum()
                .item()
            )

            a["persist_sq_sum"] += (
                (persist_error_cm ** 2)
                .sum()
                .item()
            )

            # -------------------------------
            # Moving windows only
            # -------------------------------

            if moving_mask.any():
                wm_m = (
                    wm_error_cm[
                        moving_mask
                    ]
                )

                persist_m = (
                    persist_error_cm[
                        moving_mask
                    ]
                )

                m = stats[
                    "moving"
                ][h]

                m["n"] += (
                    wm_m.numel()
                )

                m["wm_sum"] += (
                    wm_m.sum().item()
                )

                m["wm_sq_sum"] += (
                    (wm_m ** 2)
                    .sum()
                    .item()
                )

                m["persist_sum"] += (
                    persist_m.sum().item()
                )

                m["persist_sq_sum"] += (
                    (persist_m ** 2)
                    .sum()
                    .item()
                )

    result = {
        "moving_fraction":
            moving_windows
            / max(
                total_windows,
                1,
            ),
        "groups": {},
    }

    for group in [
        "all",
        "moving",
    ]:
        result[
            "groups"
        ][group] = {}

        for h in horizons:
            s = stats[
                group
            ][h]

            n = s["n"]

            if n == 0:
                result[
                    "groups"
                ][group][h] = None
                continue

            wm_mean = (
                s["wm_sum"]
                / n
            )

            wm_rmse = math.sqrt(
                s["wm_sq_sum"]
                / n
            )

            persist_mean = (
                s["persist_sum"]
                / n
            )

            persist_rmse = (
                math.sqrt(
                    s[
                        "persist_sq_sum"
                    ]
                    / n
                )
            )

            result[
                "groups"
            ][group][h] = {
                "n": n,
                "wm_mean_cm":
                    wm_mean,
                "wm_rmse_cm":
                    wm_rmse,
                "persist_mean_cm":
                    persist_mean,
                "persist_rmse_cm":
                    persist_rmse,

                # > 1 means WM beats persistence.
                # Example: 4.0 means persistence error is 4x larger.
                "persist_over_wm":
                    persist_mean
                    / max(
                        wm_mean,
                        1e-8,
                    ),
            }

    return result


def print_result(
    source_name,
    result,
):
    print()
    print("=" * 80)
    print(source_name)
    print("=" * 80)

    print(
        "moving fraction "
        "(true H16 displacement > threshold): "
        f"{100.0 * result['moving_fraction']:.2f}%"
    )

    for group in [
        "all",
        "moving",
    ]:
        print()
        print(
            f"[{group.upper()} WINDOWS]"
        )

        for h in [
            1,
            4,
            8,
            16,
        ]:
            r = (
                result[
                    "groups"
                ][group][h]
            )

            if r is None:
                print(
                    f"H={h:2d}: no samples"
                )
                continue

            print(
                f"H={h:2d}  "
                f"WM={r['wm_mean_cm']:.4f} cm  "
                f"PERSIST={r['persist_mean_cm']:.4f} cm  "
                f"persist/WM={r['persist_over_wm']:.2f}x  "
                f"(n={r['n']})"
            )


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
        "--moving-threshold-cm",
        type=float,
        default=1.0,
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
        "cube head val:",
        cube_checkpoint["val_loss"],
    )

    wm = build_world_model(
        cube_checkpoint[
            "wm_checkpoint"
        ],
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
        dirs = discover(
            root
        )

        _, val_dirs = (
            split_episode_dirs(
                dirs,
                val_ratio=0.2,
                seed=(
                    args.seed
                    + source_index
                ),
            )
        )

        dataset = CubeEvalDataset(
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

        result = evaluate(
            wm=wm,
            cube_head=cube_head,
            loader=loader,
            cube_mean=cube_mean,
            cube_std=cube_std,
            device=device,
            moving_threshold_cm=
                args.moving_threshold_cm,
        )

        print_result(
            source_name,
            result,
        )


if __name__ == "__main__":
    main()
