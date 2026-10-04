from __future__ import annotations

"""
Train cube-position decoder using FM + Diffusion feature caches only.

Why policy-only:
    - The reranker will be used on FM / Diffusion rollouts.
    - FM / DF caches already contain cube_pos.npy.
    - Cube prediction is supervised dynamics, so BOTH successful and failed
      policy trajectories are useful.

The mixed WM remains frozen. Only CubePositionHead is trained.
"""

import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm

from world_models.data import WorldModelFeatureWindowDataset
from world_models.models import LatentWorldModelVjepa
from world_models.models.cube_head import CubePositionHead


class CubeDataset(Dataset):
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
            self.cubes.append(np.load(cube_path, mmap_mode="r"))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        trajectory_id = int(sample["trajectory_id"])
        t = int(sample["t"])

        cube = self.cubes[trajectory_id]
        future = np.asarray(
            cube[t + 1 : t + self.horizon + 1]
        ).copy()

        sample["future_cube_pos"] = torch.from_numpy(
            future
        ).float()

        return sample


def discover(root):
    root = Path(root)
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "metadata.json").exists()
    )


def split_episode_dirs(dirs, val_ratio=0.2, seed=42):
    dirs = list(dirs)
    rng = random.Random(seed)
    rng.shuffle(dirs)

    n_val = max(1, round(len(dirs) * val_ratio))
    n_val = min(n_val, len(dirs) - 1)

    return dirs[n_val:], dirs[:n_val]


def build_wm(device, checkpoint):
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

    ckpt = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    wm.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    for p in wm.parameters():
        p.requires_grad = False

    wm.eval()

    print(
        f"Loaded frozen mixed WM: {checkpoint} "
        f"(epoch={ckpt.get('epoch','unknown')})"
    )

    return wm


def raw_future_latent(wm, batch):
    """
    Get raw action-conditioned future latent from the frozen dynamics model.
    """
    current = wm.encode_current_features(
        agent_features=batch["agent_features"],
        wrist_features=batch["wrist_features"],
        task_state=batch["task_state"],
        robot_config=batch["robot_config"],
    )

    return wm.dynamics(
        state_tokens=current,
        actions=batch["actions"],
    )


def compute_cube_stats(datasets):
    total = 0
    s = torch.zeros(3, dtype=torch.float64)
    ss = torch.zeros(3, dtype=torch.float64)

    for ds in datasets:
        for cube in ds.cubes:
            x = torch.from_numpy(
                np.asarray(cube).copy()
            ).double()

            total += len(x)
            s += x.sum(0)
            ss += (x * x).sum(0)

    mean = s / total
    var = ss / total - mean * mean

    std = torch.sqrt(
        torch.clamp(var, min=1e-10)
    )

    return mean.float(), std.float()


def move(batch, device):
    keys = [
        "agent_features",
        "wrist_features",
        "task_state",
        "robot_config",
        "actions",
        "future_cube_pos",
    ]

    return {
        k: batch[k].to(device, non_blocking=True)
        for k in keys
    }


@torch.no_grad()
def evaluate(wm, head, loader, cube_mean, cube_std, device):
    head.eval()

    total = 0
    squared_error = 0.0

    for batch in loader:
        batch = move(batch, device)

        latent = raw_future_latent(
            wm,
            batch,
        )

        pred_n = head(latent)

        true_n = (
            batch["future_cube_pos"]
            - cube_mean
        ) / cube_std

        loss = F.mse_loss(
            pred_n,
            true_n,
            reduction="sum",
        )

        squared_error += loss.item()
        total += true_n.numel()

    return squared_error / total


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--fm-root",
        default="data/policy_wm_features_vjepa21_fm",
    )
    ap.add_argument(
        "--df-root",
        default="data/policy_wm_features_vjepa21_df",
    )
    ap.add_argument(
        "--wm-checkpoint",
        default="checkpoints/world_model_vjepa_mixed/world_model_best.pth",
    )
    ap.add_argument(
        "--output",
        default="checkpoints/world_model_cube_head/cube_head_best.pth",
    )
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)

    roots = {
        "fm": args.fm_root,
        "diffusion": args.df_root,
    }

    train_sets = []
    val_sets = []
    split_info = {}

    for i, (name, root) in enumerate(roots.items()):
        dirs = discover(root)

        train_dirs, val_dirs = split_episode_dirs(
            dirs,
            val_ratio=0.2,
            seed=args.seed + i,
        )

        train_ds = CubeDataset(
            train_dirs,
            horizon=16,
        )
        val_ds = CubeDataset(
            val_dirs,
            horizon=16,
        )

        train_sets.append(train_ds)
        val_sets.append(val_ds)

        split_info[name] = {
            "episodes": len(dirs),
            "train_episodes": len(train_dirs),
            "val_episodes": len(val_dirs),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
        }

    print(json.dumps(split_info, indent=2))

    cube_mean, cube_std = compute_cube_stats(
        train_sets
    )

    print("cube mean:", cube_mean.tolist())
    print("cube std:", cube_std.tolist())

    train_data = ConcatDataset(train_sets)
    val_data = ConcatDataset(val_sets)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    wm = build_wm(
        device,
        args.wm_checkpoint,
    )

    head = CubePositionHead(
        latent_dim=384,
        hidden_dim=512,
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    cube_mean = cube_mean.to(device)
    cube_std = cube_std.to(device)

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        head.train()

        total = 0
        running = 0.0

        for batch in tqdm(
            train_loader,
            desc=f"epoch {epoch}",
        ):
            batch = move(batch, device)

            with torch.no_grad():
                latent = raw_future_latent(
                    wm,
                    batch,
                )

            pred_n = head(latent)

            true_n = (
                batch["future_cube_pos"]
                - cube_mean
            ) / cube_std

            loss = F.mse_loss(
                pred_n,
                true_n,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                head.parameters(),
                1.0,
            )

            optimizer.step()

            bs = batch["future_cube_pos"].shape[0]
            running += loss.item() * bs
            total += bs

        train_loss = running / total

        val_loss = evaluate(
            wm,
            head,
            val_loader,
            cube_mean,
            cube_std,
            device,
        )

        print(
            f"epoch={epoch:02d} "
            f"train={train_loss:.6f} "
            f"val={val_loss:.6f}"
        )

        if val_loss < best:
            best = val_loss

            torch.save(
                {
                    "cube_head_state_dict":
                        head.state_dict(),
                    "cube_mean":
                        cube_mean.detach().cpu(),
                    "cube_std":
                        cube_std.detach().cpu(),
                    "wm_checkpoint":
                        args.wm_checkpoint,
                    "epoch":
                        epoch,
                    "val_loss":
                        val_loss,
                    "split_info":
                        split_info,
                },
                output,
            )

            print("saved best:", output)

    print("best val:", best)


if __name__ == "__main__":
    main()
