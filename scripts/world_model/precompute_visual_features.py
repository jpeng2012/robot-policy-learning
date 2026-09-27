from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ViT_B_16_Weights

from world_models.data import load_robosuite_trajectory
from world_models.models import SpatialViTEncoder

@torch.no_grad()

def extract_pooled_features(
    encoder,
    image_paths,
    transform,
    device,
    batch_size,
    output_path,
):
    """
    Frozen ViT feature extraction.

    Image:
        [3, 224, 224]

    ViT patch tokens:
        [196, 768]

    Spatial pooling:
        [16, 768]

    We intentionally stop BEFORE the trainable
    768 -> 384 projection.
    """

    num_images = len(image_paths)

    # --------------------------------------------------------
    # Create .npy file directly on disk.
    #
    # This avoids keeping the entire trajectory feature tensor
    # in RAM.
    # --------------------------------------------------------

    grid_tokens = encoder.latent_grid_size * encoder.latent_grid_size

    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(num_images, grid_tokens, encoder.vit_dim),
    )

    for start in range(0, num_images, batch_size):
        end = min(start+batch_size, num_images)

        images = []

        for path in image_paths[start:end]:
            with Image.open(path) as image:
                image = image.convert('RGB')
                image = transform(image)
                images.append(image)

        images = torch.stack(
            images, dim=0
        ).to(device, non_blocking=True)

        # ----------------------------------------------------
        # Frozen ViT:
        #
        # [B, 3, 224, 224]
        #
        # ->
        #
        # [B, 196, 768]
        # ----------------------------------------------------

        patch_tokens = encoder.forward_backbone(images)

        B, N, D = patch_tokens.shape

        grid_size = int(N ** 0.5)

        if grid_size * grid_size != N:
            raise ValueError(
                f"Expected square patch grid, got N={N}"
            )

        features = patch_tokens.reshape(B, grid_size, grid_size, D)

        features = features.permute(0, 3, 1, 2)

        features = F.adaptive_avg_pool2d(
            features,
            output_size=(encoder.latent_grid_size, encoder.latent_grid_size),
        )

        features = features.flatten(2).transpose(1, 2)

        output[start:end] = features.cpu().numpy().astype(np.float16)

        print(
            f"    {end:5d} / "
            f"{num_images:5d}"
        )

        output.flush()

def process_trajectory(
        trajectory_file,
        output_root,
        encoder,
        transform,
        device,
        batch_size,
):
    trajectory_file = Path(trajectory_file)

    print()
    print(
        "processing:",
        trajectory_file
    )

    traj = load_robosuite_trajectory(trajectory_file)

    trajectory_dir = (output_root / trajectory_file.stem)

    meta_file = trajectory_dir / "metadata.json"

    if meta_file.exists():
        print("  features already exist, skipping")
        return

    trajectory_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    # ========================================================
    # Agent camera
    # ========================================================

    print("  agent camera")

    extract_pooled_features(
        encoder=encoder,
        image_paths=traj.agent_image_paths,
        transform=transform,
        device=device,
        batch_size=batch_size,
        output_path=(
            trajectory_dir
            / "agent_features.npy"
        ),
    )

    # ========================================================
    # Wrist camera
    # ========================================================

    print("  wrist camera")

    extract_pooled_features(
        encoder=encoder,
        image_paths=traj.wrist_image_paths,
        transform=transform,
        device=device,
        batch_size=batch_size,
        output_path=(
            trajectory_dir
            / "wrist_features.npy"
        ),
    )

    # ========================================================
    # Low-dimensional state
    # ========================================================
    np.save(
        trajectory_dir
        / "task_state.npy",
        traj.task_state,
    )

    np.save(
        trajectory_dir
        / "robot_config.npy",
        traj.robot_config,
    )

    np.save(
        trajectory_dir
        / "actions.npy",
        traj.actions,
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "source_file":
            str(trajectory_file),

        "length":
            traj.length,

        "route":
            traj.metadata.get(
                "route"
            ),

        "action_type":
            "eef_delta",

        "feature_dim":
            encoder.vit_dim,

        "visual_tokens":
            encoder.latent_grid_size * encoder.latent_grid_size,
    }

    with (trajectory_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/level3",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=(
            "data/level3_wm_features"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device:",
        device
    )

    files = sorted(
        Path(
            args.data_root
        ).glob(
            "*.pkl"
        )
    )

    if not files:

        raise RuntimeError(
            "No trajectory files found"
        )

    output_root = Path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # ViT
    # ========================================================

    transform = (
        ViT_B_16_Weights.DEFAULT
        .transforms()
    )

    encoder = SpatialViTEncoder(
        latent_dim=384,
        latent_grid_size=4,
        pretrained=True,
        freeze_backbone=True,
    ).to(
        device
    )

    encoder.eval()

    print(
        "trajectories:",
        len(files)
    )

    for trajectory_file in files:

        process_trajectory(
            trajectory_file=(
                trajectory_file
            ),
            output_root=(
                output_root
            ),
            encoder=encoder,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()