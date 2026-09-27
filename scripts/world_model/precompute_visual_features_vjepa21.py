from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from world_models.data import load_robosuite_trajectory


VJEPA_REPO = "facebookresearch/vjepa2"

VJEPA_MODEL_NAME = (
    "vjepa2_1_vit_base_384"
)

VJEPA_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/vjepa2/"
    "vjepa2_1_vitb_dist_vitG_384.pt"
)

IMAGE_SIZE = 384

PATCH_SIZE = 16

# 384 / 16 = 24
PATCH_GRID_SIZE = (
    IMAGE_SIZE // PATCH_SIZE
)

# Keep the same world-model token count
# as our previous ViT cache.
LATENT_GRID_SIZE = 4


def clean_backbone_state_dict(
    state_dict,
):
    """
    Match the cleanup used by the official
    V-JEPA torch hub loader.

    Example keys:

        module.backbone.blocks.0...
            ->
        blocks.0...
    """

    cleaned = {}

    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")

        cleaned[key] = value

    return cleaned


def load_vjepa_encoder(
    device,
    checkpoint_path=None,
):
    """
    Load frozen V-JEPA 2.1 ViT-B/16 384 encoder.

    We instantiate through torch.hub with
    pretrained=False and load the official
    checkpoint ourselves.

    This avoids depending on the checkpoint URL
    currently embedded in the upstream hub code.
    """

    print("loading V-JEPA 2.1 architecture...")

    encoder, _ = torch.hub.load(
        VJEPA_REPO,
        VJEPA_MODEL_NAME,
        pretrained=False,
    )

    if checkpoint_path is not None:

        print(
            "loading local checkpoint:",
            checkpoint_path,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    else:

        print(
            "loading official V-JEPA 2.1 "
            "ViT-B/16 checkpoint..."
        )

        checkpoint = (
            torch.hub.load_state_dict_from_url(
                VJEPA_CHECKPOINT_URL,
                map_location="cpu",
                check_hash=False,
            )
        )

    if "ema_encoder" not in checkpoint:
        raise KeyError(
            "Expected 'ema_encoder' in "
            "V-JEPA 2.1 checkpoint"
        )

    state_dict = (
        clean_backbone_state_dict(
            checkpoint[
                "ema_encoder"
            ]
        )
    )

    message = encoder.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "checkpoint load:",
        message,
    )

    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    encoder = encoder.to(
        device
    )

    print(
        "V-JEPA embed dim:",
        encoder.embed_dim,
    )

    return encoder


def load_vjepa_preprocessor():
    """
    Official V-JEPA preprocessing.

    For evaluation this performs approximately:

        resize short side
        center crop
        convert clip to tensor
        ImageNet normalization

    crop_size = 384.
    """

    processor = torch.hub.load(
        VJEPA_REPO,
        "vjepa2_preprocessor",
        crop_size=IMAGE_SIZE,
    )

    return processor


def preprocess_image(
    image_path,
    processor,
):
    """
    Convert one image into the input expected
    by V-JEPA 2.1.

    The official processor operates on clips,
    so a single image is treated as a one-frame
    clip.

    Output:

        [C, 1, 384, 384]
    """

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        frame = np.asarray(image)

    # processor expects a sequence of frames.
    # One-frame clip:
    #     [H, W, C]
    # ->
    #     [C, 1, 384, 384]

    processed = processor([frame])

    # Official preprocessor returns a list
    # of spatial views. We use one center view.
    processed = processed[0]

    return processed


@torch.no_grad()
def extract_pooled_features(
    encoder,
    image_paths,
    processor,
    device,
    batch_size,
    output_path,
    use_amp=True,
):
    """
    Extract frozen V-JEPA spatial features.
    Input image:
        [3, 1, 384, 384]
    V-JEPA patch tokens:
        [24*24, 768] = [576, 768]
    Spatial pooling:
        [24, 24, 768] -> [4, 4, 768]
    Saved output:
        [T, 16, 768]
    No trainable projection is applied.
    """

    num_images = len(image_paths)

    feature_dim = encoder.embed_dim

    num_output_tokens = LATENT_GRID_SIZE * LATENT_GRID_SIZE

    output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float16,
            shape=(num_images, num_output_tokens, feature_dim),
    )

    for start in range(0, num_images, batch_size):

        end = min(start + batch_size, num_images)

        clips = []
        for path in image_paths[start:end]:
            clip = preprocess_image(image_path=path, processor=processor)
            clips.append(clip)

        # Each item:
        #     [C, 1, 384, 384]
        # Stack:
        #     [B, C, 1, 384, 384]

        clips = torch.stack(clips, dim=0)
        clips = clips.to(device, non_blocking=True)

        amp_enabled = (use_amp and device.type == "cuda")

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            # -----------------------------------------------
            # V-JEPA image mode.
            #
            # Because temporal dimension is 1, the V-JEPA
            # 2.1 model uses its image patch embedding path.
            #
            # Output:
            #
            #     [B, 576, 768]
            # -----------------------------------------------

            patch_tokens = encoder(clips)

        if patch_tokens.ndim != 3:
            raise ValueError(
                "Expected V-JEPA output "
                f"[B, N, D], got "
                f"{patch_tokens.shape}"
            )

        B, N, D = (
            patch_tokens.shape
        )

        expected_tokens = PATCH_GRID_SIZE * PATCH_GRID_SIZE

        if N != expected_tokens:
            raise ValueError(
                "Unexpected V-JEPA token count. "
                f"Expected {expected_tokens}, "
                f"got {N}"
            )

        if D != feature_dim:
            raise ValueError(
                "Unexpected V-JEPA feature dim. "
                f"Expected {feature_dim}, "
                f"got {D}"
            )

        # -----------------------------------------------
        # [B, 576, 768] -> [B, 768, 24, 24]
        # -----------------------------------------------

        features = patch_tokens.reshape(B, PATCH_GRID_SIZE, PATCH_GRID_SIZE, D)

        features = features.permute(0, 3, 1, 2)

        # -----------------------------------------------
        # 24x24 -> 4x4
        # -----------------------------------------------

        features = F.adaptive_avg_pool2d(
                features.float(),
                output_size=(LATENT_GRID_SIZE, LATENT_GRID_SIZE),
            )

        # -----------------------------------------------
        # [B, 768, 4, 4] -> [B, 16, 768]
        # -----------------------------------------------

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
    processor,
    device,
    batch_size,
    use_amp,
    skip_existing,
):
    trajectory_file = Path(trajectory_file)

    print()
    print(
        "processing:",
        trajectory_file,
    )

    traj = load_robosuite_trajectory(trajectory_file)

    trajectory_dir = output_root / trajectory_file.stem

    meta_file = trajectory_dir / "metadata.json"

    trajectory_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    agent_output = trajectory_dir / "agent_features.npy"
    wrist_output = trajectory_dir / "wrist_features.npy"
    metadata_output = trajectory_dir / "metadata.json"

    if (
        skip_existing
        and agent_output.exists()
        and wrist_output.exists()
        and metadata_output.exists()
    ):
        print(
            "  complete feature cache exists; "
            "skipping"
        )
        return

    # ========================================================
    # Agent camera
    # ========================================================

    print("  agent camera")

    extract_pooled_features(
        encoder=encoder,
        image_paths=traj.agent_image_paths,
        processor=processor,
        device=device,
        batch_size=batch_size,
        output_path=agent_output,
        use_amp=use_amp,
    )

    # ========================================================
    # Wrist camera
    # ========================================================

    print("  wrist camera")

    extract_pooled_features(
        encoder=encoder,
        image_paths=traj.wrist_image_paths,
        processor=processor,
        device=device,
        batch_size=batch_size,
        output_path=wrist_output,
        use_amp=use_amp,
    )

    # ========================================================
    # Low-dimensional robot state
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
            traj.metadata.get("route"),

        "action_type":
            "eef_delta",

        "feature_encoder":
            "V-JEPA-2.1 ViT-B/16",

        "feature_encoder_hub_name":
            VJEPA_MODEL_NAME,

        "input_resolution":
            IMAGE_SIZE,

        "patch_size":
            PATCH_SIZE,

        "raw_patch_grid":
            PATCH_GRID_SIZE,

        "raw_patch_tokens":
            (
                PATCH_GRID_SIZE
                * PATCH_GRID_SIZE
            ),

        "feature_dim":
            encoder.embed_dim,

        "latent_grid_size":
            LATENT_GRID_SIZE,

        "visual_tokens":
            (
                LATENT_GRID_SIZE
                * LATENT_GRID_SIZE
            ),

        "feature_dtype":
            "float16",

        "frozen_encoder":
            True,
    }

    with metadata_output.open("w") as f:
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
        default="data/level3_wm_features_vjepa21_vitb384",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional local V-JEPA 2.1 ViT-B/16 checkpoint",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
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

    print(
        "output root:",
        args.output_root,
    )

    files = sorted(Path(args.data_root).glob("*.pkl"))

    if not files:
        raise RuntimeError("No trajectory files found")

    output_root = Path(args.output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Frozen V-JEPA 2.1
    # ========================================================

    encoder = load_vjepa_encoder(
        device=device,
        checkpoint_path=args.checkpoint,
    )

    processor = load_vjepa_preprocessor()

    print(
        "trajectories:",
        len(files),
    )

    print("feature cache:")

    print(
        "  encoder: "
        "V-JEPA 2.1 ViT-B/16"
    )

    print(
        "  input:   "
        "384x384"
    )

    print(
        "  patches: "
        "24x24 = 576"
    )

    print(
        "  pooled:  "
        "4x4 = 16"
    )

    print(
        "  dim:",
        encoder.embed_dim,
    )

    for trajectory_file in files:

        process_trajectory(
            trajectory_file=trajectory_file,
            output_root=output_root,
            encoder=encoder,
            processor=processor,
            device=device,
            batch_size=args.batch_size,
            use_amp=not args.no_amp,
            skip_existing=not args.no_skip_existing,
        )


if __name__ == "__main__":
    main()