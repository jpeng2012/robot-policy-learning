
from __future__ import annotations

"""
Evaluate Flow Matching with and without World-Model reranking.

Fair comparison:
    baseline:
        sample 1 FM chunk
        execute first 4 actions

    random_k:
        sample K=8 FM chunks
        choose one uniformly at random
        execute first 4 actions

    rerank:
        sample K=8 FM chunks
        clean all chunks
        imagine H=16 futures with frozen WM
        score terminal predicted states with success cost-to-go retrieval
        execute first 4 actions of the best chunk

Important:
    Run BOTH modes from this script with the same --seed and --episodes.
    Do not compare against an older baseline that used a different
    execution horizon.

Example:
    python evaluate_fm_wm_reranker.py --mode baseline --episodes 100
    python evaluate_fm_wm_reranker.py --mode rerank   --episodes 100
"""

import argparse
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from robosuite.utils.placement_samplers import UniformRandomSampler

from envs.constrained_pick_place import ConstrainedPickPlace

from policies.common.observation_encoder_spatial import ObservationEncoder
from policies.flow_matching.model import FlowMatchingPolicy

from world_models.models import LatentWorldModelVjepa

from cube_head import CubePositionHead

from fm_wm_reranker_core import (
    SuccessCostToGoScorer,
    build_descriptor_batch,
    clean_action_chunks,
)


# ============================================================
# V-JEPA config
# ============================================================

VJEPA_REPO = "facebookresearch/vjepa2"
VJEPA_MODEL_NAME = "vjepa2_1_vit_base_384"
VJEPA_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/vjepa2/"
    "vjepa2_1_vitb_dist_vitG_384.pt"
)

IMAGE_SIZE = 384
PATCH_GRID_SIZE = 24
LATENT_GRID_SIZE = 4


# ============================================================
# Environment helpers
# ============================================================

def sample_target(cube_pos):
    """
    Match the training target distribution.
    """
    target_x = np.random.uniform(
        -0.08,
        0.08,
    )

    target_y = np.random.uniform(
        0.27,
        0.32,
    )

    return np.array(
        [
            target_x,
            target_y,
            cube_pos[2],
        ],
        dtype=np.float32,
    )


def make_policy_proprio(obs):
    """
    FM policy proprio:
        joints 7
        eef xyz 3
        gripper qpos 2
        = 12
    """
    return np.concatenate(
        [
            obs["robot0_joint_pos"],
            obs["robot0_eef_pos"],
            obs["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)


def make_wm_task_state(obs):
    """
    World-model task_state:
        eef xyz      3
        eef quat     4
        gripper qpos 2
        = 9
    """
    return np.concatenate(
        [
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
            obs["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)


def make_wm_robot_config(obs):
    """
    World-model robot_config:
        Panda joint positions = 7
    """
    return np.asarray(
        obs["robot0_joint_pos"],
        dtype=np.float32,
    )


# ============================================================
# FM image preprocessing
# ============================================================

_resnet_weights = ResNet18_Weights.DEFAULT
_resnet_transform = _resnet_weights.transforms()


def process_policy_image(image_np):
    """
    Match the live preprocessing used by the existing FM rollout.

    Robosuite's camera output is vertically flipped relative to
    the JPG orientation used in the training dataset.
    """
    image_np = np.flipud(
        image_np
    ).copy()

    image = Image.fromarray(
        image_np
    ).convert("RGB")

    return _resnet_transform(
        image
    )


# ============================================================
# V-JEPA loading / live feature extraction
# ============================================================

def clean_backbone_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        key = key.replace(
            "module.",
            "",
        )
        key = key.replace(
            "backbone.",
            "",
        )

        cleaned[key] = value

    return cleaned


def load_vjepa_encoder(
    device,
    checkpoint_path=None,
):
    """
    Same V-JEPA 2.1 ViT-B/16 backbone used to build the WM cache.
    """
    print(
        "loading V-JEPA 2.1 architecture..."
    )

    encoder, _ = torch.hub.load(
        VJEPA_REPO,
        VJEPA_MODEL_NAME,
        pretrained=False,
    )

    if checkpoint_path is None:
        checkpoint = (
            torch.hub.load_state_dict_from_url(
                VJEPA_CHECKPOINT_URL,
                map_location="cpu",
                check_hash=False,
            )
        )
    else:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    state_dict = clean_backbone_state_dict(
        checkpoint["ema_encoder"]
    )

    encoder.load_state_dict(
        state_dict,
        strict=True,
    )

    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    return encoder.to(
        device
    )


def load_vjepa_preprocessor():
    return torch.hub.load(
        VJEPA_REPO,
        "vjepa2_preprocessor",
        crop_size=IMAGE_SIZE,
    )


def preprocess_vjepa_frame(
    image_np,
    processor,
):
    """
    Match the orientation used for the saved JPGs / WM cache.

    Output:
        [C,1,384,384]
    """
    frame = np.flipud(
        image_np
    ).copy()

    processed = processor(
        [frame]
    )

    # Official processor returns a list of spatial views.
    return processed[0]


@torch.no_grad()
def extract_vjepa_features(
    encoder,
    processor,
    agent_image,
    wrist_image,
    device,
):
    """
    Two live RGB views -> frozen V-JEPA features.

    Raw:
        [B=2,576,768]

    Spatial pool:
        24x24 -> 4x4

    Return:
        agent [1,16,768]
        wrist [1,16,768]
    """
    clips = torch.stack(
        [
            preprocess_vjepa_frame(
                agent_image,
                processor,
            ),
            preprocess_vjepa_frame(
                wrist_image,
                processor,
            ),
        ],
        dim=0,
    ).to(
        device,
        non_blocking=True,
    )

    amp_enabled = (
        device.type == "cuda"
    )

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp_enabled,
    ):
        patch_tokens = encoder(
            clips
        )

    if patch_tokens.shape[1] != 576:
        raise ValueError(
            "Expected V-JEPA 576 spatial tokens, "
            f"got {patch_tokens.shape}"
        )

    B, N, D = patch_tokens.shape

    feature_map = (
        patch_tokens
        .reshape(
            B,
            PATCH_GRID_SIZE,
            PATCH_GRID_SIZE,
            D,
        )
        .permute(
            0,
            3,
            1,
            2,
        )
    )

    pooled = F.adaptive_avg_pool2d(
        feature_map,
        output_size=(
            LATENT_GRID_SIZE,
            LATENT_GRID_SIZE,
        ),
    )

    pooled = (
        pooled
        .permute(
            0,
            2,
            3,
            1,
        )
        .reshape(
            B,
            LATENT_GRID_SIZE
            * LATENT_GRID_SIZE,
            D,
        )
        .float()
    )

    return (
        pooled[0:1],
        pooled[1:2],
    )


# ============================================================
# Model loading
# ============================================================

def load_fm_policy(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    vision_history_len = int(
        checkpoint["vision_history_len"]
    )

    proprio_history_len = int(
        checkpoint["proprio_history_len"]
    )

    proprio_mean = np.asarray(
        checkpoint["proprio_mean"],
        dtype=np.float32,
    )

    proprio_std = np.asarray(
        checkpoint["proprio_std"],
        dtype=np.float32,
    )

    action_horizon = int(
        checkpoint.get(
            "action_horizon",
            16,
        )
    )

    condition_dim = int(
        checkpoint.get(
            "condition_dim",
            256,
        )
    )

    hidden_dim = int(
        checkpoint.get(
            "hidden_dim",
            512,
        )
    )

    proprio_dim = int(
        checkpoint.get(
            "proprio_dim",
            12,
        )
    )

    action_dim = int(
        checkpoint.get(
            "action_dim",
            7,
        )
    )

    num_decoder_layer = int(
        checkpoint.get(
            "num_decoder_layer",
            4,
        )
    )

    num_heads = int(
        checkpoint.get(
            "num_heads",
            8,
        )
    )

    dim_feedforward = int(
        checkpoint.get(
            "dim_feedforward",
            2048,
        )
    )

    agent_encoder = resnet18(
        weights=_resnet_weights
    )

    wrist_encoder = resnet18(
        weights=_resnet_weights
    )

    agent_encoder.fc = nn.Identity()
    wrist_encoder.fc = nn.Identity()

    obs_encoder = ObservationEncoder(
        agent_encoder=agent_encoder,
        wrist_encoder=wrist_encoder,
        proprio_dim=proprio_dim,
        vision_history_len=
            vision_history_len,
        proprio_history_len=
            proprio_history_len,
        condition_dim=condition_dim,
    ).to(device)

    policy = FlowMatchingPolicy(
        observation_encoder=
            obs_encoder,
        agent_feat_dim=
            512 * vision_history_len,
        wrist_feat_dim=
            512 * vision_history_len,
        proprio_feat_dim=
            proprio_dim
            * proprio_history_len,
        action_dim=action_dim,
        action_horizon=
            action_horizon,
        hidden_dim=hidden_dim,
        num_layers=
            num_decoder_layer,
        num_heads=num_heads,
        dim_feedforward=
            dim_feedforward,
    ).to(device)

    policy.load_state_dict(
        checkpoint[
            "policy_state_dict"
        ]
    )

    policy.eval()

    return {
        "policy":
            policy,
        "checkpoint":
            checkpoint,
        "vision_history_len":
            vision_history_len,
        "proprio_history_len":
            proprio_history_len,
        "proprio_mean":
            proprio_mean,
        "proprio_std":
            proprio_std,
        "action_horizon":
            action_horizon,
    }


def load_world_model(
    checkpoint_path,
    device,
):
    """
    Architecture matches the mixed V-JEPA WM used in the current project.
    """
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
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    wm.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    wm.eval()

    return wm


def load_cube_head(
    checkpoint_path,
    device,
):
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    head = CubePositionHead(
        latent_dim=384,
        hidden_dim=512,
        visual_tokens_per_camera=16,
    ).to(device)

    head.load_state_dict(
        ckpt[
            "cube_head_state_dict"
        ]
    )

    head.eval()

    cube_mean = (
        ckpt["cube_mean"]
        .to(device)
        .float()
    )

    cube_std = (
        ckpt["cube_std"]
        .to(device)
        .float()
    )

    return (
        head,
        cube_mean,
        cube_std,
        ckpt,
    )


# ============================================================
# FM candidate sampling
# ============================================================

@torch.no_grad()
def sample_fm_candidates(
    policy,
    agent_tensor,
    wrist_tensor,
    proprio_tensor,
    K,
    num_steps,
):
    """
    Fix current observation, vary FM initial noise x0.

    Output:
        [K,H,7]
    """
    agent_batch = (
        agent_tensor.expand(
            K,
            *agent_tensor.shape[1:],
        )
    )

    wrist_batch = (
        wrist_tensor.expand(
            K,
            *wrist_tensor.shape[1:],
        )
    )

    proprio_batch = (
        proprio_tensor.expand(
            K,
            *proprio_tensor.shape[1:],
        )
    )

    chunks = policy.sample_actions(
        agent_batch,
        wrist_batch,
        proprio_batch,
        num_steps=num_steps,
    )

    return clean_action_chunks(
        chunks
    )


# ============================================================
# Physical prediction adapter
# ============================================================

@torch.no_grad()
def predict_candidate_futures(
    wm,
    cube_head,
    cube_mean,
    cube_std,
    agent_features,
    wrist_features,
    task_state,
    robot_config,
    action_chunks,
):
    """
    Predict all K futures.

    Cube:
        raw wm.dynamics latent -> cube head

    Robot physical state:
        wm.predict_future -> normalized physical dict
        then denormalize using WM statistics.

    Returns terminal H16 predictions.
    """
    K = action_chunks.shape[0]

    current_latent = (
        wm.encode_current_features(
            agent_features=
                agent_features,
            wrist_features=
                wrist_features,
            task_state=
                task_state,
            robot_config=
                robot_config,
        )
    )

    current_latent_k = (
        current_latent.expand(
            K,
            -1,
            -1,
        )
    )

    # --------------------------------------------------------
    # Raw future latent for cube decoder.
    # --------------------------------------------------------

    raw_future_latent = (
        wm.dynamics(
            state_tokens=
                current_latent_k,
            actions=
                action_chunks,
        )
    )

    pred_cube = (
        cube_head(
            raw_future_latent
        )
        * cube_std
        + cube_mean
    )

    # --------------------------------------------------------
    # Current WM API:
    # predict_future returns a dictionary with normalized
    # physical-state predictions.
    #
    # This does run the dynamics path again. That's acceptable
    # for the first K=8 experiment; optimize later if needed.
    # --------------------------------------------------------

    pred = wm.predict_future(
        current_latent=
            current_latent_k,
        actions=
            action_chunks,
    )

    if not isinstance(
        pred,
        dict,
    ):
        raise TypeError(
            "Expected current wm.predict_future() to return "
            "a dict containing task_state and robot_config. "
            f"Got {type(pred)}."
        )

    pred_task = (
        pred["task_state"]
        * wm.task_std
        + wm.task_mean
    )

    pred_config = (
        pred["robot_config"]
        * wm.config_std
        + wm.config_mean
    )

    return {
        "cube":
            pred_cube[:, -1],
        "task":
            pred_task[:, -1],
        "config":
            pred_config[:, -1],
    }


# ============================================================
# Candidate ranking
# ============================================================

@torch.no_grad()
def choose_reranked_chunk(
    *,
    wm,
    cube_head,
    cube_mean,
    cube_std,
    scorer,
    vjepa_encoder,
    vjepa_processor,
    obs,
    action_chunks,
    target_pos,
    device,
    print_scores=False,
):
    """
    World-model terminal-state reranking.
    """
    agent_features, wrist_features = (
        extract_vjepa_features(
            encoder=
                vjepa_encoder,
            processor=
                vjepa_processor,
            agent_image=
                obs["agentview_image"],
            wrist_image=
                obs[
                    "robot0_eye_in_hand_image"
                ],
            device=device,
        )
    )

    task_state = torch.from_numpy(
        make_wm_task_state(
            obs
        )
    ).unsqueeze(0).to(
        device
    )

    robot_config = torch.from_numpy(
        make_wm_robot_config(
            obs
        )
    ).unsqueeze(0).to(
        device
    )

    futures = predict_candidate_futures(
        wm=wm,
        cube_head=cube_head,
        cube_mean=cube_mean,
        cube_std=cube_std,
        agent_features=
            agent_features,
        wrist_features=
            wrist_features,
        task_state=
            task_state,
        robot_config=
            robot_config,
        action_chunks=
            action_chunks,
    )

    cube_np = (
        futures["cube"]
        .float()
        .cpu()
        .numpy()
    )

    task_np = (
        futures["task"]
        .float()
        .cpu()
        .numpy()
    )

    config_np = (
        futures["config"]
        .float()
        .cpu()
        .numpy()
    )

    descriptors = (
        build_descriptor_batch(
            cube_pos=
                cube_np,
            target_pos=
                target_pos,
            task_state=
                task_np,
            robot_config=
                config_np,
        )
    )

    score_out = scorer.score_batch(
        descriptors
    )

    best_index = int(
        np.argmax(
            score_out["score"]
        )
    )

    if print_scores:
        order = np.argsort(
            - score_out["score"]
        )

        print(
            "\n  candidate ranking:"
        )

        for rank, i in enumerate(
            order
        ):
            print(
                f"    rank={rank+1:2d} "
                f"k={i:2d} "
                f"score={score_out['score'][i]:+.4f} "
                f"steps={score_out['expected_steps'][i]:7.2f} "
                f"dist={score_out['mean_success_distance'][i]:.4f}"
            )

        print(
            "    selected:",
            best_index,
        )

    return {
        "best_index":
            best_index,
        "best_chunk":
            action_chunks[
                best_index
            ],
        "scores":
            score_out,
    }


# ============================================================
# Main rollout
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "random_k",
            "rerank",
        ],
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--K",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--flow-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--renderer",
        action="store_true",
    )

    parser.add_argument(
        "--print-candidates",
        action="store_true",
    )

    parser.add_argument(
        "--fm-checkpoint",
        default=(
            "policies/flow_matching/"
            "flowmatching_1_4_1.pth"
        ),
    )

    parser.add_argument(
        "--wm-checkpoint",
        default=(
            "checkpoints/"
            "world_model_vjepa_mixed/"
            "world_model_best.pth"
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
        "--success-db",
        default=(
            "data/"
            "success_cost_to_go_db.npz"
        ),
    )

    parser.add_argument(
        "--vjepa-checkpoint",
        default=None,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Reproducibility.
    #
    # FM randomness uses torch.
    # environment / target sampling uses numpy.
    # --------------------------------------------------------

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            args.seed
        )

    # --------------------------------------------------------
    # FM policy.
    # --------------------------------------------------------

    fm = load_fm_policy(
        args.fm_checkpoint,
        device,
    )

    policy = fm["policy"]

    vision_history_len = (
        fm[
            "vision_history_len"
        ]
    )

    proprio_history_len = (
        fm[
            "proprio_history_len"
        ]
    )

    proprio_mean = (
        fm["proprio_mean"]
    )

    proprio_std = (
        fm["proprio_std"]
    )

    action_horizon = (
        fm["action_horizon"]
    )

    if action_horizon != 16:
        raise ValueError(
            "This reranker experiment assumes H=16, "
            f"but FM checkpoint has H={action_horizon}."
        )

    # --------------------------------------------------------
    # Reranker stack only when needed.
    # --------------------------------------------------------

    wm = None
    cube_head = None
    cube_mean = None
    cube_std = None
    scorer = None
    vjepa_encoder = None
    vjepa_processor = None

    if args.mode == "rerank":
        wm = load_world_model(
            args.wm_checkpoint,
            device,
        )

        (
            cube_head,
            cube_mean,
            cube_std,
            cube_ckpt,
        ) = load_cube_head(
            args.cube_head,
            device,
        )

        scorer = (
            SuccessCostToGoScorer(
                npz_path=
                    args.success_db,
                k=20,
                temperature=1.0,
                distance_penalty=0.15,
            )
        )

        vjepa_encoder = (
            load_vjepa_encoder(
                device=device,
                checkpoint_path=
                    args.vjepa_checkpoint,
            )
        )

        vjepa_processor = (
            load_vjepa_preprocessor()
        )

    # --------------------------------------------------------
    # Environment.
    # --------------------------------------------------------

    placement_initializer = (
        UniformRandomSampler(
            name="ObjectSampler",
            x_range=[
                -0.08,
                0.08,
            ],
            y_range=[
                -0.05,
                0.02,
            ],
            rotation=None,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=[
                0,
                0,
                0.8,
            ],
            z_offset=0.01,
        )
    )

    env = ConstrainedPickPlace(
        robots="Panda",
        placement_initializer=
            placement_initializer,
        has_renderer=
            args.renderer,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ],
        camera_heights=[
            224,
            224,
        ],
        camera_widths=[
            224,
            224,
        ],
        control_freq=10,
        horizon=args.max_steps,
        ignore_done=False,
    )

    # Some robosuite versions expose seed(), some do not.
    if hasattr(
        env,
        "seed",
    ):
        try:
            env.seed(
                args.seed
            )
        except Exception:
            pass

    # --------------------------------------------------------
    # Statistics.
    # --------------------------------------------------------

    num_success = 0
    num_drop = 0
    num_invalid = 0
    num_done = 0
    num_timeout = 0

    trajectory_lengths = []

    selected_ranks = []
    selected_score_margins = []

    print()
    print("=" * 70)
    print(
        f"FM evaluation mode={args.mode}"
    )
    print(
        f"episodes={args.episodes} "
        f"K={1 if args.mode == 'baseline' else args.K} "
        f"H={action_horizon} "
        f"execute={args.execution_horizon}"
    )
    print("=" * 70)

    # ========================================================
    # Episodes
    # ========================================================

    for episode in range(
        args.episodes
    ):
        obs = env.reset()

        target_cube_pos = (
            sample_target(
                obs["cube_pos"]
            )
        )

        env.set_target(
            target_cube_pos
        )

        obs = (
            env._get_observations()
        )

        initial_cube_z = float(
            obs["cube_pos"][2]
        )

        was_lifted = False
        ever_grasped = False

        success = False
        failure_reason = None

        # ----------------------------------------------------
        # FM observation histories.
        # ----------------------------------------------------

        agent_history = deque(
            maxlen=
                vision_history_len
        )

        wrist_history = deque(
            maxlen=
                vision_history_len
        )

        proprio_history = deque(
            maxlen=
                proprio_history_len
        )

        agent_img = (
            process_policy_image(
                obs[
                    "agentview_image"
                ]
            )
        )

        wrist_img = (
            process_policy_image(
                obs[
                    "robot0_eye_in_hand_image"
                ]
            )
        )

        proprio = (
            make_policy_proprio(
                obs
            )
        )

        proprio = (
            proprio
            - proprio_mean
        ) / proprio_std

        for _ in range(
            vision_history_len
        ):
            agent_history.append(
                agent_img.clone()
            )

            wrist_history.append(
                wrist_img.clone()
            )

        for _ in range(
            proprio_history_len
        ):
            proprio_history.append(
                proprio.copy()
            )

        t = 0
        replan_count = 0

        # ====================================================
        # Receding-horizon control
        # ====================================================

        while t < args.max_steps:
            agent_tensor = (
                torch.stack(
                    list(
                        agent_history
                    ),
                    dim=0,
                )
                .unsqueeze(0)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            wrist_tensor = (
                torch.stack(
                    list(
                        wrist_history
                    ),
                    dim=0,
                )
                .unsqueeze(0)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            proprio_tensor = (
                torch.from_numpy(
                    np.stack(
                        list(
                            proprio_history
                        ),
                        axis=0,
                    )
                )
                .unsqueeze(0)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            # ------------------------------------------------
            # Sample FM chunks.
            # ------------------------------------------------

            K = (
                1
                if args.mode
                == "baseline"
                else args.K
            )

            action_chunks = (
                sample_fm_candidates(
                    policy=policy,
                    agent_tensor=
                        agent_tensor,
                    wrist_tensor=
                        wrist_tensor,
                    proprio_tensor=
                        proprio_tensor,
                    K=K,
                    num_steps=
                        args.flow_steps,
                )
            )

            # ------------------------------------------------
            # Select chunk.
            # ------------------------------------------------

            if args.mode == "baseline":
                selected_chunk = (
                    action_chunks[0]
                )

            elif args.mode == "random_k":
                random_index = int(
                    np.random.randint(
                        0,
                        K,
                    )
                )

                selected_chunk = (
                    action_chunks[
                        random_index
                    ]
                )

            else:
                rerank_out = (
                    choose_reranked_chunk(
                        wm=wm,
                        cube_head=
                            cube_head,
                        cube_mean=
                            cube_mean,
                        cube_std=
                            cube_std,
                        scorer=
                            scorer,
                        vjepa_encoder=
                            vjepa_encoder,
                        vjepa_processor=
                            vjepa_processor,
                        obs=obs,
                        action_chunks=
                            action_chunks,
                        target_pos=
                            target_cube_pos,
                        device=device,
                        print_scores=(
                            args.print_candidates
                            and episode < 3
                        ),
                    )
                )

                selected_chunk = (
                    rerank_out[
                        "best_chunk"
                    ]
                )

                scores = (
                    rerank_out[
                        "scores"
                    ]["score"]
                )

                order = np.argsort(
                    - scores
                )

                selected_ranks.append(
                    int(
                        np.where(
                            order
                            == rerank_out[
                                "best_index"
                            ]
                        )[0][0]
                    )
                )

                if len(scores) > 1:
                    sorted_scores = (
                        np.sort(
                            scores
                        )[::-1]
                    )

                    selected_score_margins.append(
                        float(
                            sorted_scores[0]
                            - sorted_scores[1]
                        )
                    )

            # ------------------------------------------------
            # Execute first M actions.
            #
            # IMPORTANT:
            # selected_chunk has already been cleaned before
            # the WM sees it, so simulator executes exactly
            # what the WM evaluated.
            # ------------------------------------------------

            should_break = False

            n_execute = min(
                args.execution_horizon,
                args.max_steps - t,
            )

            for action_idx in range(
                n_execute
            ):
                action = (
                    selected_chunk[
                        action_idx
                    ]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                obs, reward, done, info = (
                    env.step(
                        action
                    )
                )

                t += 1

                if args.renderer:
                    env.render()

                cube_pos = np.asarray(
                    obs["cube_pos"],
                    dtype=np.float32,
                )

                # --------------------------------------------
                # Update FM histories after every action.
                # --------------------------------------------

                agent_img = (
                    process_policy_image(
                        obs[
                            "agentview_image"
                        ]
                    )
                )

                wrist_img = (
                    process_policy_image(
                        obs[
                            "robot0_eye_in_hand_image"
                        ]
                    )
                )

                proprio = (
                    make_policy_proprio(
                        obs
                    )
                )

                proprio = (
                    proprio
                    - proprio_mean
                ) / proprio_std

                agent_history.append(
                    agent_img
                )

                wrist_history.append(
                    wrist_img
                )

                proprio_history.append(
                    proprio
                )

                # --------------------------------------------
                # Invalid state.
                # --------------------------------------------

                if (
                    cube_pos[2] < 0.7
                    or not np.all(
                        np.isfinite(
                            cube_pos
                        )
                    )
                ):
                    failure_reason = (
                        "invalid_cube_pose"
                    )
                    num_invalid += 1
                    should_break = True
                    break

                # --------------------------------------------
                # Grasp / lift status.
                # --------------------------------------------

                grasped = (
                    env._check_grasp(
                        gripper=
                            env.robots[
                                0
                            ].gripper,
                        object_geoms=
                            env.cube
                            .contact_geoms,
                    )
                )

                if grasped:
                    ever_grasped = True

                if (
                    cube_pos[2]
                    > initial_cube_z
                    + 0.05
                ):
                    was_lifted = True

                # --------------------------------------------
                # Success.
                # --------------------------------------------

                target_half_size = (
                    obs[
                        "target_half_size"
                    ]
                )

                x_in_target = (
                    abs(
                        cube_pos[0]
                        - target_cube_pos[0]
                    )
                    < target_half_size[0]
                )

                y_in_target = (
                    abs(
                        cube_pos[1]
                        - target_cube_pos[1]
                    )
                    < target_half_size[1]
                )

                z_error = abs(
                    cube_pos[2]
                    - target_cube_pos[2]
                )

                success = (
                    x_in_target
                    and y_in_target
                    and z_error < 0.025
                    and not grasped
                    and was_lifted
                )

                if success:
                    num_success += 1

                    trajectory_lengths.append(
                        t
                    )

                    should_break = True
                    break

                # --------------------------------------------
                # Drop.
                # --------------------------------------------

                if (
                    was_lifted
                    and ever_grasped
                    and not grasped
                    and not (
                        x_in_target
                        and y_in_target
                    )
                    and cube_pos[2]
                    < initial_cube_z
                    + 0.04
                ):
                    failure_reason = (
                        "dropped_object"
                    )
                    num_drop += 1
                    should_break = True
                    break

                if done:
                    failure_reason = (
                        "environment_done"
                    )
                    num_done += 1
                    should_break = True
                    break

            replan_count += 1

            if should_break:
                break

        if (
            not success
            and failure_reason is None
            and t >= args.max_steps
        ):
            failure_reason = (
                "timeout"
            )
            num_timeout += 1

        print(
            f"Episode {episode:03d} | "
            f"success={success} | "
            f"steps={t:3d} | "
            f"replans={replan_count:3d} | "
            f"failure={failure_reason} | "
            f"total={num_success}/{episode+1}"
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print(
        f"FM {args.mode.upper()} RESULT"
    )
    print("=" * 70)

    print(
        f"Success: "
        f"{num_success}/{args.episodes} "
        f"({100.0 * num_success / args.episodes:.1f}%)"
    )

    print(
        "Failures:"
    )
    print(
        "  drop:   ",
        num_drop,
    )
    print(
        "  invalid:",
        num_invalid,
    )
    print(
        "  done:   ",
        num_done,
    )
    print(
        "  timeout:",
        num_timeout,
    )

    if trajectory_lengths:
        print(
            "Successful trajectory length:"
        )
        print(
            f"  mean={np.mean(trajectory_lengths):.1f}"
        )
        print(
            f"  median={np.median(trajectory_lengths):.1f}"
        )
        print(
            f"  min={np.min(trajectory_lengths)}"
        )
        print(
            f"  max={np.max(trajectory_lengths)}"
        )

    if (
        args.mode == "rerank"
        and selected_score_margins
    ):
        print(
            "Reranker diagnostics:"
        )
        print(
            "  mean top1-top2 score margin:",
            f"{np.mean(selected_score_margins):.4f}",
        )

    env.close()


if __name__ == "__main__":
    main()
