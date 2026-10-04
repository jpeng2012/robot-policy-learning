
from __future__ import annotations

"""
Core utilities for FM + World Model + success-trajectory retrieval reranking.

Pipeline
--------
current observation
    -> sample K Flow-Matching action chunks [K,H,7]
    -> clean actions (zero rotation, binary gripper, clip)
    -> frozen WM predicts K future trajectories
    -> cube head predicts future cube xyz from raw WM latent
    -> physical decoder supplies predicted task_state / robot_config
    -> build the same 17-D descriptor used by success_cost_to_go_db
    -> retrieve successful states and score each candidate
    -> select highest score

The only model-version-specific piece is `physical_decoder`.
It must convert raw future latent into:
    task_state   [K,H,9] in PHYSICAL units
    robot_config [K,H,7] in PHYSICAL units

Everything else is independent of the exact WM decoder implementation.
"""

import numpy as np
import torch


# ============================================================
# Action cleanup
# ============================================================

def clean_action_chunks(action_chunks: torch.Tensor) -> torch.Tensor:
    """
    Apply the action convention that matched the expert / environment.

    Input:
        [K,H,7]

    dims:
        0:3   translation
        3:6   rotation  -> force exactly zero
        6     gripper   -> {-1,+1}
    """
    a = action_chunks.clone()

    a[..., 3:6] = 0.0

    a[..., 6] = torch.where(
        a[..., 6] > 0,
        torch.ones_like(a[..., 6]),
        -torch.ones_like(a[..., 6]),
    )

    return a.clamp(-1.0, 1.0)


# ============================================================
# Success cost-to-go database
# ============================================================

class SuccessCostToGoScorer:
    """
    kNN success-manifold scorer.

    Expected descriptor:
        [cube - eef]       3
        [target - cube]    3
        cube_z             1
        eef_z              1
        gripper_qpos       2
        joints             7
                           --
                           17

    Score:
        score =
            - expected_steps / step_scale
            - distance_penalty * mean_success_distance

    Higher is better.
    """

    def __init__(
        self,
        npz_path="data/success_cost_to_go_db.npz",
        k=20,
        temperature=1.0,
        distance_penalty=0.5,
    ):
        db = np.load(npz_path)

        # Support a few sensible key names so this utility is robust
        # to small naming changes in the DB builder.
        self.x = self._get(
            db,
            [
                "descriptors",
                "features",
                "X",
                "states",
            ],
        ).astype(np.float32)

        self.steps = self._get(
            db,
            [
                "steps_remaining",
                "remaining_steps",
                "cost_to_go",
                "steps",
            ],
        ).astype(np.float32).reshape(-1)

        self.mean = self._get_optional(
            db,
            [
                "feature_mean",
                "descriptor_mean",
                "mean",
                "x_mean",
            ],
        )

        self.std = self._get_optional(
            db,
            [
                "feature_std",
                "descriptor_std",
                "std",
                "x_std",
            ],
        )

        # If mean/std were not stored explicitly, derive them exactly
        # from the database descriptors.
        if self.mean is None:
            self.mean = self.x.mean(axis=0)

        if self.std is None:
            self.std = self.x.std(axis=0)

        self.mean = np.asarray(
            self.mean,
            dtype=np.float32,
        ).reshape(-1)

        self.std = np.asarray(
            self.std,
            dtype=np.float32,
        ).reshape(-1)

        self.std = np.maximum(
            self.std,
            1e-6,
        )

        # DB may contain raw descriptors or already standardized
        # descriptors. We deliberately standardize from raw values here.
        self.xz = (
            self.x - self.mean
        ) / self.std

        self.k = int(k)
        self.temperature = float(temperature)
        self.distance_penalty = float(distance_penalty)

        self.step_scale = float(
            np.percentile(
                self.steps,
                90,
            )
        )

    @staticmethod
    def _get(db, names):
        for name in names:
            if name in db:
                return db[name]

        raise KeyError(
            f"None of these keys exist in DB: {names}. "
            f"Available keys: {list(db.keys())}"
        )

    @staticmethod
    def _get_optional(db, names):
        for name in names:
            if name in db:
                return db[name]
        return None

    def score_batch(
        self,
        descriptors: np.ndarray,
    ):
        """
        descriptors:
            [K,17]

        Returns dict of arrays:
            expected_steps          [K]
            mean_success_distance   [K]
            score                   [K]
        """
        q = np.asarray(
            descriptors,
            dtype=np.float32,
        )

        if q.ndim == 1:
            q = q[None]

        qz = (
            q - self.mean[None]
        ) / self.std[None]

        # K candidates is small (8), and DB ~87k states.
        # Brute-force numpy distance is perfectly acceptable for
        # the first experiment and avoids ANN implementation noise.
        diff = (
            qz[:, None, :]
            - self.xz[None, :, :]
        )

        dist2 = np.sum(
            diff * diff,
            axis=-1,
        )

        k = min(
            self.k,
            self.xz.shape[0],
        )

        # [K_candidates, k]
        idx = np.argpartition(
            dist2,
            kth=k - 1,
            axis=1,
        )[:, :k]

        nearest_dist2 = np.take_along_axis(
            dist2,
            idx,
            axis=1,
        )

        nearest_dist = np.sqrt(
            nearest_dist2
        )

        nearest_steps = self.steps[
            idx
        ]

        # ----------------------------------------------------
        # Numerically stable Gaussian weighting.
        #
        # Previous version directly computed:
        #
        #   exp(-d^2 / (2*tau^2))
        #
        # For an OOD query, every neighbor can be far enough
        # that all float32 weights underflow to exactly zero.
        # Then both weighted numerators become zero and the
        # scorer falsely returns:
        #
        #   expected_steps = 0
        #   mean_distance  = 0
        #   score          = 0
        #
        # which incorrectly looks like an ideal terminal state.
        #
        # Subtracting the row-wise maximum logit is equivalent
        # to softmax stabilization and preserves the relative
        # Gaussian weights while guaranteeing at least one
        # finite weight per query.
        # ----------------------------------------------------

        logits = (
            - nearest_dist2
            / (
                2.0
                * self.temperature
                * self.temperature
            )
        )

        logits = (
            logits
            - np.max(
                logits,
                axis=1,
                keepdims=True,
            )
        )

        w = np.exp(
            logits
        )

        ws = w.sum(
            axis=1
        )

        expected_steps = (
            w * nearest_steps
        ).sum(axis=1) / ws

        mean_success_distance = (
            w * nearest_dist
        ).sum(axis=1) / ws

        score = (
            - expected_steps
            / self.step_scale
            - self.distance_penalty
            * mean_success_distance
        )

        if not (
            np.all(np.isfinite(expected_steps))
            and np.all(np.isfinite(mean_success_distance))
            and np.all(np.isfinite(score))
        ):
            raise FloatingPointError(
                "Non-finite success-retrieval score. "
                "Check descriptor normalization and kNN distances."
            )

        return {
            "expected_steps":
                expected_steps,
            "mean_success_distance":
                mean_success_distance,
            "score":
                score,
        }


# ============================================================
# Descriptor
# ============================================================

def build_descriptor_batch(
    cube_pos: np.ndarray,
    target_pos: np.ndarray,
    task_state: np.ndarray,
    robot_config: np.ndarray,
) -> np.ndarray:
    """
    Build the same 17-D success-retrieval descriptor.

    cube_pos:
        [K,3]

    target_pos:
        [3] or [K,3]

    task_state:
        [K,9]
        expected layout:
            eef xyz      0:3
            eef quat     3:7
            gripper qpos 7:9

    robot_config:
        [K,7]
    """
    cube_pos = np.asarray(
        cube_pos,
        dtype=np.float32,
    )

    task_state = np.asarray(
        task_state,
        dtype=np.float32,
    )

    robot_config = np.asarray(
        robot_config,
        dtype=np.float32,
    )

    target_pos = np.asarray(
        target_pos,
        dtype=np.float32,
    )

    if target_pos.ndim == 1:
        target_pos = np.broadcast_to(
            target_pos[None, :],
            cube_pos.shape,
        )

    eef_pos = task_state[:, 0:3]
    gripper = task_state[:, 7:9]

    desc = np.concatenate(
        [
            cube_pos - eef_pos,        # 3
            target_pos - cube_pos,     # 3
            cube_pos[:, 2:3],          # 1
            eef_pos[:, 2:3],           # 1
            gripper,                   # 2
            robot_config,              # 7
        ],
        axis=1,
    )

    assert desc.shape[1] == 17

    return desc.astype(
        np.float32,
    )


# ============================================================
# Candidate reranking
# ============================================================

@torch.no_grad()
def rerank_action_chunks(
    *,
    wm,
    cube_head,
    scorer: SuccessCostToGoScorer,
    agent_features: torch.Tensor,
    wrist_features: torch.Tensor,
    task_state: torch.Tensor,
    robot_config: torch.Tensor,
    action_chunks: torch.Tensor,
    target_pos: np.ndarray,
    cube_mean: torch.Tensor,
    cube_std: torch.Tensor,
    physical_decoder,
):
    """
    Rank K action chunks using terminal H16 predicted state.

    Inputs
    ------
    current features:
        agent_features  [1,16,768]
        wrist_features  [1,16,768]
        task_state      [1,9] physical units
        robot_config    [1,7] physical units

    action_chunks:
        [K,H,7]

    physical_decoder:
        callable(raw_future_latent, wm) -> (task_state, robot_config)

        task_state:
            [K,H,9] physical units

        robot_config:
            [K,H,7] physical units

    Returns
    -------
    dict with:
        best_index
        best_chunk
        all_chunks
        scores
        expected_steps
        mean_success_distance
        terminal_cube
        terminal_task
        terminal_config
    """
    device = action_chunks.device

    action_chunks = clean_action_chunks(
        action_chunks
    )

    K = action_chunks.shape[0]

    # --------------------------------------------------------
    # Encode current state only once.
    # --------------------------------------------------------

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

    # All K candidate chunks start from the SAME current state.
    current_latent = (
        current_latent.expand(
            K,
            -1,
            -1,
        )
    )

    # --------------------------------------------------------
    # Predict raw future latent for all K candidates in one batch.
    # --------------------------------------------------------

    raw_future_latent = wm.dynamics(
        state_tokens=current_latent,
        actions=action_chunks,
    )

    # --------------------------------------------------------
    # Cube xyz decoder.
    # --------------------------------------------------------

    cube_norm = cube_head(
        raw_future_latent
    )

    cube_pred = (
        cube_norm
        * cube_std
        + cube_mean
    )

    # --------------------------------------------------------
    # Existing physical-state decoder from your current WM.
    # This is deliberately supplied as an adapter because the
    # older project copy and your current local version differ.
    # --------------------------------------------------------

    pred_task, pred_config = (
        physical_decoder(
            raw_future_latent,
            wm,
        )
    )

    if pred_task.shape[:2] != action_chunks.shape[:2]:
        raise ValueError(
            f"physical_decoder task shape {pred_task.shape} "
            f"does not match actions {action_chunks.shape}"
        )

    if pred_config.shape[:2] != action_chunks.shape[:2]:
        raise ValueError(
            f"physical_decoder config shape {pred_config.shape} "
            f"does not match actions {action_chunks.shape}"
        )

    # --------------------------------------------------------
    # Terminal H16 prediction.
    # --------------------------------------------------------

    terminal_cube = (
        cube_pred[:, -1]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    terminal_task = (
        pred_task[:, -1]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    terminal_config = (
        pred_config[:, -1]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    descriptors = build_descriptor_batch(
        cube_pos=terminal_cube,
        target_pos=target_pos,
        task_state=terminal_task,
        robot_config=terminal_config,
    )

    score_out = scorer.score_batch(
        descriptors
    )

    best_index = int(
        np.argmax(
            score_out["score"]
        )
    )

    return {
        "best_index":
            best_index,

        "best_chunk":
            action_chunks[
                best_index
            ],

        "all_chunks":
            action_chunks,

        "score":
            score_out["score"],

        "expected_steps":
            score_out[
                "expected_steps"
            ],

        "mean_success_distance":
            score_out[
                "mean_success_distance"
            ],

        "terminal_cube":
            terminal_cube,

        "terminal_task":
            terminal_task,

        "terminal_config":
            terminal_config,

        "descriptors":
            descriptors,
    }


# ============================================================
# Adapter examples
# ============================================================

def physical_decoder_from_dict_api(
    raw_future_latent,
    wm,
):
    """
    EXAMPLE adapter.

    Use this if your current WM has a decoder method returning:

        {
            "task_state":   normalized [K,H,9],
            "robot_config": normalized [K,H,7],
        }

    Rename `decode_future` below to the exact method already used
    in your evaluate_world_model_physical.py.

    This function intentionally raises until that single method name
    is filled in, rather than silently guessing.
    """

    raise NotImplementedError(
        "Paste the exact physical-decoder call from "
        "evaluate_world_model_physical.py here. "
        "After that, denormalize with "
        "wm.task_std/task_mean and "
        "wm.config_std/config_mean."
    )


def print_candidate_scores(result):
    """
    Useful diagnostic during the first few rollouts.
    """
    order = np.argsort(
        - result["score"]
    )

    print(
        "\nWM reranker candidates:"
    )

    for rank, i in enumerate(
        order
    ):
        print(
            f"  rank={rank+1:2d} "
            f"k={i:2d} "
            f"score={result['score'][i]:+.4f} "
            f"steps={result['expected_steps'][i]:7.2f} "
            f"dist={result['mean_success_distance'][i]:.4f}"
        )

    print(
        "  selected:",
        result["best_index"],
    )
