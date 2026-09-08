# Build dataset index for Level 3 vision BC.
#
# Important:
# - LEFT / RIGHT split at EPISODE level
# - RGB images remain as JPG files on disk
# - Dataset stores image paths, proprioception, and actions
# - 4-frame observation history
# - 1-step and 16-step action targets
#
# Vision policy inputs:
#   agentview RGB history
#   wrist RGB history
#   robot joint positions
#   gripper joint positions
#
# We intentionally DO NOT provide:
#   eef_pos
#   cube_pos
#   target_pos
#   barrier_pos
#
# Those must now be inferred visually.

import glob
import os
import pickle
import random

import numpy as np


# ============================================================
# Config
# ============================================================

data_root = "data/level3"

proprio_history_len = 4
vision_history_len = 1
action_horizon = 16

val_ratio = 0.1
seed = 42


STATE_TO_ID = {
    "APPROACH": 0,
    "DESCEND": 1,
    "GRASP": 2,
    "LIFT": 3,
    "TRANSPORT": 4,
    "LOWER": 5,
    "RELEASE": 6,
}


# ============================================================
# Helpers
# ============================================================

def make_proprio(step):
    """
    Privileged robot state only.

    Panda:
        joint_pos     : 7
        eef_pos       : 3
        gripper_qpos  : 2

    total = 9
    """

    obs = step["obs"]

    return np.concatenate([
        obs["joint_pos"],        # 7
        obs["eef_pos"],          # 3
        obs["gripper_qpos"],     # 2
    ]).astype(np.float32)

def split_files(files, val_ratio):
    """
    Split at the EPISODE level.
    """

    n_val = int(len(files) * val_ratio)

    # Make sure we get at least one validation episode
    # if the route contains enough demos.
    if len(files) > 1:
        n_val = max(1, n_val)

    val_files = files[:n_val]
    train_files = files[n_val:]

    return train_files, val_files


def build_samples(files):
    """
    Construct sample indexes without loading JPG images.

    Returns:

        agent_paths : (N, vision_history_len)
        wrist_paths : (N, vision_history_len)

        proprio     : (N, proprio_history_len, 12)

        Y_one       : (N, 7)
        Y_chunk     : (N, action_horizon, 7)

        states      : (N,)
        routes      : (N,)
    """

    all_agent_paths = []
    all_wrist_paths = []

    all_proprio = []

    all_y_one = []
    all_y_chunk = []

    all_states = []
    all_routes = []

    for file in files:

        with open(file, "rb") as f:
            demo = pickle.load(f)

        traj = demo["trajectory"]
        route = demo["route"]

        T = len(traj)

        # ----------------------------------------------------
        # Precompute proprio + action arrays
        # ----------------------------------------------------

        proprio = [
            make_proprio(step)
            for step in traj
        ]

        actions = np.stack([
            step["action"]
            for step in traj
        ]).astype(np.float32)

        # ----------------------------------------------------
        # Generate one sample for every timestep
        # ----------------------------------------------------

        for t in range(T):

            # =================================================
            # Observation history
            # =================================================

            agent_history = []
            wrist_history = []
            proprio_history = []

            for h in range(vision_history_len):

                idx = (
                    t
                    - vision_history_len
                    + 1
                    + h
                )

                # Repeat first observation if history
                # extends before episode start
                if idx < 0:
                    idx = 0

                step = traj[idx]

                agent_history.append(
                    step["agent_image"]
                )

                wrist_history.append(
                    step["wrist_image"]
                )


            for h in range(proprio_history_len):
            
                idx = (
                    t
                    - proprio_history_len
                    + 1
                    + h
                )

                # Repeat first observation if history
                # extends before episode start
                if idx < 0:
                    idx = 0

                step = traj[idx]

                proprio_history.append(
                    proprio[idx]
                )

            # =================================================
            # One-step target
            # =================================================

            target_idx = min(t+1, T-1)
            y_one = actions[target_idx]

            # =================================================
            # Action chunk
            # =================================================

            chunk = []

            for k in range(action_horizon):

                idx = t + k

                # Pad end of trajectory by repeating
                # final action.
                if idx >= T:
                    idx = T - 1

                chunk.append(
                    actions[idx]
                )

            y_chunk = np.stack(
                chunk,
                axis=0,
            )

            # =================================================
            # Save sample
            # =================================================

            all_agent_paths.append(
                agent_history
            )

            all_wrist_paths.append(
                wrist_history
            )

            all_proprio.append(
                np.stack(
                    proprio_history,
                    axis=0,
                )
            )

            all_y_one.append(
                y_one
            )

            all_y_chunk.append(
                y_chunk
            )

            # -------------------------------------------------
            # Metadata only
            # -------------------------------------------------

            state = traj[target_idx][
                "expert_state"
            ]

            all_states.append(
                STATE_TO_ID[state]
            )

            all_routes.append(
                0 if route == "LEFT" else 1
            )

    # ========================================================
    # Convert arrays
    # ========================================================

    agent_paths = np.asarray(
        all_agent_paths
    )

    wrist_paths = np.asarray(
        all_wrist_paths
    )

    proprio = np.stack(
        all_proprio
    ).astype(np.float32)

    Y_one = np.stack(
        all_y_one
    ).astype(np.float32)

    Y_chunk = np.stack(
        all_y_chunk
    ).astype(np.float32)

    states = np.asarray(
        all_states,
        dtype=np.int64,
    )

    routes = np.asarray(
        all_routes,
        dtype=np.int64,
    )

    # ========================================================
    # Diagnostic action statistics
    # ========================================================

    print()

    for state_name, state_id in STATE_TO_ID.items():

        mask = states == state_id

        if mask.sum() == 0:
            continue

        state_actions = Y_one[mask]

        print(
            state_name,
            "N =",
            mask.sum(),
            "mean action =",
            np.round(
                state_actions.mean(axis=0),
                3,
            ),
        )

    return (
        agent_paths,
        wrist_paths,
        proprio,
        Y_one,
        Y_chunk,
        states,
        routes,
    )


# ============================================================
# 1. Find successful episode metadata files
# ============================================================

left_files = sorted(
    glob.glob(
        os.path.join(
            data_root,
            "left_*.pkl",
        )
    )
)

right_files = sorted(
    glob.glob(
        os.path.join(
            data_root,
            "right_*.pkl",
        )
    )
)


print(
    "LEFT episodes:",
    len(left_files),
)

print(
    "RIGHT episodes:",
    len(right_files),
)


if len(left_files) == 0:
    raise RuntimeError(
        "No LEFT episodes found"
    )

if len(right_files) == 0:
    raise RuntimeError(
        "No RIGHT episodes found"
    )


# ============================================================
# 2. Shuffle routes independently
# ============================================================

rng = random.Random(seed)

rng.shuffle(left_files)
rng.shuffle(right_files)


# ============================================================
# 3. Episode-level stratified split
# ============================================================

train_left, val_left = split_files(
    left_files,
    val_ratio,
)

train_right, val_right = split_files(
    right_files,
    val_ratio,
)


train_files = (
    train_left
    + train_right
)

val_files = (
    val_left
    + val_right
)


rng.shuffle(train_files)
rng.shuffle(val_files)


print()

print(
    "Train episodes:",
    len(train_files),
)

print(
    "  LEFT:",
    len(train_left),
    "RIGHT:",
    len(train_right),
)

print(
    "Val episodes:",
    len(val_files),
)

print(
    "  LEFT:",
    len(val_left),
    "RIGHT:",
    len(val_right),
)


# ============================================================
# 4. Build train dataset
# ============================================================

print("\nBuilding TRAIN samples")

(
    agent_train,
    wrist_train,
    proprio_train,

    Y_one_train,
    Y_chunk_train,

    state_train,
    route_train,

) = build_samples(
    train_files
)


# ============================================================
# 5. Build validation dataset
# ============================================================

print("\nBuilding VAL samples")

(
    agent_val,
    wrist_val,
    proprio_val,

    Y_one_val,
    Y_chunk_val,

    state_val,
    route_val,

) = build_samples(
    val_files
)


# ============================================================
# 6. Inspect
# ============================================================

print("\n================================")
print("TRAIN")
print("================================")

print(
    "agent paths:",
    agent_train.shape,
)

print(
    "wrist paths:",
    wrist_train.shape,
)

print(
    "proprio:",
    proprio_train.shape,
)

print(
    "Y_one:",
    Y_one_train.shape,
)

print(
    "Y_chunk:",
    Y_chunk_train.shape,
)

print(
    "route samples:",
    np.bincount(
        route_train
    ),
)

print(
    "state samples:",
    np.bincount(
        state_train,
        minlength=len(STATE_TO_ID),
    ),
)


print("\n================================")
print("VAL")
print("================================")

print(
    "agent paths:",
    agent_val.shape,
)

print(
    "wrist paths:",
    wrist_val.shape,
)

print(
    "proprio:",
    proprio_val.shape,
)

print(
    "Y_one:",
    Y_one_val.shape,
)

print(
    "Y_chunk:",
    Y_chunk_val.shape,
)

print(
    "route samples:",
    np.bincount(
        route_val
    ),
)

print(
    "state samples:",
    np.bincount(
        state_val,
        minlength=len(STATE_TO_ID),
    ),
)


# ============================================================
# 7. Sanity-check paths
# ============================================================

print("\nExample image history:")

for h in range(vision_history_len):

    agent_path = os.path.join(
        data_root,
        agent_train[0, h],
    )

    wrist_path = os.path.join(
        data_root,
        wrist_train[0, h],
    )

    print(
        f"h={h}:",
        agent_path,
    )

    if not os.path.exists(
        agent_path
    ):
        raise FileNotFoundError(
            agent_path
        )

    if not os.path.exists(
        wrist_path
    ):
        raise FileNotFoundError(
            wrist_path
        )


# ============================================================
# 8. Save index
# ============================================================

output_path = (
    "data/level3_hist_1_4_dataset.npz"
)

np.savez_compressed(
    output_path,

    # Image path histories
    agent_train=agent_train,
    wrist_train=wrist_train,

    agent_val=agent_val,
    wrist_val=wrist_val,

    # Proprio history
    proprio_train=proprio_train,
    proprio_val=proprio_val,

    # Actions
    Y_one_train=Y_one_train,
    Y_chunk_train=Y_chunk_train,

    Y_one_val=Y_one_val,
    Y_chunk_val=Y_chunk_val,

    # Metadata
    route_train=route_train,
    state_train=state_train,

    route_val=route_val,
    state_val=state_val,

    # Useful dataset config
    vision_history_len=np.asarray(
        vision_history_len,
        dtype=np.int64,
    ),

    proprio_history_len=np.asarray(
        proprio_history_len,
        dtype=np.int64,
    ),

    action_horizon=np.asarray(
        action_horizon,
        dtype=np.int64,
    ),
)


print(
    "\nSaved:",
    output_path,
)