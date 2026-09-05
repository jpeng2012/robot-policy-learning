# Seperate dataset building script for level 2, 
# which has two routes (LEFT and RIGHT). This 
# script builds a single dataset file with both routes, 
# and splits into train/val sets at the episode level.

import glob
import pickle
import random
import numpy as np


history_len = 4
action_horizon = 16
val_ratio = 0.1
seed = 42


def make_feature(step):
    obs = step["obs"]

    return np.concatenate([
        obs["eef_pos"],             # 3
        obs["gripper_qpos"],        # 2
        obs["cube_pos"],            # 3
        obs["target_pos"],          # 3
        obs["barrier_pos"],         # 3
        obs["barrier_half_size"],   # 3
        obs["prev_action"],         # 7
    ]).astype(np.float32)


def split_files(files, val_ratio):
    n_val = int(len(files) * val_ratio)

    val_files = files[:n_val]
    train_files = files[n_val:]

    return train_files, val_files


def build_samples(files):

    all_x = []
    all_y_one = []
    all_y_chunk = []
    all_routes = []

    for file in files:

        with open(file, "rb") as f:
            demo = pickle.load(f)

        traj = demo["trajectory"]
        route = demo["route"]

        features = [
            make_feature(step)
            for step in traj
        ]

        actions = np.stack([
            step["action"]
            for step in traj
        ]).astype(np.float32)

        T = len(traj)

        for t in range(T):

            # -------------------------------------------------
            # 4-frame observation history
            # -------------------------------------------------

            history = []

            for h in range(history_len):

                idx = t - history_len + 1 + h

                # Repeat first frame for missing history
                if idx < 0:
                    idx = 0

                history.append(features[idx])

            x = np.concatenate(history)

            # -------------------------------------------------
            # One-step BC target
            # -------------------------------------------------

            y_one = actions[t]

            # -------------------------------------------------
            # Action chunk target
            # -------------------------------------------------

            chunk = []

            for k in range(action_horizon):

                idx = t + k

                # Repeat final action near episode end
                if idx >= T:
                    idx = T - 1

                chunk.append(actions[idx])

            y_chunk = np.stack(chunk)

            all_x.append(x)
            all_y_one.append(y_one)
            all_y_chunk.append(y_chunk)

            # Metadata only
            all_routes.append(
                0 if route == "LEFT" else 1
            )

    return (
        np.stack(all_x),
        np.stack(all_y_one),
        np.stack(all_y_chunk),
        np.asarray(all_routes, dtype=np.int64),
    )


# ============================================================
# 1. Find episodes
# ============================================================

left_files = sorted(
    glob.glob("data/level2/left_*.pkl")
)

right_files = sorted(
    glob.glob("data/level2/right_*.pkl")
)

print("LEFT episodes:", len(left_files))
print("RIGHT episodes:", len(right_files))


# ============================================================
# 2. Shuffle independently
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

train_files = train_left + train_right
val_files = val_left + val_right

rng.shuffle(train_files)
rng.shuffle(val_files)


print()
print("Train episodes:", len(train_files))
print(
    "  LEFT:",
    len(train_left),
    "RIGHT:",
    len(train_right),
)

print("Val episodes:", len(val_files))
print(
    "  LEFT:",
    len(val_left),
    "RIGHT:",
    len(val_right),
)


# ============================================================
# 4. Convert train and val independently
# ============================================================

(
    X_train,
    Y_one_train,
    Y_chunk_train,
    route_train,
) = build_samples(train_files)

(
    X_val,
    Y_one_val,
    Y_chunk_val,
    route_val,
) = build_samples(val_files)


# ============================================================
# 5. Inspect
# ============================================================

print("\nTRAIN")
print("X:", X_train.shape)
print("Y_one:", Y_one_train.shape)
print("Y_chunk:", Y_chunk_train.shape)
print("route samples:", np.bincount(route_train))

print("\nVAL")
print("X:", X_val.shape)
print("Y_one:", Y_one_val.shape)
print("Y_chunk:", Y_chunk_val.shape)
print("route samples:", np.bincount(route_val))


# ============================================================
# 6. Save
# ============================================================

np.savez_compressed(
    "data/level2_dataset.npz",

    X_train=X_train,
    Y_one_train=Y_one_train,
    Y_chunk_train=Y_chunk_train,
    route_train=route_train,

    X_val=X_val,
    Y_one_val=Y_one_val,
    Y_chunk_val=Y_chunk_val,
    route_val=route_val,
)

print("\nSaved data/level2_dataset.npz")