import glob
import pickle
import numpy as np


history_len = 4
action_horizon = 16


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
    ]).astype(np.float32)           # 24 input features per frame


all_x = []
all_y_one = []
all_y_chunk = []
all_routes = []

files = sorted(
    glob.glob("data/level2/*.pkl")
)

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

        # --------------------------------------
        # Observation history
        # --------------------------------------

        history = []

        for h in range(history_len):

            idx = t - history_len + 1 + h

            if idx < 0:
                idx = 0

            history.append(
                features[idx]
            )

        x = np.concatenate(history)

        # --------------------------------------
        # One-step BC target
        # --------------------------------------

        y_one = actions[t]

        # --------------------------------------
        # Action chunk target
        # --------------------------------------

        chunk = []

        for k in range(action_horizon):

            idx = t + k

            if idx >= T:
                idx = T - 1

            chunk.append(
                actions[idx]
            )

        chunk = np.stack(chunk)

        all_x.append(x)
        all_y_one.append(y_one)
        all_y_chunk.append(chunk)

        # Metadata for analysis only
        all_routes.append(
            0 if route == "LEFT" else 1
        )


X = np.stack(all_x)
Y_one = np.stack(all_y_one)
Y_chunk = np.stack(all_y_chunk)
routes = np.asarray(all_routes)

print("X:", X.shape)
print("Y_one:", Y_one.shape)
print("Y_chunk:", Y_chunk.shape)

print(
    "route samples:",
    np.bincount(routes)
)

np.savez_compressed(
    "data/level2_dataset.npz",
    X=X,
    Y_one=Y_one,
    Y_chunk=Y_chunk,
    route=routes,
)