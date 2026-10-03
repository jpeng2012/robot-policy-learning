from pathlib import Path
import json
import numpy as np


def verify_episode(episode_dir):

    episode_dir = Path(episode_dir)

    with open(episode_dir / "metadata.json") as f:
        metadata = json.load(f)

    arrays = {
        name: np.load(episode_dir / f"{name}.npy")
        for name in [
            "actions",
            "task_state",
            "robot_config",
            "cube_pos",
            "target_pos",
        ]
    }

    T = len(arrays["actions"])

    assert arrays["actions"].shape == (T, 7)
    assert arrays["task_state"].shape == (T + 1, 9)
    assert arrays["robot_config"].shape == (T + 1, 7)
    assert arrays["cube_pos"].shape == (T + 1, 3)
    assert arrays["target_pos"].shape == (T + 1, 3)

    assert len(list((episode_dir / "agent").glob("*.jpg"))) == T + 1
    assert len(list((episode_dir / "wrist").glob("*.jpg"))) == T + 1

    assert all(np.isfinite(a).all() for a in arrays.values())

    assert np.all(np.abs(arrays["actions"]) <= 1.00001)

    assert metadata["num_actions"] == T

    print(
        episode_dir.name,
        f"T={T}",
        f"success={metadata['success']}",
        f"failure={metadata.get('failure_type')}",
        "PASS",
    )

print("start")
for episode_dir in sorted(
    Path("data/policy_rollouts").glob("flowmatching_*")
):
    verify_episode(episode_dir)



from world_models.data import (
    WorldModelFeatureWindowDataset
)


root = Path(
    "data/policy_wm_features_vjepa21"
)

trajectory_dirs = sorted(
    p
    for p in root.iterdir()
    if p.is_dir()
)

dataset = WorldModelFeatureWindowDataset(
    trajectory_dirs=trajectory_dirs,
    horizon=16,
)

print(
    "Number of windows:",
    len(dataset),
)

sample = dataset[0]

for key, value in sample.items():

    if hasattr(value, "shape"):
        print(
            key,
            tuple(value.shape),
        )