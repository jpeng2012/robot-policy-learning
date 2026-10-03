from pathlib import Path
import numpy as np


def load_arrays(root, name):

    arrays = []

    for episode in sorted(
        Path(root).iterdir()
    ):
        if not episode.is_dir():
            continue

        path = episode / f"{name}.npy"

        if path.exists():
            arrays.append(
                np.load(path)
            )

    return np.concatenate(
        arrays,
        axis=0,
    )


expert_root = (
    "data/level3_wm_features_vjepa21_vitb384"
)

fm_root = (
    "data/policy_wm_features_vjepa21_df"
)


for name in [
    "task_state",
    "robot_config",
    "actions",
]:

    expert = load_arrays(
        expert_root,
        name,
    )

    fm = load_arrays(
        fm_root,
        name,
    )

    print()
    print("====", name, "====")

    print("expert mean:")
    print(
        np.round(
            expert.mean(axis=0),
            4,
        )
    )

    print("FM mean:")
    print(
        np.round(
            fm.mean(axis=0),
            4,
        )
    )

    print("expert std:")
    print(
        np.round(
            expert.std(axis=0),
            4,
        )
    )

    print("FM std:")
    print(
        np.round(
            fm.std(axis=0),
            4,
        )
    )

    print("expert min:")
    print(
        np.round(
            expert.min(axis=0),
            4,
        )
    )

    print("FM min:")
    print(
        np.round(
            fm.min(axis=0),
            4,
        )
    )

    print("expert max:")
    print(
        np.round(
            expert.max(axis=0),
            4,
        )
    )

    print("FM max:")
    print(
        np.round(
            fm.max(axis=0),
            4,
        )
    )

    # How OOD is FM under expert normalization?
    mean = expert.mean(
        axis=0
    )

    std = expert.std(
        axis=0
    )

    std = np.maximum(
        std,
        1e-6,
    )

    fm_z = (
        fm - mean
    ) / std

    print("FM |z| mean:")
    print(
        np.round(
            np.abs(fm_z).mean(axis=0),
            2,
        )
    )

    print("FM |z| p99:")
    print(
        np.round(
            np.percentile(
                np.abs(fm_z),
                99,
                axis=0,
            ),
            2,
        )
    )