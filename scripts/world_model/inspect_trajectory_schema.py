from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path


def find_alias(mapping, *names):
    for name in names:
        if name in mapping:
            return name
    return None


def inspect_file(path: Path) -> bool:
    with path.open("rb") as f:
        demo = pickle.load(f)

    trajectory = demo.get("trajectory", [])

    if not trajectory:
        print(f"[EMPTY] {path}")
        return False

    step = trajectory[0]
    obs = step.get("obs", {})

    checks = {
        "joint_pos": find_alias(
            obs,
            "joint_pos",
            "robot0_joint_pos",
        ),
        "eef_pos": find_alias(
            obs,
            "eef_pos",
            "robot0_eef_pos",
        ),
        "eef_quat": find_alias(
            obs,
            "eef_quat",
            "robot0_eef_quat",
        ),
        "gripper_qpos": find_alias(
            obs,
            "gripper_qpos",
            "robot0_gripper_qpos",
        ),
        "cube_pos": find_alias(obs, "cube_pos"),
        "target_pos": find_alias(obs, "target_pos"),
        "barrier_pos": find_alias(obs, "barrier_pos"),
    }

    has_agent = "agent_image" in step
    has_wrist = "wrist_image" in step
    has_action = "action" in step

    compatible = (
        all(
            checks[name] is not None
            for name in [
                "joint_pos",
                "eef_pos",
                "eef_quat",
                "gripper_qpos",
            ]
        )
        and has_agent
        and has_wrist
        and has_action
    )

    print()
    print(path)
    print("  trajectory length:", len(trajectory))
    print("  route:", demo.get("route"))
    print("  image/action fields:")
    print("    agent_image:", has_agent)
    print("    wrist_image:", has_wrist)
    print("    action:", has_action)
    print("  observation fields:")

    for name, alias in checks.items():
        print(
            f"    {name:18s}: "
            + (alias if alias is not None else "MISSING")
        )

    print("  WM-compatible:", "YES" if compatible else "NO")

    return compatible


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "pattern",
        nargs="?",
        default="data/level3/*.pkl",
        help="Glob for demo pickle files.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    files = [
        Path(path)
        for path in sorted(glob.glob(args.pattern))
    ]

    if args.limit > 0:
        files = files[:args.limit]

    if not files:
        raise SystemExit(
            f"No files matched: {args.pattern}"
        )

    num_ok = 0

    for path in files:
        num_ok += int(inspect_file(path))

    print()
    print(f"Compatible: {num_ok}/{len(files)}")


if __name__ == "__main__":
    main()
