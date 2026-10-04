
from __future__ import annotations
import argparse, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    feature_root = Path(args.feature_root)

    copied = 0
    missing = 0

    for feat_dir in sorted(
        p for p in feature_root.iterdir()
        if p.is_dir()
    ):
        raw_dir = raw_root / feat_dir.name

        if not raw_dir.is_dir():
            print("[missing raw episode]", raw_dir)
            missing += 1
            continue

        for name in [
            "cube_pos.npy",
            "target_pos.npy",
        ]:
            src = raw_dir / name
            dst = feat_dir / name

            if not src.exists():
                print("[missing]", src)
                missing += 1
                continue

            if dst.exists() and not args.overwrite:
                continue

            shutil.copy2(src, dst)
            copied += 1

    print("copied files:", copied)
    print("missing:", missing)

if __name__ == "__main__":
    main()
