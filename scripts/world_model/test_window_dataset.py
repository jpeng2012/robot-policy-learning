import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from torchvision.models import ResNet18_Weights
from world_models.data import WorldModelWindowDataset


files = sorted(
    Path("data/level3").glob("left_*.pkl")
)[:2]

transform = (
    ResNet18_Weights.DEFAULT.transforms()
)

dataset = WorldModelWindowDataset(
    trajectory_files=files,
    history_len=1,
    horizon=16,
    image_transform=transform,
)

print(
    "num windows:",
    len(dataset),
)

sample = dataset[0]

for key, value in sample.items():

    if hasattr(
        value,
        "shape",
    ):
        print(
            key,
            tuple(value.shape),
        )

    else:
        print(
            key,
            value,
        )