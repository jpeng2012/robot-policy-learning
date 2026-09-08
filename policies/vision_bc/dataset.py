import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class VisionBCDataset(Dataset):
    def __init__(
            self, 
            dataset_npz,
            data_root,
            split="train",
            transform=None,
            proprio_mean=None,
            proprio_std=None
            ):
        data = np.load(dataset_npz, allow_pickle=False)

        self.data_root = data_root
        self.transform = transform
        self.proprio_mean = proprio_mean
        self.proprio_std = proprio_std

        if split == "train":
            self.agent_paths = data["agent_train"]
            self.wrist_paths = data["wrist_train"]
            self.actions = data["Y_one_train"].astype(np.float32)
            self.states = data["state_train"].astype(np.int64)
            self.proprio = data["proprio_train"].astype(np.float32)
        elif split == "val":
            self.agent_paths = data["agent_val"]
            self.wrist_paths = data["wrist_val"]
            self.actions = data["Y_one_val"].astype(np.float32)
            self.states = data["state_val"].astype(np.int64)
            self.proprio = data["proprio_val"].astype(np.float32)
        else:
            raise ValueError(f"Invalid split: {split}")

        self.vision_history_len = self.agent_paths.shape[1]
        self.proprio_history_len = self.proprio.shape[1]

    def __len__(self):
        return len(self.actions)

    def _load_image(self, rel_path):
        image_path = os.path.join(self.data_root, str(rel_path))
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        agent_image = []
        wrist_image = []

        for h in range(self.vision_history_len):
            agent_image.append(self._load_image(self.agent_paths[idx, h]))
            wrist_image.append(self._load_image(self.wrist_paths[idx, h]))

        #(T, C, H, W)
        agent_image = torch.stack(agent_image, dim=0)
        wrist_image = torch.stack(wrist_image, dim=0)
        proprio = self.proprio[idx]
        if self.proprio_mean is not None and self.proprio_std is not None:
            proprio = (proprio - self.proprio_mean) / self.proprio_std
        proprio = torch.from_numpy(proprio.astype(np.float32))
        action = torch.from_numpy(self.actions[idx])
        state = torch.tensor(self.states[idx], dtype=torch.long)

        return {
            "agent": agent_image,
            "wrist": wrist_image,
            "proprio": proprio,
            "action": action,
            "state": state,
        }


class VisionChunkBCDataset(Dataset):
    def __init__(
            self, 
            dataset_npz,
            data_root,
            split="train",
            transform=None,
            proprio_mean=None,
            proprio_std=None
            ):
        data = np.load(dataset_npz, allow_pickle=False)

        self.data_root = data_root
        self.transform = transform
        self.proprio_mean = proprio_mean
        self.proprio_std = proprio_std

        if split == "train":
            self.agent_paths = data["agent_train"]
            self.wrist_paths = data["wrist_train"]
            self.actions = data["Y_chunk_train"].astype(np.float32)
            self.states = data["state_train"].astype(np.int64)
            self.proprio = data["proprio_train"].astype(np.float32)
        elif split == "val":
            self.agent_paths = data["agent_val"]
            self.wrist_paths = data["wrist_val"]
            self.actions = data["Y_chunk_val"].astype(np.float32)
            self.states = data["state_val"].astype(np.int64)
            self.proprio = data["proprio_val"].astype(np.float32)
        else:
            raise ValueError(f"Invalid split: {split}")

        self.vision_history_len = self.agent_paths.shape[1]
        self.proprio_history_len = self.proprio.shape[1]

    def __len__(self):
        return len(self.actions)

    def _load_image(self, rel_path):
        image_path = os.path.join(self.data_root, str(rel_path))
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        agent_image = []
        wrist_image = []

        for h in range(self.vision_history_len):
            agent_image.append(self._load_image(self.agent_paths[idx, h]))
            wrist_image.append(self._load_image(self.wrist_paths[idx, h]))

        #(T, C, H, W)
        agent_image = torch.stack(agent_image, dim=0)
        wrist_image = torch.stack(wrist_image, dim=0)
        proprio = self.proprio[idx]
        if self.proprio_mean is not None and self.proprio_std is not None:
            proprio = (proprio - self.proprio_mean) / self.proprio_std
        proprio = torch.from_numpy(proprio.astype(np.float32))
        action = torch.from_numpy(self.actions[idx])
        state = torch.tensor(self.states[idx], dtype=torch.long)

        return {
            "agent": agent_image,
            "wrist": wrist_image,
            "proprio": proprio,
            "action": action,
            "state": state,
        }