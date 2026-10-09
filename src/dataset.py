import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


TASK_TO_ID = {
    "pick milk": 0,
    "pick bread": 1,
    "pick cereal": 2,
    "pick can": 3,
}

class PairedLocalizationDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)

        path = self.root / f"{split}.jsonl"

        with path.open() as file:
            self.samples = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self.root / sample["image"]
        with Image.open(image_path) as img:
            image_array = np.array(
                img.convert("RGB"), copy=True
            )

        # Convert HWC to CHW and scale pixels to [0, 1]
        image = torch.from_numpy(image_array)
        image = image.permute(2, 0, 1).float() / 255.0

        # Encode the task
        task_id = torch.tensor(
            TASK_TO_ID[sample["task"]],
            dtype=torch.long,
        )

        # Ground-truth location
        target_xy = torch.tensor(
            sample["target_xy"],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "task_id": task_id,
            "target_xy": target_xy,
            "scene_id": sample["scene_id"],
        }
