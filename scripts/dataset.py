# scripts/dataset.py

import os

import imageio.v3 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class LiverSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = imageio.imread(img_path)
        mask = imageio.imread(mask_path)

        # Expand grayscale to 3-channel (needed for SegFormer)
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=0)  # (3, H, W)
        else:
            raise ValueError(f"Expected grayscale input, got shape {image.shape}")

        mask = torch.from_numpy(mask).long()  # (H, W), class labels as integers
        image = torch.from_numpy(image).float() / 255.0  # Normalize to [0,1]

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
