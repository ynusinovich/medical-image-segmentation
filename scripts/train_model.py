# scripts/train_model.py

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # isort:skip
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from scripts.dataset import LiverSegmentationDataset
from scripts.utils.mlflow_logger import MLFlowLogger


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.rand() < self.p:
            # Flip width dimension
            image = torch.flip(image, dims=[2])  # (C, H, W) → flip W
            mask = torch.flip(mask, dims=[1])  # (H, W) → flip W
        return image, mask


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def compute_mean_iou(preds, labels, num_classes):
    preds = preds.flatten()
    labels = labels.flatten()
    return jaccard_score(
        labels, preds, average="macro", labels=list(range(num_classes))
    )


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        outputs = model(pixel_values=images).logits  # (B, C, H, W)

        # Upsample to match target size
        outputs = torch.nn.functional.interpolate(
            outputs,
            size=(masks.shape[-2], masks.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def compute_class_iou(preds, labels, num_classes):
    preds = preds.flatten()
    labels = labels.flatten()
    ious = []
    for cls in range(num_classes):
        iou = jaccard_score(
            labels == cls, preds == cls, average="binary", zero_division=0
        )
        ious.append(iou)
    return ious


def validate(model, dataloader, loss_fn, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    tumor_ious = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images, masks = images.to(device), masks.to(device)

            outputs = model(pixel_values=images).logits
            outputs = torch.nn.functional.interpolate(
                outputs,
                size=(masks.shape[-2], masks.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            batch_ious = compute_class_iou(preds.cpu(), masks.cpu(), num_classes)
            total_iou += np.mean(batch_ious)

            tumor_ious.append(batch_ious[2])  # Class 2 = tumor

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_tumor_iou = np.mean(tumor_ious)

    return avg_loss, avg_iou, avg_tumor_iou


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up MLFlow logger
    logger = MLFlowLogger(
        experiment_name=config["experiment_name"], run_name=config["run_name"]
    )

    # Log hyperparams
    logger.log_params(
        {
            "architecture": config["model"]["architecture"],
            "backbone": config["model"]["backbone"],
            "epochs": config["train"]["epochs"],
            "batch_size": config["train"]["batch_size"],
            "learning_rate": config["train"]["learning_rate"],
        }
    )

    # Transforms
    train_transform = RandomHorizontalFlip(p=0.5)
    val_transform = None

    # Dataset and Dataloader
    train_dataset = LiverSegmentationDataset(
        images_dir="data/processed/train/images/",
        masks_dir="data/processed/train/masks/",
        transform=train_transform,
    )
    val_dataset = LiverSegmentationDataset(
        images_dir="data/processed/val/images/",
        masks_dir="data/processed/val/masks/",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    # Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        config["model"]["backbone"],
        num_labels=config["model"]["output_classes"],
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    model.to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"].get("weight_decay", 0.0),
    )

    # Class weights: [background, liver, tumor]
    class_weights = torch.tensor(
        [0.2, 0.3, config["loss"]["tumor_weight"]], device=device
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Training Loop
    best_val_iou = 0.0

    for epoch in range(config["train"]["epochs"]):
        print(f"Epoch {epoch+1}/{config['train']['epochs']}")

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_iou, val_tumor_iou = validate(
            model, val_loader, loss_fn, device, config["model"]["output_classes"]
        )

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f} | Val Tumor IoU: {val_tumor_iou:.4f}"
        )

        # Log metrics to MLFlow
        logger.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mIoU": val_iou,
                "val_tumor_IoU": val_tumor_iou,
            },
            step=epoch,
        )

        # Save best model based on tumor IoU
        if val_tumor_iou > best_val_iou:
            best_val_iou = val_tumor_iou
            save_path = (
                f"models/best_model_trial_{config.get('trial_number', 'final')}.pth"
            )
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.log_artifact(save_path)

    logger.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
