import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # isort:skip
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from scripts.dataset import LiverSegmentationDataset
from scripts.utils.mlflow_logger import MLFlowLogger


def compute_mean_iou(preds, labels, num_classes):
    preds = preds.flatten()
    labels = labels.flatten()
    from sklearn.metrics import jaccard_score

    return jaccard_score(
        labels, preds, average="macro", labels=list(range(num_classes))
    )


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        outputs = model(pixel_values=images).logits
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


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best trial from Optuna study
    # If saving study in storage:
    # study = optuna.load_study(study_name="liver_segmentation_tuning", storage="sqlite:///optuna_study.db")
    # best_params = study.best_trial.params
    # Since I printed out best params:
    best_params = {
        "learning_rate": 1.5783884776573395e-05,
        "tumor_weight": 2.2290977491083463,
        "weight_decay": 0.0024270420812147852,
    }

    # Logger (optional)
    logger = MLFlowLogger(
        experiment_name="liver_segmentation_final_training", run_name="final_run"
    )

    # Datasets
    train_dataset = LiverSegmentationDataset(
        images_dir="data/processed/train/images/",
        masks_dir="data/processed/train/masks/",
    )
    val_dataset = LiverSegmentationDataset(
        images_dir="data/processed/val/images/", masks_dir="data/processed/val/masks/"
    )

    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    train_loader = DataLoader(
        combined_dataset, batch_size=8, shuffle=True, num_workers=2
    )

    # Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=3,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    model.to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
    )

    tumor_weight = best_params["tumor_weight"]
    class_weights = torch.tensor([1.0, 1.0, tumor_weight], dtype=torch.float32).to(
        device
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")

    for epoch in range(30):
        print(f"Epoch {epoch+1}/30")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        print(f"Train Loss: {train_loss:.4f}")

        logger.log_metrics({"train_loss": train_loss}, step=epoch)

    # Save final best model
    save_path = "models/best_model.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.log_artifact(save_path)

    logger.end_run()


if __name__ == "__main__":
    main()
