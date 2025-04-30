import os
import subprocess

import optuna
import torch
import yaml


def objective(trial):

    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    tumor_weight = trial.suggest_uniform("tumor_weight", 1.0, 3.0)
    weight_decay = trial.suggest_uniform("weight_decay", 0.0, 0.01)

    # Update config dynamically
    config = {
        "experiment_name": "liver_segmentation_tuning",
        "run_name": f"trial_{trial.number}",
        "train": {
            "epochs": 30,
            "batch_size": 8,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "image_size": 256,
        },
        "model": {
            "architecture": "segformer",
            "backbone": "nvidia/segformer-b0-finetuned-ade-512-512",
            "input_channels": 3,
            "output_classes": 3,
        },
        "loss": {"tumor_weight": tumor_weight},
    }

    # Save temporary config
    os.makedirs("configs/trials", exist_ok=True)
    trial_config_path = f"configs/trials/trial_{trial.number}.yaml"
    with open(trial_config_path, "w") as f:
        yaml.dump(config, f)

    # Run training subprocess
    result = subprocess.run(
        [
            "pipenv",
            "run",
            "python",
            "scripts/train_model.py",
            "--config",
            trial_config_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Trial {trial.number} failed:")
        print(result.stderr)
        return 0.0

    # Parse MLFlow logs or extract best val_tumor_IoU
    # (easier: parse final printout)
    stdout_lines = result.stdout.split("\n")
    best_val_tumor_iou = None
    for line in stdout_lines:
        if "Val Tumor IoU" in line:
            parts = line.strip().split("|")
            for p in parts:
                if "Val Tumor IoU" in p:
                    best_val_tumor_iou = float(p.strip().split(":")[-1])
    if best_val_tumor_iou is None:
        best_val_tumor_iou = 0.0  # Fail-safe

    return best_val_tumor_iou


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    print(study.best_trial.params)
    print(f"Best val_tumor_IoU: {study.best_value:.4f}")
