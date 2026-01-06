#!/usr/bin/env python3
"""
Train RF-DETR model on a custom dataset.

Usage:
    1. Prepare your dataset using prepare_dataset.py
    2. Update CONFIG below
    3. Run: python train.py
"""
from rfdetr import RFDETRMedium  # or RFDETRNano, RFDETRSmall, RFDETRLarge
import warnings
warnings.filterwarnings("ignore", message=".*cuda capability.*")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Dataset
    "dataset_dir": "./merged_dataset",
    "output_dir": "./output/detector",

    # Training
    "epochs": 100,
    "batch_size": 16,
    "grad_accum_steps": 1,
    "lr": 1e-4,
    "resolution": 576,

    # Logging (optional)
    "wandb": False,
    "wandb_project": "rfdetr-training",
    "wandb_run": "run-1",
    "tensorboard": True,

    # Early stopping
    "early_stopping": True,
}

# =============================================================================
# TRAINING
# =============================================================================

if __name__ == "__main__":
    # Initialize model
    model = RFDETRMedium(resolution=CONFIG["resolution"])

    # Train
    model.train(
        dataset_dir=CONFIG["dataset_dir"],
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        grad_accum_steps=CONFIG["grad_accum_steps"],
        lr=CONFIG["lr"],
        output_dir=CONFIG["output_dir"],
        wandb=CONFIG["wandb"],
        project=CONFIG["wandb_project"] if CONFIG["wandb"] else None,
        run=CONFIG["wandb_run"] if CONFIG["wandb"] else None,
        early_stopping=CONFIG["early_stopping"],
        tensorboard=CONFIG["tensorboard"],
    )
