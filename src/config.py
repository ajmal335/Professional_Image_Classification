"""
Project Configuration File

This file contains all configurable parameters used throughout the project.
Changing values here updates the entire project.
"""

from pathlib import Path
import torch

# ======================================================        
# Project Paths
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ======================================================
# Dataset
# ======================================================

IMAGE_SIZE = 224

NUM_CLASSES = 2

MODEL_NAME = "ResNet18"

BATCH_SIZE = 32

NUM_WORKERS = 0

PIN_MEMORY = True

# ======================================================
# Training
# ======================================================

EPOCHS = 3

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

SEED = 42

# ======================================================
# Device
# ======================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ======================================================
# Mixed Precision
# ======================================================

USE_AMP = torch.cuda.is_available()

# ======================================================
# Early Stopping
# ======================================================

PATIENCE = 5

# ======================================================
# Checkpoints
# ======================================================

BEST_MODEL_NAME = "best_model.pth"

LAST_MODEL_NAME = "last_model.pth"