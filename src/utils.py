"""
utils.py

Utility functions for training and evaluation.
"""
from src.config import DEVICE
import random
import numpy as np
import torch


# ======================================================
# Set Random Seed
# ======================================================

def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ======================================================
# Save Checkpoint
# ======================================================

def save_checkpoint(state: dict, filename: str):
    """
    Save model checkpoint.
    """

    torch.save(state, filename)


# ======================================================
# Load Checkpoint
# ======================================================

def load_checkpoint(filename:str, model, optimizer=None):
    """
    Load checkpoint.
    """

    checkpoint = torch.load(filename, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint


# ======================================================
# Accuracy
# ======================================================

def calculate_accuracy(outputs, labels):
    """
    Calculate classification accuracy.
    """

    _, predictions = torch.max(outputs, dim=1)

    correct = (predictions == labels).sum().item()

    accuracy = correct / labels.size(0)

    return accuracy


# ======================================================
# Average Meter
# ======================================================

class AverageMeter:
    """
    Keeps track of average values.
    """

    def __init__(self):
        self.reset()

    def reset(self):

        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n=1):

        self.sum += value * n

        self.count += n

        self.avg = self.sum / self.count

# ======================================================
# Early Stopping
# ======================================================

class EarlyStopping:
    """
    Stops training when validation loss
    stops improving.
    """

    def __init__(self, patience=5):

        self.patience = patience

        self.best_loss = float("inf")

        self.counter = 0

        self.early_stop = False

    def __call__(self, val_loss):

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.early_stop = True