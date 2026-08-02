"""
model.py

Creates deep learning models.
"""

import torch.nn as nn
from torchvision import models

from src.config import MODEL_NAME, NUM_CLASSES


def build_model() -> nn.Module:

    if MODEL_NAME == "ResNet18":

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

    else:
        raise ValueError(
            f"Unsupported model: {MODEL_NAME}"
        )

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    # Train classifier
    for param in model.fc.parameters():
        param.requires_grad = True

    return model