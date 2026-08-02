"""
evaluate.py

Evaluate the trained model on the test dataset.
"""

import torch
import torch.nn as nn

from src.config import (DEVICE, CHECKPOINT_DIR, BEST_MODEL_NAME)
from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import load_checkpoint, calculate_accuracy


def main():

    # -------------------------
    # Load Data
    # -------------------------

    _, _, test_loader, class_names = get_dataloaders()

    print(f"Classes: {class_names}")

    # -------------------------
    # Build Model
    # -------------------------

    model = build_model()

    model.to(DEVICE)

    # -------------------------
    # Load Best Model
    # -------------------------

    load_checkpoint(
        f"{CHECKPOINT_DIR}/{BEST_MODEL_NAME}",
        model
    )

    print("Best model loaded successfully!")

    # -------------------------
    # Loss Function
    # -------------------------

    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Evaluation Mode
    # -------------------------

    model.eval()

    total_loss = 0
    total_accuracy = 0
    total_samples = 0

    # -------------------------
    # Testing Loop
    # -------------------------

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            accuracy = calculate_accuracy(
                outputs,
                labels
            )

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_accuracy += accuracy * batch_size
            total_samples += batch_size

    # -------------------------
    # Final Metrics
    # -------------------------

    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples

    print("\n========== Test Results ==========")
    print(f"Test Loss     : {avg_loss:.4f}")
    print(f"Test Accuracy : {avg_accuracy*100:.2f}%")
    print("==================================")


if __name__ == "__main__":
    main()