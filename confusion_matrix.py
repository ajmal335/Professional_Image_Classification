"""
confusion_matrix.py

Generates confusion matrix and classification report.
"""

import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.config import (
    DEVICE,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
)

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import load_checkpoint


def main():

    # ---------------------------------------
    # Load Dataset
    # ---------------------------------------

    _, _, test_loader, class_names = get_dataloaders()

    print("Classes:", class_names)

    # ---------------------------------------
    # Build Model
    # ---------------------------------------

    model = build_model()

    model.to(DEVICE)

    # ---------------------------------------
    # Load Trained Model
    # ---------------------------------------

    load_checkpoint(
        CHECKPOINT_DIR / BEST_MODEL_NAME,
        model
    )

    print("Model Loaded Successfully!")

    model.eval()

    y_true = []
    y_pred = []

    # ---------------------------------------
    # Prediction Loop
    # ---------------------------------------

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            y_true.extend(labels.numpy())

            y_pred.extend(
                predictions.cpu().numpy()
            )

    # ---------------------------------------
    # Confusion Matrix
    # ---------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    plt.figure(figsize=(6,6))

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.savefig(
        "outputs/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ---------------------------------------
    # Classification Report
    # ---------------------------------------

    print("\nClassification Report\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )


if __name__ == "__main__":
    main()