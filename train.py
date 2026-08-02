"""
train.py

Main training script.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
)

from src.dataset import get_dataloaders
from src.model import build_model
from src.engine import train_one_epoch, validate
from src.utils import (
    set_seed,
    save_checkpoint,
    EarlyStopping,
)


def main():

    # ----------------------------
    # Set random seed
    # ----------------------------

    set_seed()

    # ----------------------------
    # Load Dataset
    # ----------------------------

    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print(f"Classes: {class_names}")

    # ----------------------------
    # Build Model
    # ----------------------------

    model = build_model()

    model.to(DEVICE)
    print(f"Using device: {DEVICE}")

    # ----------------------------
    # Loss Function
    # ----------------------------

    criterion = nn.CrossEntropyLoss()

    # ----------------------------
    # Optimizer
    # ----------------------------

    optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

    # ----------------------------
    # Early Stopping
    # ----------------------------

    early_stopping = EarlyStopping(
        patience=PATIENCE
    )

    best_loss = float("inf")

    # ----------------------------
    # Training History
    # ----------------------------

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    # ----------------------------
    # Training Loop
    # ----------------------------

    for epoch in range(EPOCHS):

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save history

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # ----------------------------
        # Save Best Model
        # ----------------------------

        if val_loss < best_loss:

            best_loss = val_loss

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                CHECKPOINT_DIR / BEST_MODEL_NAME,
            )

            print(f"✅ Best model saved to {CHECKPOINT_DIR / BEST_MODEL_NAME}")

        # ----------------------------
        # Early Stopping
        # ----------------------------

        early_stopping(val_loss)

        if early_stopping.early_stop:

            print("Early stopping triggered.")

            break

    print("\nTraining Finished!")
    
    print("\nTraining History")

    print("Train Losses:", train_losses)
    print("Validation Losses:", val_losses)

    print("Train Accuracies:", train_accuracies)
    print("Validation Accuracies:", val_accuracies)


if __name__ == "__main__":
    main()