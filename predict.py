"""
predict.py

Predict a single image using the trained model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from src.config import (
    DEVICE,
    IMAGE_SIZE,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
)
from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import load_checkpoint


def main():

    # ---------------------------------
    # Load class names
    # ---------------------------------

    _, _, _, class_names = get_dataloaders()

    # ---------------------------------
    # Build model
    # ---------------------------------

    model = build_model().to(DEVICE)

    load_checkpoint(
        CHECKPOINT_DIR / BEST_MODEL_NAME,
        model
    )

    model.eval()

    # ---------------------------------
    # Image path
    # ---------------------------------

    image_path = input(
        "\nEnter image path: "
    ).strip()

    image_path = Path(image_path)

    if not image_path.exists():
        print("Image not found!")
        return

    # ---------------------------------
    # Original Image
    # ---------------------------------

    original_image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = transform(original_image).unsqueeze(0).to(DEVICE)

    # ---------------------------------
    # Prediction
    # ---------------------------------

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = class_names[prediction.item()]

    confidence = confidence.item() * 100

    # ---------------------------------
    # Print
    # ---------------------------------

    print("\n==============================")
    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.2f}%")
    print("==============================")

    # ---------------------------------
    # Show image
    # ---------------------------------

    plt.figure(figsize=(6,6))

    plt.imshow(original_image)

    plt.title(
        f"{predicted_class}\nConfidence: {confidence:.2f}%"
    )

    plt.axis("off")

    plt.savefig(
        "outputs/prediction.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()