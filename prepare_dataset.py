"""
prepare_dataset.py

Prepare the Cats vs Dogs dataset for training.

Directory Structure:

raw_data/
    PetImages/
        Cat/
        Dog/

↓

data/
    train/
        Cat/
        Dog/
    val/
        Cat/
        Dog/
    test/
        Cat/
        Dog/
"""

import os
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# ==========================================================
# Configuration
# ==========================================================

random.seed(42)

RAW_DATA_DIR = Path("raw_data/PetImages")
OUTPUT_DIR = Path("data")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

CLASSES = ["Cat", "Dog"]


# ==========================================================
# Create Output Directories
# ==========================================================

def create_directories():

    for split in ["train", "val", "test"]:

        for class_name in CLASSES:

            directory = OUTPUT_DIR / split / class_name

            directory.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Check Image
# ==========================================================

def is_valid_image(image_path):

    try:

        with Image.open(image_path) as img:
            img.verify()

        return True

    except Exception:

        return False


# ==========================================================
# Copy Images
# ==========================================================

def copy_images(image_list, destination):

    for image_path in tqdm(image_list, desc=f"Copying -> {destination.name}"):

        shutil.copy2(image_path, destination / image_path.name)


# ==========================================================
# Prepare One Class
# ==========================================================

def prepare_class(class_name):

    print(f"\nProcessing {class_name} images...")

    class_dir = RAW_DATA_DIR / class_name

    image_paths = []

    for image_file in class_dir.iterdir():

        if is_valid_image(image_file):

            image_paths.append(image_file)

    print(f"Valid Images: {len(image_paths)}")

    random.shuffle(image_paths)

    total_images = len(image_paths)

    train_end = int(TRAIN_RATIO * total_images)

    val_end = train_end + int(VAL_RATIO * total_images)

    train_images = image_paths[:train_end]
    val_images = image_paths[train_end:val_end]
    test_images = image_paths[val_end:]

    copy_images(
        train_images,
        OUTPUT_DIR / "train" / class_name
    )

    copy_images(
        val_images,
        OUTPUT_DIR / "val" / class_name
    )

    copy_images(
        test_images,
        OUTPUT_DIR / "test" / class_name
    )

    print("\nSummary")

    print(f"Train : {len(train_images)}")

    print(f"Validation : {len(val_images)}")

    print(f"Test : {len(test_images)}")


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 50)
    print("Preparing Dataset")
    print("=" * 50)

    create_directories()

    for class_name in CLASSES:

        prepare_class(class_name)

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()