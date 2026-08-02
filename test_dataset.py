from src.dataset import get_dataloaders


def main():

    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print("Classes:", class_names)

    images, labels = next(iter(train_loader))

    print("Image Batch Shape:", images.shape)
    print("Label Batch Shape:", labels.shape)


if __name__ == "__main__":
    main()