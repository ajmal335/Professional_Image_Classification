"""
engine.py

Contains training and validation loops.
"""

import torch
from tqdm import tqdm

from src.utils import AverageMeter, calculate_accuracy


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one epoch.
    """

    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        accuracy = calculate_accuracy(outputs, labels)

        loss_meter.update(
            loss.item(),
            images.size(0)
        )

        accuracy_meter.update(
            accuracy,
            images.size(0)
        )

        progress_bar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{accuracy_meter.avg:.4f}"
        )

    return loss_meter.avg, accuracy_meter.avg

def validate(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Validate the model.
    """

    model.eval()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            leave=False
        )

        for images, labels in progress_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            accuracy = calculate_accuracy(
                outputs,
                labels
            )

            loss_meter.update(
                loss.item(),
                images.size(0)
            )

            accuracy_meter.update(
                accuracy,
                images.size(0)
            )

            progress_bar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                acc=f"{accuracy_meter.avg:.4f}"
            )

    return loss_meter.avg, accuracy_meter.avg