import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from app.dataset import FaceLandmarkDataset
from app.transforms import (
    get_train_transforms,
    get_test_transforms,
)
from app.trainer import Trainer

from models.resnet import FaceLandmarkModel

from configs.train_config import *


def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using Device : {device}")

    train_dataset = FaceLandmarkDataset(
        xml_file=TRAIN_XML,
        root_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        transform=get_train_transforms(IMAGE_SIZE),
    )

    val_dataset = FaceLandmarkDataset(
        xml_file=TEST_XML,
        root_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        transform=get_test_transforms(IMAGE_SIZE),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = FaceLandmarkModel()

    criterion = nn.SmoothL1Loss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = trainer.train_epoch()

        val_loss = trainer.validate()

        scheduler.step(val_loss)

        trainer.save_best(epoch + 1, val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Train Loss : {train_loss:.6f}"
        )

        print(
            f"Val Loss   : {val_loss:.6f}"
        )

    return history


if __name__ == "__main__":
    train()