from pathlib import Path

import torch
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        checkpoint_dir,
    ):

        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.criterion = criterion

        self.device = device

        self.best_loss = float("inf")

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def train_epoch(self):

        self.model.train()

        running_loss = 0

        progress = tqdm(
            self.train_loader,
            desc="Training",
        )

        for images, landmarks in progress:

            images = images.to(self.device)
            landmarks = landmarks.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                landmarks,
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.5f}"
            )

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0

        for images, landmarks in self.val_loader:

            images = images.to(self.device)
            landmarks = landmarks.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                landmarks,
            )

            running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def save_best(self, epoch, val_loss):

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": val_loss,
            }

            torch.save(
                checkpoint,
                self.checkpoint_dir / "best_model.pth",
            )

            print("✔ Best model saved.")