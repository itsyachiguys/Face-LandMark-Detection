from pathlib import Path

import matplotlib.pyplot as plt
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

        self.train_losses = []
        self.val_losses = []

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def train_epoch(self):

        self.model.train()

        running_loss = 0.0

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

        epoch_loss = running_loss / len(self.train_loader)

        self.train_losses.append(epoch_loss)

        return epoch_loss

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0

        for images, landmarks in self.val_loader:

            images = images.to(self.device)
            landmarks = landmarks.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                landmarks,
            )

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.val_loader)

        self.val_losses.append(epoch_loss)

        return epoch_loss

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

            print(
                f"✔ Best model saved "
                f"(Validation Loss = {val_loss:.6f})"
            )

        self.plot_losses()

    def plot_losses(self):

        if len(self.train_losses) == 0:
            return

        output_dir = Path("outputs")
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.train_losses,
            label="Training Loss",
            linewidth=2,
        )

        plt.plot(
            self.val_losses,
            label="Validation Loss",
            linewidth=2,
        )

        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.grid(True)
        plt.legend()

        plt.savefig(
            output_dir / "loss_curve.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()