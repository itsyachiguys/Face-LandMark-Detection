import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss

from app.dataset import FaceLandmarkDataset
from models.resnet import FaceLandmarkModel


MODEL_PATH = "models/checkpoints/best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    print("=" * 40)
    print("Loading Test Dataset...")
    print("=" * 40)

    test_dataset = FaceLandmarkDataset(
        xml_file="data/labels_ibug_300W_test.xml",
        root_dir="data",
        image_size=224,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    print(f"Test Images : {len(test_dataset)}")

    model = FaceLandmarkModel()

    # -----------------------------
    # Load trained model
    # -----------------------------
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    # Supports both checkpoint and state_dict files
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from Epoch {checkpoint.get('epoch', 'Unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state_dict.")

    model.to(DEVICE)
    model.eval()

    criterion = MSELoss()

    total_loss = 0.0

    print("\nEvaluating...\n")

    with torch.no_grad():

        for images, landmarks in test_loader:

            images = images.to(DEVICE)
            landmarks = landmarks.to(DEVICE)

            predictions = model(images)

            loss = criterion(
                predictions,
                landmarks,
            )

            total_loss += loss.item()

    average_mse = total_loss / len(test_loader)
    average_rmse = average_mse ** 0.5

    print("=" * 40)
    print("Evaluation Results")
    print("=" * 40)
    print(f"Device       : {DEVICE}")
    print(f"Test Images  : {len(test_dataset)}")
    print(f"Average MSE  : {average_mse:.6f}")
    print(f"Average RMSE : {average_rmse:.6f}")
    print("=" * 40)


if __name__ == "__main__":
    main()