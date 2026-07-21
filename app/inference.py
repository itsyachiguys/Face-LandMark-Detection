import cv2
import numpy as np
import torch

from app.transforms import get_test_transforms
from models.resnet import FaceLandmarkModel


class LandmarkPredictor:

    def __init__(self, model_path):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = FaceLandmarkModel()

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = get_test_transforms(224)

    @torch.no_grad()
    def predict(self, face):

        # Convert BGR → Grayscale
        gray = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY,
        )

        # Albumentations expects numpy array
        transformed = self.transform(
            image=gray,
        )

        tensor = transformed["image"]

        tensor = tensor.unsqueeze(0).to(
            self.device
        )

        prediction = self.model(tensor)

        prediction = (
            prediction
            .squeeze(0)
            .cpu()
            .numpy()
        )

        return prediction