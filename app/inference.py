import cv2
import numpy as np
import torch
from PIL import Image

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

        gray = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY,
        )

        pil = Image.fromarray(gray)

        tensor = self.transform(pil)

        tensor = tensor.unsqueeze(0).to(self.device)

        prediction = self.model(tensor)

        prediction = prediction.squeeze().cpu().numpy()

        return prediction