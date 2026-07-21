import os
import xml.etree.ElementTree as ET

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

from app.transforms import get_train_transforms


class FaceLandmarkDataset(Dataset):

    def __init__(
        self,
        xml_file,
        root_dir="data",
        image_size=224,
        transform=None,
    ):

        self.root_dir = root_dir
        self.image_size = image_size

        if transform is None:
            transform = get_train_transforms(image_size)

        self.transform = transform

        self.image_paths = []
        self.landmarks = []
        self.bounding_boxes = []

        self._load_annotations(xml_file)

    def _load_annotations(self, xml_file):

        tree = ET.parse(xml_file)
        root = tree.getroot()

        images = root.find("images")

        if images is None:
            raise ValueError("Could not find <images> in XML.")

        for image in images.findall("image"):

            image_path = os.path.join(
                self.root_dir,
                image.attrib["file"],
            )

            box = image.find("box")

            left = int(float(box.attrib["left"]))
            top = int(float(box.attrib["top"]))
            width = int(float(box.attrib["width"]))
            height = int(float(box.attrib["height"]))

            points = []

            for part in box.findall("part"):

                x = float(part.attrib["x"])
                y = float(part.attrib["y"])

                points.append((x, y))

            self.image_paths.append(image_path)
            self.landmarks.append(points)
            self.bounding_boxes.append(
                (
                    left,
                    top,
                    width,
                    height,
                )
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image = Image.open(
            self.image_paths[index]
        ).convert("L")

        image = np.array(image)

        left, top, width, height = self.bounding_boxes[index]

        # -------------------------------------------------
        # Expand bounding box (20% padding)
        # -------------------------------------------------

        padding = 0.20

        pad_x = int(width * padding)
        pad_y = int(height * padding)

        left = max(0, left - pad_x)
        top = max(0, top - pad_y)

        right = min(
            image.shape[1],
            left + width + (2 * pad_x),
        )

        bottom = min(
            image.shape[0],
            top + height + (2 * pad_y),
        )

        image = image[
            top:bottom,
            left:right,
        ]

        crop_w = right - left
        crop_h = bottom - top

        landmarks = np.array(
            self.landmarks[index],
            dtype=np.float32,
        )

        landmarks[:, 0] -= left
        landmarks[:, 1] -= top

        transformed = self.transform(
            image=image,
            keypoints=landmarks.tolist(),
        )

        image = transformed["image"]

        landmarks = np.array(
            transformed["keypoints"],
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Normalize using transformed image dimensions
        # -------------------------------------------------

        _, h, w = image.shape

        landmarks[:, 0] /= w
        landmarks[:, 1] /= h

        # -------------------------------------------------
        # Keep coordinates inside image
        # -------------------------------------------------

        landmarks[:, 0] = np.clip(
            landmarks[:, 0],
            0.0,
            1.0,
        )

        landmarks[:, 1] = np.clip(
            landmarks[:, 1],
            0.0,
            1.0,
        )

        landmarks = torch.tensor(
            landmarks,
            dtype=torch.float32,
        )

        return image, landmarks