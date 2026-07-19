import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from app.transforms import get_train_transforms


class FaceLandmarkDataset(Dataset):
    def __init__(self, xml_file, root_dir="data", image_size=224):
        self.root_dir = root_dir
        self.transform = get_train_transforms(image_size)

        self.image_paths = []
        self.landmarks = []
        self.bounding_boxes = []

        self._load_annotations(xml_file)

    def _load_annotations(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        images = root.find("images")

        if images is None:
            raise ValueError("Could not find <images> in XML file.")

        for image in images.findall("image"):

            image_path = os.path.join(
                self.root_dir,
                image.attrib["file"]
            )

            box = image.find("box")

            left = int(box.attrib["left"])
            top = int(box.attrib["top"])
            width = int(box.attrib["width"])
            height = int(box.attrib["height"])

            points = []

            for part in box.findall("part"):
                x = float(part.attrib["x"])
                y = float(part.attrib["y"])
                points.append([x, y])

            self.image_paths.append(image_path)
            self.landmarks.append(points)
            self.bounding_boxes.append(
                (left, top, width, height)
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image = Image.open(
            self.image_paths[index]
        ).convert("L")

        left, top, width, height = self.bounding_boxes[index]

        image = image.crop(
            (
                left,
                top,
                left + width,
                top + height,
            )
        )

        image = self.transform(image)

        landmarks = np.array(
            self.landmarks[index],
            dtype=np.float32,
        )

        landmarks[:, 0] -= left
        landmarks[:, 1] -= top

        landmarks[:, 0] /= width
        landmarks[:, 1] /= height

        landmarks = torch.tensor(
            landmarks,
            dtype=torch.float32,
        )

        return image, landmarks