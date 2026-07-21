import matplotlib.pyplot as plt
import torch

from app.dataset import FaceLandmarkDataset


def main():

    dataset = FaceLandmarkDataset(
        xml_file="data/labels_ibug_300W_train.xml",
        root_dir="data",
        image_size=224,
    )

    image, landmarks = dataset[0]

    image = image.squeeze().numpy()

    landmarks = landmarks.numpy()

    plt.figure(figsize=(6, 6))

    plt.imshow(image, cmap="gray")

    print(
        landmarks.min(),
        landmarks.max(),
    )

    for x, y in landmarks:

        plt.scatter(
            x * 224,
            y * 224,
            c="red",
            s=10,
        )

    plt.title("Augmented Training Sample")

    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()