import matplotlib.pyplot as plt

from app.dataset import FaceLandmarkDataset


def main():

    dataset = FaceLandmarkDataset(
        xml_file="data/labels_ibug_300W_train.xml",
        root_dir="data",
    )

    image, landmarks = dataset[0]

    image = image.squeeze().numpy()

    plt.figure(figsize=(6, 6))

    plt.imshow(image, cmap="gray")

    plt.scatter(
        landmarks[:, 0] * image.shape[1],
        landmarks[:, 1] * image.shape[0],
        c="red",
        s=8,
    )

    plt.title("Sample Face with 68 Landmarks")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()