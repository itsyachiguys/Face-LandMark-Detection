import torch

from models.resnet import FaceLandmarkModel


def main():
    model = FaceLandmarkModel()

    dummy = torch.randn(4, 1, 224, 224)

    output = model(dummy)

    print("=" * 40)
    print("Model Test")
    print("=" * 40)
    print("Input :", dummy.shape)
    print("Output:", output.shape)


if __name__ == "__main__":
    main()