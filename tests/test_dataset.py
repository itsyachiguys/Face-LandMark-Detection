from torch.utils.data import DataLoader

from app.dataset import FaceLandmarkDataset


dataset = FaceLandmarkDataset(
    xml_file="data/labels_ibug_300W_train.xml",
    root_dir="data",
)

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print(f"Number of Images : {len(dataset)}")

image, landmarks = dataset[0]

print(f"Image Shape      : {image.shape}")
print(f"Landmarks Shape  : {landmarks.shape}")

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

images, labels = next(iter(loader))

print(f"Batch Images     : {images.shape}")
print(f"Batch Labels     : {labels.shape}")