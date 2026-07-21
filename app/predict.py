import os

import cv2
import matplotlib.pyplot as plt

from app.face_detector import FaceDetector
from app.inference import LandmarkPredictor


IMAGE_PATH = "assets/test.jpg"
MODEL_PATH = "models/checkpoints/best_model.pth"
OUTPUT_PATH = "outputs/prediction.png"


def main():

    detector = FaceDetector()
    predictor = LandmarkPredictor(MODEL_PATH)

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print(f"Could not load image: {IMAGE_PATH}")
        return

    faces = detector.detect(image)

    if len(faces) == 0:
        print("No face detected.")
        return

    x, y, w, h = faces[0]

    # Draw face bounding box
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2,
    )

    face = image[y:y + h, x:x + w]

    prediction = predictor.predict(face)

    for px, py in prediction:

        px = int(px * w + x)
        py = int(py * h + y)

        cv2.circle(
            image,
            (px, py),
            2,
            (0, 0, 255),
            -1,
        )

    os.makedirs("outputs", exist_ok=True)

    # Save image before converting to RGB
    cv2.imwrite(OUTPUT_PATH, image)

    print(f"Prediction saved to: {OUTPUT_PATH}")

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("68 Facial Landmarks")

    plt.show()


if __name__ == "__main__":
    main()