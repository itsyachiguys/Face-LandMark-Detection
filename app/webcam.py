import cv2

from app.face_detector import FaceDetector
from app.inference import LandmarkPredictor


MODEL_PATH = "models/checkpoints/best_model.pth"


def main():

    detector = FaceDetector()
    predictor = LandmarkPredictor(MODEL_PATH)

    camera = cv2.VideoCapture(0)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    frame_count = 0
    faces = []

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame_count += 1

        # Detect face every 5 frames
        if frame_count % 5 == 0:

            detected = detector.detect(frame)

            if len(detected) > 0:
                faces = detected
            else:
                faces = []

        # Draw landmarks using last detected face(s)
        for (x, y, w, h) in faces:

            padding_x = int(w * 0.08)
            padding_y = int(h * 0.08)

            x1 = x + padding_x
            y1 = y + padding_y

            x2 = x + w - padding_x
            y2 = y + h - padding_y

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            prediction = predictor.predict(face)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            for px, py in prediction:

                px = int(px * (x2 - x1) + x1)
                py = int(py * (y2 - y1) + y1)

                cv2.circle(
                    frame,
                    (px, py),
                    2,
                    (0, 0, 255),
                    -1,
                )

        cv2.imshow(
            "Face Landmark Detection",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()