import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.image_utils import load_image, to_rgb


def detect_faces(image_path):
    """Detect faces using OpenCV's built-in Haar cascade classifier."""
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    result = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(f"Detected {len(faces)} face(s).")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(to_rgb(img))
    axes[0].set_title("Original")
    axes[1].imshow(to_rgb(result))
    axes[1].set_title(f"Detections ({len(faces)} faces)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()
    detect_faces(args.image)
