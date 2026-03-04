import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.image_utils import load_image, to_rgb


def segment_image(image_path):
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh_global = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(to_rgb(img))
    axes[0].set_title("Original")
    axes[1].imshow(thresh_global, cmap="gray")
    axes[1].set_title("Global Threshold")
    axes[2].imshow(thresh_otsu, cmap="gray")
    axes[2].set_title("Otsu's Threshold")
    axes[3].imshow(thresh_adaptive, cmap="gray")
    axes[3].set_title("Adaptive Threshold")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Segmentation")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()
    segment_image(args.image)
