import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.image_utils import load_image, to_grayscale, show_image, save_image


def process_image(image_path):
    img = load_image(image_path)

    gray = to_grayscale(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("Canny Edges")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic Image Processing")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()
    process_image(args.image)
