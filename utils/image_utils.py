import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path):
    """Load an image from disk using OpenCV (BGR format)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def to_rgb(img_bgr):
    """Convert a BGR image (OpenCV) to RGB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_grayscale(img_bgr):
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def show_image(img, title="Image", cmap=None):
    """Display an image using matplotlib."""
    plt.figure()
    plt.title(title)
    if len(img.shape) == 2 or cmap is not None:
        plt.imshow(img, cmap=cmap or "gray")
    else:
        plt.imshow(to_rgb(img))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_image(img, path):
    """Save an image to disk."""
    cv2.imwrite(path, img)
