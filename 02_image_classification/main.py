import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def extract_hog_features(img, img_size=(64, 64)):
    img_resized = cv2.resize(img, img_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features


def load_dataset(dataset_path):
    """Load images from a directory structured as dataset/<class_name>/<image>."""
    features, labels = [], []
    for label in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, label)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            features.append(extract_hog_features(img))
            labels.append(label)
    return np.array(features), np.array(labels)


def train_and_evaluate(dataset_path):
    print(f"Loading dataset from: {dataset_path}")
    X, y = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = SVC(kernel="rbf", C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification with HOG + SVM")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    args = parser.parse_args()
    train_and_evaluate(args.dataset)
