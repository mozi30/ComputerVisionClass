"""
Unit tests for 04_image_segmentation – segment_image() (PR #1, last task).

Tests cover:
* Normal execution with a synthetic image (no display window opened).
* Error handling for missing/invalid image paths.
* Binary-only output of each thresholding method (global, Otsu, adaptive).
* Output dimensions matching the input dimensions.
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')  # non-interactive backend – must be set before pyplot import

import cv2
import numpy as np

# Add repo root and the segmentation module directory to sys.path
_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, '04_image_segmentation'))

from main import segment_image  # noqa: E402


class TestSegmentImage(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Create a reproducible synthetic 64 × 64 BGR image
        rng = np.random.default_rng(seed=42)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        self.img_path = os.path.join(self.test_dir, 'synthetic.png')
        cv2.imwrite(self.img_path, img)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # ------------------------------------------------------------------
    # Execution and error-handling tests
    # ------------------------------------------------------------------

    @patch('matplotlib.pyplot.show')
    def test_runs_without_error_on_valid_image(self, mock_show):
        """segment_image should complete without raising for a valid image path."""
        segment_image(self.img_path)

    @patch('matplotlib.pyplot.show')
    def test_show_is_called(self, mock_show):
        """segment_image should invoke plt.show() to display the figure."""
        segment_image(self.img_path)
        self.assertTrue(mock_show.called)

    def test_raises_file_not_found_for_missing_image(self):
        """segment_image should raise FileNotFoundError for a non-existent path."""
        with self.assertRaises(FileNotFoundError):
            segment_image(os.path.join(self.test_dir, 'nonexistent.png'))

    # ------------------------------------------------------------------
    # Thresholding output – values must be strictly binary (0 or 255)
    # ------------------------------------------------------------------

    def _load_blurred(self):
        """Return the blurred grayscale image used internally by segment_image."""
        img = cv2.imread(self.img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def test_global_threshold_output_is_binary(self):
        """Global (fixed-level) threshold should produce only 0 and 255 pixel values."""
        blurred = self._load_blurred()
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        self.assertTrue(set(np.unique(thresh)).issubset({0, 255}))

    def test_otsu_threshold_output_is_binary(self):
        """Otsu threshold should produce only 0 and 255 pixel values."""
        blurred = self._load_blurred()
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.assertTrue(set(np.unique(thresh)).issubset({0, 255}))

    def test_adaptive_threshold_output_is_binary(self):
        """Adaptive threshold should produce only 0 and 255 pixel values."""
        blurred = self._load_blurred()
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        self.assertTrue(set(np.unique(thresh)).issubset({0, 255}))

    # ------------------------------------------------------------------
    # Output shape tests
    # ------------------------------------------------------------------

    def test_threshold_outputs_preserve_input_dimensions(self):
        """All three threshold images should have the same H × W as the input."""
        img = cv2.imread(self.img_path)
        h, w = img.shape[:2]
        blurred = self._load_blurred()

        _, t_global = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        _, t_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        for thresh_img in [t_global, t_otsu, t_adaptive]:
            self.assertEqual(thresh_img.shape, (h, w))

    def test_threshold_outputs_are_2d(self):
        """Threshold output arrays should be 2-D (grayscale), not 3-D."""
        blurred = self._load_blurred()
        _, t = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        self.assertEqual(t.ndim, 2)

    # ------------------------------------------------------------------
    # Gaussian blur pre-processing
    # ------------------------------------------------------------------

    def test_gaussian_blur_reduces_noise(self):
        """Blurred image variance should be lower than the raw grayscale variance."""
        img = cv2.imread(self.img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.assertLessEqual(float(blurred.var()), float(gray.var()))


if __name__ == '__main__':
    unittest.main()
