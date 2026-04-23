"""
Unit tests for T6 – class-balance verification (PR #2, last task).

Tests the ``class_distribution`` function and the ``CLASSES`` constant
defined in ``02_image_classification/cifar10_utils.py``.
"""
import sys
import os
import unittest
from collections import Counter

# Allow direct import from the sibling directory (name starts with a digit,
# so standard package import is not possible).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_image_classification'))

from cifar10_utils import CLASSES, class_distribution  # noqa: E402


# ---------------------------------------------------------------------------
# Lightweight dataset mocks – no torch / filesystem required
# ---------------------------------------------------------------------------

class _SamplesDataset:
    """Mimics CIFAR10PJReddie: exposes a ``samples`` list of (path, label)."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)


class _SubsetDataset:
    """Mimics a ``random_split`` Subset: exposes ``.dataset`` and ``.indices``."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)


class _TransformSubset:
    """Mimics TransformSubset: exposes ``.subset`` (a _SubsetDataset)."""

    def __init__(self, dataset, indices):
        self.subset = _SubsetDataset(dataset, indices)

    def __len__(self):
        return len(self.subset.indices)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _balanced_samples():
    """Return 10 samples, one per class (perfectly balanced)."""
    return [(f'img_{i}.png', i) for i in range(10)]


def _unbalanced_samples():
    """Return samples where class 0 has 5 items, classes 1-9 have 1 each."""
    samples = [(f'img_0_{i}.png', 0) for i in range(5)]
    samples += [(f'img_{c}.png', c) for c in range(1, 10)]
    return samples


# ---------------------------------------------------------------------------
# Tests for the CLASSES constant
# ---------------------------------------------------------------------------

class TestClasses(unittest.TestCase):

    def test_has_ten_entries(self):
        self.assertEqual(len(CLASSES), 10)

    def test_contains_all_cifar10_names(self):
        expected = {
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck',
        }
        self.assertEqual(set(CLASSES), expected)

    def test_first_class_is_airplane(self):
        self.assertEqual(CLASSES[0], 'airplane')

    def test_last_class_is_truck(self):
        self.assertEqual(CLASSES[-1], 'truck')


# ---------------------------------------------------------------------------
# Tests for class_distribution – branch 1: dataset with .samples
# ---------------------------------------------------------------------------

class TestClassDistributionSamplesDataset(unittest.TestCase):

    def test_returns_counter(self):
        ds = _SamplesDataset(_balanced_samples())
        result = class_distribution(ds, 'test')
        self.assertIsInstance(result, Counter)

    def test_balanced_counts_are_one_each(self):
        ds = _SamplesDataset(_balanced_samples())
        counts = class_distribution(ds, 'test')
        for cls_idx in range(10):
            self.assertEqual(counts[cls_idx], 1)

    def test_total_equals_dataset_size(self):
        ds = _SamplesDataset(_balanced_samples())
        counts = class_distribution(ds, 'test')
        self.assertEqual(sum(counts.values()), len(ds))

    def test_unbalanced_counts(self):
        ds = _SamplesDataset(_unbalanced_samples())
        counts = class_distribution(ds, 'test')
        self.assertEqual(counts[0], 5)
        for cls_idx in range(1, 10):
            self.assertEqual(counts[cls_idx], 1)

    def test_missing_class_not_in_counts(self):
        # Dataset only contains class 0 – all other classes are absent.
        ds = _SamplesDataset([('img.png', 0)] * 5)
        counts = class_distribution(ds, 'test')
        self.assertEqual(counts.get(1, 0), 0)
        self.assertEqual(counts.get(9, 0), 0)

    def test_single_sample(self):
        ds = _SamplesDataset([('img.png', 3)])
        counts = class_distribution(ds, 'test')
        self.assertEqual(counts[3], 1)
        self.assertEqual(sum(counts.values()), 1)


# ---------------------------------------------------------------------------
# Tests for class_distribution – branch 2: TransformSubset (.subset)
# ---------------------------------------------------------------------------

class TestClassDistributionTransformSubset(unittest.TestCase):

    def test_full_subset_total(self):
        base = _SamplesDataset(_balanced_samples())
        ds = _TransformSubset(base, list(range(10)))
        counts = class_distribution(ds, 'val')
        self.assertEqual(sum(counts.values()), 10)

    def test_partial_indices_total(self):
        base = _SamplesDataset(_balanced_samples())
        ds = _TransformSubset(base, list(range(5)))
        counts = class_distribution(ds, 'val')
        self.assertEqual(sum(counts.values()), 5)

    def test_partial_indices_correct_classes(self):
        base = _SamplesDataset(_balanced_samples())  # label i == index i
        ds = _TransformSubset(base, [0, 2, 4])       # classes 0, 2, 4
        counts = class_distribution(ds, 'val')
        self.assertEqual(counts[0], 1)
        self.assertEqual(counts[2], 1)
        self.assertEqual(counts[4], 1)
        self.assertEqual(sum(counts.values()), 3)


# ---------------------------------------------------------------------------
# Tests for class_distribution – branch 3: random_split Subset (.dataset)
# ---------------------------------------------------------------------------

class TestClassDistributionRandomSplitSubset(unittest.TestCase):

    def test_full_subset_total(self):
        base = _SamplesDataset(_balanced_samples())
        ds = _SubsetDataset(base, list(range(10)))
        counts = class_distribution(ds, 'train')
        self.assertEqual(sum(counts.values()), 10)

    def test_partial_indices_total(self):
        base = _SamplesDataset(_balanced_samples())
        ds = _SubsetDataset(base, list(range(5)))
        counts = class_distribution(ds, 'train')
        self.assertEqual(sum(counts.values()), 5)

    def test_partial_indices_correct_classes(self):
        base = _SamplesDataset(_balanced_samples())
        ds = _SubsetDataset(base, [1, 3, 7])  # classes 1, 3, 7
        counts = class_distribution(ds, 'train')
        self.assertEqual(counts[1], 1)
        self.assertEqual(counts[3], 1)
        self.assertEqual(counts[7], 1)
        self.assertEqual(sum(counts.values()), 3)

    def test_returns_counter(self):
        base = _SamplesDataset(_balanced_samples())
        ds = _SubsetDataset(base, list(range(10)))
        result = class_distribution(ds, 'train')
        self.assertIsInstance(result, Counter)


if __name__ == '__main__':
    unittest.main()
