"""
Utility functions and constants for CIFAR-10 classification.
Extracted from 6.CNN_multiclass_CIFAR10.ipynb for reuse and testing.
"""
from collections import Counter

# CIFAR-10 class names in label-index order
CLASSES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]


def class_distribution(dataset, name: str) -> Counter:
    """Count samples per class and print a summary table.

    Supports three dataset types:

    * Objects with a ``samples`` attribute — list of ``(path, label)`` tuples
      (e.g. ``CIFAR10PJReddie``).
    * Objects with a ``subset`` attribute — ``TransformSubset`` wrapper whose
      ``subset.dataset.samples`` and ``subset.indices`` are accessible.
    * Objects with ``dataset`` and ``indices`` attributes — ``random_split``
      ``Subset`` style.

    Parameters
    ----------
    dataset:
        Dataset to analyse.
    name : str
        Human-readable label used in the printed header.

    Returns
    -------
    Counter
        Mapping from class index (int) to sample count (int).
    """
    if hasattr(dataset, 'samples'):
        # CIFAR10PJReddie
        all_labels = [label for _, label in dataset.samples]
    elif hasattr(dataset, 'subset'):
        # TransformSubset (validation)
        all_labels = [
            dataset.subset.dataset.samples[dataset.subset.indices[i]][1]
            for i in range(len(dataset))
        ]
    else:
        # random_split Subset — access via underlying dataset + indices
        all_labels = [dataset.dataset.samples[i][1] for i in dataset.indices]

    counts = Counter(all_labels)
    total = sum(counts.values())

    print(f'\n=== {name} ({total} samples) ===')
    print(f'  {"Class":<12} {"Count":>6}  {"Share":>7}')
    print('  ' + '-' * 28)
    for idx, cls in enumerate(CLASSES):
        n = counts.get(idx, 0)
        pct = 100 * n / total if total > 0 else 0
        print(f'  {cls:<12} {n:>6}  {pct:>6.2f} %')

    return counts
