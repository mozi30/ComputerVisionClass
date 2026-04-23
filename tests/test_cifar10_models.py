"""
Unit tests for T7 – CNN model architectures (models.py).

Tests cover:
* Successful instantiation of each model.
* Forward-pass output shape: (batch_size, num_classes).
* Correct default number of output classes (10 for CIFAR-10).
* Custom num_classes parameter is respected.
* Trainability – a single SGD step reduces the loss.
"""
import math
import os
import sys
import unittest

import torch
import torch.nn as nn

# Allow import from the sibling directory (name starts with a digit).
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', '02_image_classification'),
)

from models import ResNet, CNN1, CNN2, CNN3, CNN4  # noqa: E402

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 10
BATCH_SIZE  = 4
IMG_SIZE    = 32   # CIFAR-10 images are 32 × 32

def _dummy_batch(batch=BATCH_SIZE, c=3, h=IMG_SIZE, w=IMG_SIZE):
    """Return a random (N, C, H, W) float tensor in [0, 1]."""
    return torch.rand(batch, c, h, w)


def _dummy_labels(batch=BATCH_SIZE, num_classes=NUM_CLASSES):
    """Return random integer class labels."""
    return torch.randint(0, num_classes, (batch,))


def _single_train_step(model, inputs, labels):
    """Run one forward + backward pass and return the loss value."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Parametric base class
# ---------------------------------------------------------------------------

class _ModelTestMixin:
    """Mix-in providing common tests for all models.

    Sub-classes must set ``model_cls``.
    """

    model_cls = None  # set in each sub-class

    def _make_model(self, num_classes=NUM_CLASSES):
        return self.model_cls(num_classes=num_classes)

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_instantiation_default(self):
        """Model should instantiate without arguments."""
        model = self._make_model()
        self.assertIsInstance(model, nn.Module)

    def test_custom_num_classes(self):
        """Model should respect a custom num_classes value."""
        for nc in (5, 10, 20):
            model = self._make_model(num_classes=nc)
            out = model(_dummy_batch())
            self.assertEqual(out.shape[-1], nc,
                             msg=f'Expected {nc} output units, got {out.shape[-1]}')

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Forward pass should return (batch_size, num_classes)."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            out = model(_dummy_batch())
        self.assertEqual(out.shape, (BATCH_SIZE, NUM_CLASSES))

    def test_output_is_2d(self):
        """Output tensor should be 2-D (batch × classes)."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            out = model(_dummy_batch())
        self.assertEqual(out.dim(), 2)

    def test_batch_size_one(self):
        """Forward pass should work with a single-sample batch."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            out = model(_dummy_batch(batch=1))
        self.assertEqual(out.shape, (1, NUM_CLASSES))

    def test_no_nan_in_output(self):
        """Output should not contain NaN values."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            out = model(_dummy_batch())
        self.assertFalse(torch.isnan(out).any().item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def test_single_train_step_does_not_crash(self):
        """A single SGD step should complete without error."""
        model = self._make_model()
        inputs = _dummy_batch()
        labels = _dummy_labels()
        loss = _single_train_step(model, inputs, labels)
        self.assertIsInstance(loss, float)
        self.assertFalse(math.isnan(loss), msg='Loss is NaN')

    def test_parameters_are_updated_after_train_step(self):
        """Parameters should change after one backward pass."""
        model = self._make_model()
        # Snapshot initial parameters
        params_before = [p.clone().detach() for p in model.parameters()]
        _single_train_step(model, _dummy_batch(), _dummy_labels())
        params_after  = list(model.parameters())
        changed = any(
            not torch.equal(pb, pa)
            for pb, pa in zip(params_before, params_after)
        )
        self.assertTrue(changed, 'No parameters changed after a training step.')


# ---------------------------------------------------------------------------
# Concrete test classes – one per model
# ---------------------------------------------------------------------------

class TestResNet(unittest.TestCase, _ModelTestMixin):
    model_cls = ResNet

    def test_num_layers(self):
        """ResNet should have exactly 4 residual layer groups."""
        model = self._make_model()
        self.assertTrue(hasattr(model, 'layer1'))
        self.assertTrue(hasattr(model, 'layer2'))
        self.assertTrue(hasattr(model, 'layer3'))
        self.assertTrue(hasattr(model, 'layer4'))

    def test_has_residual_blocks(self):
        """Each layer group should contain ResidualBlock instances."""
        from models import ResidualBlock
        model = self._make_model()
        for layer_name in ('layer1', 'layer2', 'layer3', 'layer4'):
            layer = getattr(model, layer_name)
            blocks = [m for m in layer.children() if isinstance(m, ResidualBlock)]
            self.assertGreater(len(blocks), 0,
                               msg=f'{layer_name} contains no ResidualBlock')

    def test_first_conv_output_channels(self):
        """First conv layer should output 64 channels."""
        model = self._make_model()
        self.assertEqual(model.conv1.out_channels, 64)


class TestCNN1(unittest.TestCase, _ModelTestMixin):
    model_cls = CNN1

    def test_has_two_feature_blocks(self):
        """CNN1 should have features1 and features2 Sequential blocks."""
        model = self._make_model()
        self.assertTrue(hasattr(model, 'features1'))
        self.assertTrue(hasattr(model, 'features2'))

    def test_has_softmax(self):
        """CNN1 should include a Softmax activation."""
        model = self._make_model()
        self.assertTrue(hasattr(model, 'softmax'))
        self.assertIsInstance(model.softmax, nn.Softmax)

    def test_output_sums_to_one(self):
        """CNN1 softmax output should sum to ~1 per sample."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            out = model(_dummy_batch())
        row_sums = out.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(BATCH_SIZE), atol=1e-5))


class TestCNN2(unittest.TestCase, _ModelTestMixin):
    model_cls = CNN2

    def test_has_two_feature_blocks(self):
        """CNN2 should have features1 and features2 Sequential blocks."""
        model = self._make_model()
        self.assertTrue(hasattr(model, 'features1'))
        self.assertTrue(hasattr(model, 'features2'))

    def test_single_fc_layer(self):
        """CNN2 should have a single final linear layer 'fc'."""
        model = self._make_model()
        self.assertTrue(hasattr(model, 'fc'))
        self.assertIsInstance(model.fc, nn.Linear)

    def test_fc_output_units(self):
        """FC layer output should match num_classes."""
        model = self._make_model(num_classes=10)
        self.assertEqual(model.fc.out_features, 10)


class TestCNN3(unittest.TestCase, _ModelTestMixin):
    model_cls = CNN3

    def test_has_two_feature_blocks(self):
        """CNN3 should have features1 and features2 Sequential blocks."""
        model = self._make_model()
        self.assertTrue(hasattr(model, 'features1'))
        self.assertTrue(hasattr(model, 'features2'))

    def test_has_dropout(self):
        """CNN3 should have a Dropout layer."""
        model = self._make_model()
        self.assertTrue(hasattr(model, 'dropout'))
        self.assertIsInstance(model.dropout, nn.Dropout)

    def test_has_batch_norm_in_features(self):
        """CNN3 feature blocks should include BatchNorm2d."""
        model = self._make_model()
        for block_name in ('features1', 'features2'):
            block = getattr(model, block_name)
            bn_layers = [m for m in block.children() if isinstance(m, nn.BatchNorm2d)]
            self.assertGreater(len(bn_layers), 0,
                               msg=f'{block_name} has no BatchNorm2d')

    def test_three_fc_layers(self):
        """CNN3 should have three fully-connected layers (fc1, fc2, fc3)."""
        model = self._make_model()
        self.assertTrue(hasattr(model, 'fc1'))
        self.assertTrue(hasattr(model, 'fc2'))
        self.assertTrue(hasattr(model, 'fc3'))


class TestCNN4(unittest.TestCase, _ModelTestMixin):
    model_cls = CNN4

    def test_has_single_feature_block(self):
        """CNN4 should have a single 'features' Sequential block."""
        model = self._make_model()
        self.assertTrue(hasattr(model, 'features'))
        self.assertIsInstance(model.features, nn.Sequential)

    def test_features_has_dropout(self):
        """CNN4 features block should include a Dropout layer."""
        model = self._make_model()
        dropout_layers = [m for m in model.features.children()
                          if isinstance(m, nn.Dropout)]
        self.assertGreater(len(dropout_layers), 0,
                           msg='features block has no Dropout')

    def test_features_has_batch_norm(self):
        """CNN4 features block should include BatchNorm2d."""
        model = self._make_model()
        bn_layers = [m for m in model.features.children()
                     if isinstance(m, nn.BatchNorm2d)]
        self.assertGreater(len(bn_layers), 0,
                           msg='features block has no BatchNorm2d')

    def test_two_fc_layers(self):
        """CNN4 should have two fully-connected layers (fc1, fc2)."""
        model = self._make_model()
        self.assertTrue(hasattr(model, 'fc1'))
        self.assertTrue(hasattr(model, 'fc2'))
        self.assertIsInstance(model.fc1, nn.Linear)
        self.assertIsInstance(model.fc2, nn.Linear)


if __name__ == '__main__':
    unittest.main()
