import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fixed_eval import make_split, fit_linear_probe, score_test_with_selected_probe
from vr_regularization import variance_floor_loss, shadow_should_stop


class BasicExtensionTests(unittest.TestCase):
    def test_split_is_complete_and_seeded(self):
        a = make_split(100, 123)
        b = make_split(100, 123)
        self.assertEqual((len(a["train"]), len(a["valid"]), len(a["test"])), (10, 10, 80))
        self.assertTrue(torch.equal(a["train"], b["train"]))
        self.assertEqual(len(torch.unique(torch.cat([a["train"], a["valid"], a["test"]]))), 100)

    def test_validation_only_probe_returns_finite_accuracy(self):
        torch.manual_seed(23)
        z = torch.randn(100, 8)
        y = (z[:, 0] > 0).long()
        split = make_split(100, 47)
        probe = fit_linear_probe(z, y, split, eval_epochs=8, eval_optimizer="Adam", eval_lr=0.01, eval_weight_decay=0, eval_seed=94)
        self.assertGreaterEqual(probe.val_accuracy, 0)
        self.assertLessEqual(probe.val_accuracy, 100)
        test_acc = score_test_with_selected_probe(z, y, split, probe.classifier_state)
        self.assertGreaterEqual(test_acc, 0)
        self.assertLessEqual(test_acc, 100)

    def test_variance_penalty(self):
        x = torch.ones((50, 4), requires_grad=True)
        penalty = variance_floor_loss(x, x, gamma=0.005)
        self.assertGreater(penalty.item(), 0)
        penalty.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        variable = torch.randn(50, 4)
        self.assertEqual(variance_floor_loss(variable, variable, gamma=0.005).item(), 0)

    def test_shadow(self):
        self.assertIsNone(shadow_should_stop({50: 6.2347}))
        self.assertTrue(shadow_should_stop({50: 6.2347, 75: 6.2347, 100: 6.2347}))
        self.assertFalse(shadow_should_stop({50: 5.14, 75: 5.01, 100: 4.81}))


if __name__ == "__main__":
    unittest.main()
