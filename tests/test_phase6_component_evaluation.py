"""服务器端执行的四格评价数学与只读接口测试；不是训练实验。"""
import unittest

import numpy as np
import torch

from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator
from scripts.evaluate_phase6_components import paired_mediators, state_digest, true_conditional_grid
from synthetic_scm import SCMConfig, SyntheticSCM


class ComponentEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.scm = SyntheticSCM(SCMConfig(d_u=2, d_xc=3, d_xa=2, d_mc=4, d_ma=4))
        self.model = FrontDoorEstimator(3, 2, FrontDoorConfig(mediator_dim=4, hidden_dim=8,
            time_steps=2, mediator_family='tanh_normal')).eval()
        self.rng = np.random.default_rng(42)

    def test_known_cross_grid_equals_existing_pointwise_conditional(self):
        ma = np.tanh(self.rng.normal(size=(2, 3, 4)))
        rc = self.rng.normal(size=(2, 5, 3))
        ra = self.rng.normal(size=(2, 5, 2))
        grid = true_conditional_grid(self.scm, ma, rc, ra, 1.)
        self.assertEqual(grid.shape, (2, 3, 5))
        for b in range(2):
            for s in range(3):
                expected = self.scm.conditional_outcome(np.repeat(ma[b, s][None], 5, axis=0), rc[b], ra[b], 1.)
                np.testing.assert_allclose(grid[b, s], expected, rtol=1e-12, atol=1e-12)

    def test_zero_confounding_removes_known_reference_dependence(self):
        ma = np.tanh(self.rng.normal(size=(2, 3, 4)))
        rc, ra = self.rng.normal(size=(2, 5, 3)), self.rng.normal(size=(2, 5, 2))
        first = true_conditional_grid(self.scm, ma, rc, ra, 0.)
        second = true_conditional_grid(self.scm, ma, rc*3+2, ra-5, 0.)
        np.testing.assert_array_equal(first, second)

    def test_pairing_is_reproducible_and_model_state_unchanged(self):
        xc, xa = torch.randn(2, 3), torch.randn(2, 2)
        before = state_digest(self.model)
        with torch.no_grad():
            first = paired_mediators(self.model, self.scm, xc, xa, 7, 42)
            second = paired_mediators(self.model, self.scm, xc, xa, 7, 42)
        for one, two in zip(first, second):
            self.assertEqual(one.shape, (2, 7, 4))
            np.testing.assert_array_equal(one, two)
            self.assertTrue(np.isfinite(one).all())
            self.assertTrue((abs(one) <= 1).all())
        self.assertEqual(before, state_digest(self.model))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_unbounded_coordinate_substitution_is_rejected(self):
        self.model.config.mediator_family = 'normal'
        with self.assertRaises(ValueError), torch.no_grad():
            paired_mediators(self.model, self.scm, torch.zeros(2, 3), torch.zeros(2, 2), 3, 42)
        with self.assertRaises(ValueError):
            true_conditional_grid(self.scm, np.full((2, 3, 4), 2.),
                np.zeros((2, 5, 3)), np.zeros((2, 5, 2)), 1.)


if __name__ == '__main__':
    unittest.main()
