"""服务器执行的条件结果对照小测试；不验证真实生理因果结论。"""
import inspect
import unittest

import numpy as np
import torch
from scipy.special import expit

from scripts.phase6_outcome_control import LinearOutcomeControl, fit_head, logistic_objective


class OutcomeControlTests(unittest.TestCase):
    def data(self) -> dict:
        rng = np.random.default_rng(42)
        ma = rng.normal(size=(128, 2))
        xc, xa = rng.normal(size=(128, 2)), np.ones((128, 1))
        y = (rng.random(128) < expit(1.5*ma[:, 0]-.7*ma[:, 1]+.2)).astype(np.int64)
        return dict(m_a=ma, x_c=xc, x_a=xa, y=y)

    def test_analytic_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(42)
        design, theta = rng.normal(size=(16, 4)), rng.normal(size=4)
        y = np.tile([0., 1.], 8)
        _, gradient = logistic_objective(theta, design, y)
        numerical = np.empty(4)
        for k in range(4):
            delta = np.eye(4)[k]*1e-6
            numerical[k] = (logistic_objective(theta+delta, design, y)[0]-
                            logistic_objective(theta-delta, design, y)[0])/2e-6
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-8)

    def test_fit_improves_training_nll_with_finite_two_class_probabilities(self):
        view = self.data()
        head, report = fit_head(view)
        self.assertTrue(report['converged'])
        prevalence = view['y'].mean()
        null = -prevalence*np.log(prevalence)-(1-prevalence)*np.log1p(-prevalence)
        self.assertLess(report['final_mean_nll'], null)
        with torch.no_grad():
            ma = torch.tensor(view['m_a'], dtype=torch.float32)
            ref = torch.tensor(np.concatenate((view['x_c'], view['x_a']), 1), dtype=torch.float32)
            p = head(ma, ref).softmax(-1)
        self.assertEqual(p.shape, (128, 2))
        self.assertTrue(torch.isfinite(p).all())
        torch.testing.assert_close(p.sum(-1), torch.ones(128))
        self.assertTrue((head.weight[-1] == 0).item())

    def test_training_view_rejects_hidden_or_test_fields(self):
        for forbidden in ('u', 'm_c', 'true_do', 'test_y'):
            view = self.data() | {forbidden: np.zeros(128)}
            with self.assertRaises(ValueError):
                fit_head(view)

    def test_head_interface_has_no_mc_and_rejects_misaligned_axes(self):
        self.assertEqual(list(inspect.signature(LinearOutcomeControl.forward).parameters),
                         ['self', 'm_a', 'reference'])
        head = LinearOutcomeControl(2, 3, np.ones(5), 0.)
        with self.assertRaises(ValueError):
            head(torch.ones(2, 4, 2), torch.ones(2, 3))


if __name__ == '__main__':
    unittest.main()
