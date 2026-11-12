"""事实CE多采样估计器、默认兼容和梯度接口测试；待服务器执行。"""
import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator, ReferenceBank, frontdoor_loss
from ModelTrain import validate_resume_objective
from mainNeuroFDAN import get_args


class FactualMCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); torch.set_num_threads(1)
        self.cfg = FrontDoorConfig(hidden_dim=8, mediator_dim=4, lif_neurons=8,
                                   dropout=0., S_train=3, S_eval=3, K_train=2, K_eval=2)
        self.model = FrontDoorEstimator(5, 3, self.cfg)
        self.xc, self.xa = torch.randn(2, 5), torch.randn(2, 3)
        self.y = torch.tensor([1, 0])

    def test_mean_ce_not_ce_of_mean_probability_and_all_samples_receive_gradient(self):
        logits = torch.tensor([[[-1., 1.], [2., -2.], [-.5, .5]],
                               [[1., -1.], [-2., 2.], [.3, -.3]]], requires_grad=True)
        out = dict(logits_obs=logits[:, 0], logits_obs_samples=logits,
                   kl_c=torch.zeros(2), kl_a=torch.zeros(2))
        loss = frontdoor_loss(out, self.y, self.cfg)['loss_obs']
        expected = torch.stack([F.cross_entropy(logits[:, s], self.y) for s in range(3)]).mean()
        torch.testing.assert_close(loss, expected)
        mixture_ce = F.nll_loss(logits.softmax(-1).mean(1).log(), self.y)
        self.assertGreater(float(loss.detach() - mixture_ce.detach()), .1)
        loss.backward()
        self.assertTrue((logits.grad.abs().sum(-1) > 0).all())

    def test_single_default_preserves_original_random_call_path(self):
        torch.manual_seed(9); first = self.model(self.xc, self.xa, fd_mode='none')
        torch.manual_seed(9); second = self.model(self.xc, self.xa, fd_mode='none', factual_ce_mode='single')
        self.assertNotIn('logits_obs_samples', first)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_existing_s_and_head_are_reused_with_gradients(self):
        out = self.model(self.xc, self.xa, fd_mode='none', factual_ce_mode='all_samples')
        self.assertEqual(out['logits_obs_samples'].shape, (2, 3, 2))
        self.assertEqual(out['m_a'].shape, (2, 3, 4))
        torch.testing.assert_close(out['logits_obs'], out['logits_obs_samples'][:, 0])
        frontdoor_loss(out, self.y, self.cfg)['loss'].backward()
        for module in (self.model.cns, self.model.ans, self.model.outcome):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_single_sample_modes_agree(self):
        torch.manual_seed(8); a = self.model(self.xc, self.xa, fd_mode='none', S=1)
        torch.manual_seed(8); b = self.model(self.xc, self.xa, fd_mode='none', S=1, factual_ce_mode='all_samples')
        torch.testing.assert_close(frontdoor_loss(a, self.y, self.cfg)['loss'], frontdoor_loss(b, self.y, self.cfg)['loss'])

    def test_frontdoor_two_probability_means_unchanged(self):
        ids = torch.tensor([[1, 0, 0], [3, 0, 0]])
        bank = ReferenceBank(self.xc, self.xa, ids)
        out = self.model(self.xc, self.xa, bank, ids, factual_ce_mode='all_samples')
        torch.testing.assert_close(out['probs_do'], out['logits_combinations'].softmax(-1).mean(2).mean(1))

    def test_cli_and_resume_mode_guard(self):
        self.assertEqual(get_args(argv=['--label', 'arousal']).factual_ce_mode, 'single')
        args = SimpleNamespace(lr=.00002, lambda_obs=1., fire_weight=.028, factual_ce_mode='all_samples')
        state = dict(optimizer=dict(param_groups=[dict(lr=.00002)]), arguments=dict(lambda_obs=1., fire_weight=.028))
        with self.assertRaises(ValueError): validate_resume_objective(state, args, 'DEAP')
        state['arguments']['factual_ce_mode'] = 'all_samples'
        validate_resume_objective(state, args, 'DEAP')

    def test_integrated_model_cpu_forward_backward(self):
        from NeuroFDAN import NeuroFDAN
        model = NeuroFDAN(ecg_channels=2, frontdoor_config=self.cfg)
        out = model(torch.randn(2, 32, 768), torch.randn(2, 3, 768),
                    fd_mode='none', factual_ce_mode='all_samples')
        loss = frontdoor_loss(out, self.y, self.cfg)['loss'] + .028 * out['fire_loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
