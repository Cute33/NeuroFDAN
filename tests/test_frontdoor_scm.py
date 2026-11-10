"""第一阶段：形状、结构、梯度、经验参考与独立 oracle 积分测试。"""
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from frontdoor_scm import (FrontDoorConfig, FrontDoorEstimator, ReferenceBank,
                           ReferenceEncoder, OutcomeHead, frontdoor_loss, gaussian_kl)
from synthetic_scm import SyntheticSCM


def sources(n: int) -> torch.Tensor:
    return torch.tensor([[i // 4, i % 4, 0] for i in range(n)], dtype=torch.long)


class FrontDoorTests(unittest.TestCase):
    """随机输入只检查实现，不用分类准确率替代因果验证。"""
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.model = FrontDoorEstimator(5, 3, FrontDoorConfig(dropout=0.0))
        self.xc, self.xa = torch.randn(6, 5), torch.randn(6, 3)
        self.ids = sources(6)

    def test_hierarchical_shapes_and_clamp(self):
        output = self.model(self.xc, self.xa, source_ids=self.ids)
        for key in ('mu_c', 'logvar_c'):
            self.assertEqual(output[key].shape, (6, 64))
        for key in ('m_c', 'mu_a', 'logvar_a', 'm_a'):
            self.assertEqual(output[key].shape, (6, 4, 64))
        self.assertEqual(output['logits_combinations'].shape, (6, 4, 16, 2))
        self.assertTrue(torch.all(output['logvar_a'] >= -6))
        self.assertTrue(torch.all(output['logvar_a'] <= 2))

    def test_conditional_ans_and_gradient(self):
        mc = torch.randn(6, 4, 64, requires_grad=True)
        first = self.model.ans(self.xa, mc)['mu_a']
        changed = self.model.ans(self.xa, mc + 2)['mu_a']
        self.assertGreater((first - changed).abs().max().item(), 1e-3)
        gradient = torch.autograd.grad(first.square().sum(), mc)[0]
        self.assertGreater(gradient.abs().sum().item(), 0)
        cns = self.model.cns(self.xc, 4)
        objective = self.model.ans(self.xa, cns['m_c'])['mu_a'].sum()
        grad = torch.autograd.grad(objective, self.model.cns.parameters_head.weight)[0]
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_cns_current_reaches_lif_spikes(self):
        mc = torch.randn(6, 4, 64, requires_grad=True)
        captured = []
        hook = self.model.ans.snn.register_forward_hook(lambda module, args, output: captured.append(output[0]))
        self.model.ans(self.xa, mc)
        hook.remove()
        gradient = torch.autograd.grad(captured[0].sum(), mc)[0]
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_ans_no_cns_features_or_current_input_argument(self):
        self.assertEqual(list(inspect.signature(self.model.ans.forward).parameters), ['x_a', 'm_c'])
        mc = torch.randn(6, 4, 64)
        first = self.model.ans(self.xa, mc)['mu_a']
        self.model.cns(self.xc + 100, 4)
        second = self.model.ans(self.xa, mc)['mu_a']
        torch.testing.assert_close(first, second)

    def test_cns_ans_ablation_eliminates_path(self):
        model = FrontDoorEstimator(5, 3, FrontDoorConfig(no_cns2ans=True))
        mc = torch.randn(6, 4, 64, requires_grad=True)
        first = model.ans(self.xa, mc)['mu_a']
        second = model.ans(self.xa, mc + 100)['mu_a']
        torch.testing.assert_close(first, second)
        self.assertIsNone(torch.autograd.grad(first.sum(), mc, allow_unused=True)[0])
        self.assertIsNone(model.ans.cns_current)

    def test_outcome_inputs_and_nondegenerate_dependence(self):
        head = self.model.outcome
        self.assertEqual(list(inspect.signature(head.forward).parameters), ['m_a', 'reference'])
        ma, ref = torch.randn(6, 64, requires_grad=True), torch.randn(6, 8, requires_grad=True)
        gradients = torch.autograd.grad(head(ma, ref)[:, 1].sum(), (ma, ref))
        for gradient in gradients:
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_do_head_isolated_from_current_input(self):
        ma, ref = torch.randn(6, 4, 64), torch.randn(6, 7, 8)
        first = self.model.marginalize(ma, ref)[0]
        self.model.cns(self.xc + 100, 4)
        second = self.model.marginalize(ma, ref)[0]
        torch.testing.assert_close(first, second)

    def test_softmax_before_two_means(self):
        class ToyHead(nn.Module):
            def forward(self, ma, ref):
                z = ma[..., 0] + ref[..., 0]
                return torch.stack((torch.zeros_like(z), z), -1)
        self.model.outcome = ToyHead()
        ma = torch.zeros(1, 2, 64)
        ma[0, 1, 0] = 8
        ref = torch.zeros(1, 3, 8)
        ref[0, :, 0] = torch.tensor([-2., -1., 2.])
        probability, logits = self.model.marginalize(ma, ref)
        torch.testing.assert_close(probability, logits.softmax(-1).mean(2).mean(1))
        self.assertGreater((probability - logits.mean((1, 2)).softmax(-1)).abs().max().item(), .05)
        torch.testing.assert_close(probability.sum(-1), torch.ones(1))
        self.assertTrue(torch.all((probability >= 0) & (probability <= 1)))

    def test_none_never_samples_reference_bank(self):
        bank = ReferenceBank(self.xc, self.xa, self.ids)
        with patch.object(bank, 'sample', side_effect=AssertionError('不应执行参考积分')):
            output = self.model(self.xc, self.xa, reference_bank=bank, fd_mode='none')
        self.assertNotIn('probs_do', output)
        self.assertEqual(output['probs_obs'].shape, (6, 2))

    def test_bank_pairing_exclusion_and_provenance(self):
        xc = torch.arange(6).float()[:, None]
        bank = ReferenceBank(xc, xc + 100, self.ids)
        rc, ra, ids = bank.sample(6, 30, self.ids)
        torch.testing.assert_close(ra, rc + 100)
        self.assertFalse((ids == self.ids[:, None]).all(-1).any())
        recovered = ids[..., 0] * 4 + ids[..., 1]
        torch.testing.assert_close(rc[..., 0], recovered.float())
        _, _, restricted = bank.sample(3, 20, excluded_subjects=(0,))
        self.assertTrue((restricted[..., 0] != 0).all())
        self.assertFalse(list(bank.parameters()))

    def test_bank_heldout_duplicates_and_empty(self):
        with self.assertRaises(ValueError):
            ReferenceBank(self.xc, self.xa, self.ids, forbidden_subjects=(1,))
        with self.assertRaises(ValueError):
            ReferenceBank(self.xc, self.xa, self.ids * 0)
        bank = ReferenceBank(self.xc[:1], self.xa[:1], self.ids[:1])
        with self.assertRaises(ValueError):
            bank.sample(1, 1, self.ids[:1])
        short = ReferenceBank(self.xc, self.xa, self.ids, max_size=2048)
        self.assertTrue(short.shortfall)
        self.assertEqual(short.used_count, 6)

    def test_full_reference_shapes_and_no_pooling(self):
        for a, dim in [(3, 1120), (4, 1152)]:
            c, ans = torch.randn(2, 32, 32), torch.randn(2, a, 32)
            enc = ReferenceEncoder(1024, a * 32)
            self.assertEqual(enc(c, ans).shape, (2, dim))
            torch.testing.assert_close(enc(c, ans), torch.cat((c.flatten(1), ans.flatten(1)), 1))
            self.assertEqual(ReferenceEncoder(1024, a * 32, 64)(c, ans).shape, (2, 64))

    def test_complete_gnn_feature_interface(self):
        for a, reference_dim in [(3, 1120), (4, 1152)]:
            model = FrontDoorEstimator(1024, a * 32)
            output = model(torch.randn(2, 32, 32), torch.randn(2, a, 32),
                           source_ids=sources(2), S=2, K=3)
            self.assertEqual(output['probs_do'].shape, (2, 2))
            self.assertEqual(model.reference_encoder.output_dim, reference_dim)
            self.assertEqual(model.outcome.net[0].in_features, reference_dim + 64)

    def test_projection_gradient_and_optimizer_step_cpu(self):
        model = FrontDoorEstimator(5, 3, FrontDoorConfig(ref_dim=4))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.reference_encoder.projection.weight.detach().clone()
        output = model(self.xc, self.xa, source_ids=self.ids)
        loss = frontdoor_loss(output, torch.arange(6) % 2, model.config)['loss']
        loss.backward()
        self.assertGreater(model.reference_encoder.projection.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.reference_encoder.projection.weight))
        model.eval()
        bank = ReferenceBank(self.xc, self.xa, self.ids)
        with torch.no_grad():
            evaluated = model(self.xc, self.xa, bank, self.ids)
        self.assertEqual(evaluated['probs_do'].shape, (6, 2))

    def test_kl_axes_and_oracle_opt_in(self):
        mu, lv = torch.ones(6, 64), torch.zeros(6, 64)
        torch.testing.assert_close(gaussian_kl(mu, lv), torch.full((6,), 32.))
        output = self.model(self.xc, self.xa, source_ids=self.ids)
        torch.testing.assert_close(output['kl_a'], gaussian_kl(output['mu_a'], output['logvar_a']).mean(1))
        normal = frontdoor_loss(output, torch.arange(6) % 2, self.model.config)
        self.assertNotIn('loss_do_oracle', normal)
        oracle = frontdoor_loss(output, torch.arange(6) % 2, self.model.config, oracle_probs=torch.full((6, 2), .5))
        self.assertIn('loss_do_oracle', oracle)

    def test_invalid_shapes_and_singleton_training(self):
        with self.assertRaises(ValueError):
            self.model(self.xc[:1], self.xa[:1], source_ids=self.ids[:1])
        with self.assertRaises(ValueError):
            self.model.outcome(torch.randn(2, 64), torch.randn(1, 8))
        self.model.eval()
        with self.assertRaises(ValueError):
            self.model(self.xc, self.xa)

    def test_tanh_normal_likelihood_and_support(self):
        model = FrontDoorEstimator(5, 3, FrontDoorConfig(mediator_family='tanh_normal'))
        out = model.cns(self.xc, 3)
        self.assertTrue((out['m_c'].abs() < 1).all())
        value = torch.zeros(6, 64)
        expected = .5 * 1.8378770664093453 * 64
        self.assertAlmostEqual(model.cns.nll(value, value, value).item(), expected, places=4)
        with self.assertRaises(ValueError):
            frontdoor_loss({}, torch.zeros(6), model.config)

    def test_tanh_density_matches_transformed_distribution(self):
        model = FrontDoorEstimator(5, 3, FrontDoorConfig(mediator_family='tanh_normal'))
        mu, lv = torch.randn(6, 64) * .2, torch.randn(6, 64) * .2
        value = torch.randn(6, 64).tanh()
        distribution = torch.distributions.TransformedDistribution(
            torch.distributions.Normal(mu, (.5 * lv).exp()),
            [torch.distributions.TanhTransform()])
        expected = -distribution.log_prob(value).sum(-1).mean()
        torch.testing.assert_close(model.cns.nll(value, mu, lv), expected)

    @unittest.skipUnless(torch.cuda.is_available(), '当前环境无可用 CUDA，未执行 GPU 测试')
    def test_cuda_forward_backward(self):
        model = self.model.cuda()
        output = model(self.xc.cuda(), self.xa.cuda(), source_ids=self.ids.cuda())
        frontdoor_loss(output, (torch.arange(6) % 2).cuda(), model.config)['loss'].backward()


class OracleTests(unittest.TestCase):
    """数值容差依据独立随机误差；不要求单条随机曲线严格单调。"""
    def test_gamma_oracle_direct_agreement(self):
        scm = SyntheticSCM()
        point = scm.sample(3, 0, 11)
        for gamma in (0., .5, 1., 2.):
            direct, dse = scm.direct_intervention(point['x_c'], point['x_a'], gamma, 32768, 100)
            oracle, ose = scm.frontdoor_integral(point['x_c'], point['x_a'], gamma, 2048, 1024, 200)
            self.assertTrue(np.all(abs(direct - oracle) < 5 * np.sqrt(dse ** 2 + ose ** 2) + .01))

    def test_oracle_independent_of_direct_function(self):
        scm = SyntheticSCM()
        point = scm.sample(1, 1, 1)
        with patch.object(scm, 'direct_intervention', side_effect=AssertionError('禁止复用直接干预真值')):
            probability, _ = scm.frontdoor_integral(point['x_c'], point['x_a'], 1, 64, 32, 9)
        self.assertTrue(np.all((probability > 0) & (probability < 1)))

    def test_invalid_point_shapes_are_rejected(self):
        scm = SyntheticSCM()
        with self.assertRaises(ValueError):
            scm.direct_intervention(np.zeros((2, 16)), np.zeros((1, 8)), 1., 10, 0)
        with self.assertRaises(ValueError):
            scm.frontdoor_integral(np.zeros((1, 16)), np.zeros((1, 8)), 1., 0, 10, 0)

    def test_convergence_in_repeated_runs(self):
        scm = SyntheticSCM()
        point = scm.sample(2, 1, 12)
        truth, _ = scm.direct_intervention(point['x_c'], point['x_a'], 1, 65536, 111)
        errors = []
        for size in (32, 2048):
            errors.append(np.mean([(scm.frontdoor_integral(point['x_c'], point['x_a'], 1, size, size, seed)[0] - truth) ** 2
                                   for seed in range(4)]))
        self.assertLess(errors[1], errors[0])

    def test_outcome_conditional_independence_moments(self):
        scm = SyntheticSCM()
        data = scm.sample(8192, 1, 337)
        conditional = scm.conditional_outcome(data['m_a'], data['x_c'], data['x_a'], 1)
        residual = data['y'] - conditional
        functions = data['m_c'][:, :8]
        scores = residual[:, None] * functions
        z = scores.mean(0) / (scores.std(0, ddof=1) / np.sqrt(len(residual)))
        self.assertLess(np.abs(z).max(), 4.5)
        self.assertGreater(np.linalg.norm(scm.C_ca), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
