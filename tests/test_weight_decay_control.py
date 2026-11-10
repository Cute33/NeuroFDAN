"""耦合L2对照入口与只读RMS的CPU测试；本轮未在云端执行。"""
import copy
from types import SimpleNamespace
import unittest

import torch

from mainNeuroFDAN import get_args
from ModelTrain import build_optimizer, optimizer_weight_decay, parameter_rms, validate_resume_weight_decay


class WeightDecayControlTests(unittest.TestCase):
    def test_cli_defaults_and_explicit_zero(self):
        self.assertEqual(get_args(argv=['--label', 'arousal']).weight_decay, .01)
        self.assertEqual(get_args(argv=['--label', 'arousal', '--weight_decay', '0']).weight_decay, 0.)
        for invalid in ('-1', 'nan', 'inf'):
            with self.subTest(value=invalid), self.assertRaises(SystemExit):
                get_args(argv=['--label', 'arousal', '--weight_decay=' + invalid])

    def test_old_namespace_keeps_original_decay_and_validation(self):
        self.assertEqual(optimizer_weight_decay(SimpleNamespace()), .01)
        with self.assertRaises(ValueError):
            optimizer_weight_decay(SimpleNamespace(weight_decay=float('nan')))

    def test_default_adam_step_matches_original_configuration(self):
        # 小型CPU线性层只验证默认更新兼容，不代表NeuroFDAN真实训练。
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 2)
        original = copy.deepcopy(model)
        new_optimizer = build_optimizer(model, SimpleNamespace(lr=.0001))
        old_optimizer = torch.optim.Adam(original.parameters(), lr=.0001, betas=(.9, .999),
                                         eps=1e-8, weight_decay=.01, amsgrad=False)
        for candidate, optimizer in ((model, new_optimizer), (original, old_optimizer)):
            candidate(torch.ones(2, 3)).square().mean().backward()
            optimizer.step()
        for actual, expected in zip(model.parameters(), original.parameters()):
            self.assertTrue(torch.equal(actual, expected))

    def test_zero_is_recorded_and_mismatched_resume_is_rejected(self):
        model = torch.nn.Linear(3, 2)
        optimizer = build_optimizer(model, SimpleNamespace(lr=.0001, weight_decay=0.))
        state = dict(optimizer=optimizer.state_dict())
        self.assertEqual(state['optimizer']['param_groups'][0]['weight_decay'], 0.)
        validate_resume_weight_decay(state, SimpleNamespace(weight_decay=0.))
        with self.assertRaises(ValueError):
            validate_resume_weight_decay(state, SimpleNamespace(weight_decay=.01))
        old_state = dict(optimizer=dict(param_groups=[dict(weight_decay=.01)]))
        validate_resume_weight_decay(old_state, SimpleNamespace())

    def test_rms_distinguishes_kernel_scales_without_mutation(self):
        model = torch.nn.Linear(3, 2)
        with torch.no_grad():
            model.weight.fill_(1e-12)
            model.bias.fill_(1.)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        rms = parameter_rms(model)
        self.assertAlmostEqual(rms['weight'] / 1e-12, 1., places=6)
        self.assertEqual(rms['bias'], 1.)
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with torch.no_grad():
            model.weight.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            parameter_rms(model)


if __name__ == '__main__':
    unittest.main()
