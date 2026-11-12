"""分项记录不改变更新、加权项一致及CLI开关测试，待服务器运行。"""
import copy
import unittest

import torch

from frontdoor_scm import FrontDoorConfig, frontdoor_loss
from ModelTrain import loss_component_gradient_report
from NeuroFDAN import NeuroFDAN
from mainNeuroFDAN import get_args


class FireGradientControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cfg = FrontDoorConfig(hidden_dim=8, lif_neurons=8, mediator_dim=4,
                              S_train=2, time_steps=2, dropout=0.)
        self.model = NeuroFDAN(ecg_channels=2, frontdoor_config=cfg)
        self.xc, self.xa = torch.randn(2, 32, 768), torch.randn(2, 3, 768)
        self.y = torch.tensor([0, 1])

    def step(self, model, diagnostic):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.00002)
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(42)
        out = model(self.xc, self.xa, fd_mode='none')
        terms = frontdoor_loss(out, self.y, model.frontdoor_config, epoch=5)
        loss = terms['loss'] + .028 * out['fire_loss']
        rng = torch.get_rng_state().clone()
        if diagnostic:
            report = loss_component_gradient_report(model, out, terms, loss, 5, 1., .028)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertAlmostEqual(sum(v for k, v in report['weighted_terms'].items() if k != 'training_total'), float(loss.detach()), places=6)
            self.assertEqual(report['weights'], dict(lambda_obs=1., beta=.0005, fire=.028))
        loss.backward()
        optimizer.step()

    def test_diagnostic_preserves_rng_gradients_and_adam_update(self):
        baseline, recorded = copy.deepcopy(self.model), copy.deepcopy(self.model)
        self.step(baseline, False)
        self.step(recorded, True)
        for name, tensor in baseline.state_dict().items():
            torch.testing.assert_close(tensor, recorded.state_dict()[name], rtol=0, atol=0)
        for a, b in zip(baseline.parameters(), recorded.parameters()):
            if a.grad is None:
                self.assertIsNone(b.grad)
            else:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_default_objective_is_unchanged_and_flag_is_opt_in(self):
        args = get_args(argv=['--label', 'arousal'])
        self.assertEqual((args.fire_weight, args.loss_gradient_interval), (2.8, 0))
        args = get_args(argv=['--label', 'arousal', '--fire_weight', '.028', '--loss_gradient_interval', '10'])
        self.assertEqual((args.fire_weight, args.loss_gradient_interval), (.028, 10))
        with self.assertRaises(SystemExit):
            get_args(argv=['--label', 'arousal', '--loss_gradient_interval', '-1'])


if __name__ == '__main__':
    unittest.main()
