"""方案A目标、梯度、基线兼容与分阶段守卫；待服务器CPU执行。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from classification_objectives import classification_terms, integrated_observational_nll
from frontdoor_scm import FrontDoorConfig, ReferenceBank, frontdoor_loss
from mainNeuroFDAN import get_args
from ModelTrain import loss_component_gradient_report, validate_resume_objective, run_fold
from NeuroFDAN import NeuroFDAN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scheme_a_fit_check import fit_gate


class SchemeATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(42)
        self.cfg = FrontDoorConfig(hidden_dim=8, lif_neurons=8, mediator_dim=4, ref_dim=8,
                                   S_train=2, K_train=2, S_eval=2, K_eval=2, dropout=0.)
        self.y = torch.tensor([0, 1])

    def outputs(self):
        obs = torch.randn(2, 2, requires_grad=True)
        combos = torch.randn(2, 2, 3, 2, requires_grad=True)
        p = combos.softmax(-1).mean(2).mean(1)
        kl = torch.rand(2, requires_grad=True)
        return dict(logits_obs=obs, probs_do=p, logits_do=p.log(), logits_combinations=combos,
                    kl_c=kl, kl_a=kl * .5, fire_loss=kl.mean() * .1)

    def test_nll_scores_probability_average_not_probability_as_logits(self):
        probs = torch.tensor([[.9, .1], [.2, .8]], requires_grad=True)
        out = dict(probs_do=probs, logits_do=probs.log())
        actual = integrated_observational_nll(out, self.y)
        expected = -out['probs_do'][torch.arange(2), self.y].log().mean()
        torch.testing.assert_close(actual, expected)
        wrong = F.cross_entropy(out['probs_do'], self.y)
        self.assertGreater(float((actual - wrong).detach().abs()), 1e-5)

    def test_replaces_factual_ce_and_reaches_cross_combinations(self):
        out = self.outputs()
        old = classification_terms(out, self.y, self.cfg, objective='factual')['loss']
        self.assertIsNone(torch.autograd.grad(old, out['logits_combinations'], allow_unused=True, retain_graph=True)[0])
        new = classification_terms(out, self.y, self.cfg, objective='integrated_observational', lambda_obs=0.)
        self.assertIsNone(torch.autograd.grad(new['loss'], out['logits_obs'], allow_unused=True, retain_graph=True)[0])
        g = torch.autograd.grad(new['loss'], out['logits_combinations'], retain_graph=True)[0]
        self.assertTrue((g.abs().sum(-1) > 0).all())
        self.assertEqual(float(new['weighted_obs_loss'].detach()), 0.)

    def test_default_factual_loss_and_gradient_are_bitwise_unchanged(self):
        for weight in (1., .3):
            out = self.outputs()
            old = frontdoor_loss(out, self.y, self.cfg, epoch=4)
            expected = old['loss'] + (weight - 1.) * old['loss_obs'] + .028 * out['fire_loss']
            new = classification_terms(out, self.y, self.cfg, epoch=4, lambda_obs=weight)
            actual = new['loss'] + .028 * out['fire_loss']
            self.assertTrue(torch.equal(actual, expected))
            tensors = (out['logits_obs'], out['kl_c'])
            ga = torch.autograd.grad(actual, tensors, retain_graph=True)
            gb = torch.autograd.grad(expected, tensors, retain_graph=True)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(ga, gb)))

    def test_rejects_none_mode_implicit_auxiliary_task_and_invalid_probabilities(self):
        out = self.outputs()
        with self.assertRaises(ValueError):
            classification_terms(out, self.y, self.cfg, objective='integrated_observational')
        with self.assertRaises(ValueError):
            integrated_observational_nll(dict(logits_obs=out['logits_obs']), self.y)
        bad = dict(out, probs_do=torch.full((2, 2), float('nan')))
        with self.assertRaises(FloatingPointError): integrated_observational_nll(bad, self.y)
        bad = dict(out, probs_do=torch.tensor([[0., 1.], [1., 0.]]),
                   logits_do=torch.tensor([[1e-38, 1.], [1., 1e-38]]).log())
        with self.assertRaises(FloatingPointError): integrated_observational_nll(bad, self.y)

    def test_real_model_backward_reaches_cns_ans_current_and_outcome(self):
        model = NeuroFDAN(ecg_channels=2, frontdoor_config=self.cfg)
        xc, xa = torch.randn(4, 32, 768), torch.randn(4, 3, 768)
        ids = torch.tensor([[1, i, 0] for i in range(4)])
        bank = ReferenceBank(xc[2:], xa[2:], ids[2:]); bank.raw_windows = True
        out = model(xc[:2], xa[:2], reference_bank=bank, source_ids=ids[:2])
        terms = classification_terms(out, self.y, self.cfg, objective='integrated_observational', lambda_obs=0.)
        total = terms['loss'] + .028 * out['fire_loss']
        report = loss_component_gradient_report(model, out, terms, total, 1, 0., .028, 'integrated_observational')
        self.assertEqual(report['reference_term'], 'integrated_observational_nll')
        task = report['gradients']['integrated_observational_nll']
        for key in ('cns', 'ans', 'cns2ans', 'outcome'):
            self.assertGreater(task[key]['l2'], 0.)
            self.assertIn('cosine_to_classification', task[key])
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        total.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_cli_baseline_and_integrated_guards(self):
        self.assertEqual(get_args(argv=['--label', 'arousal']).classification_objective, 'factual')
        args = get_args(argv=['--label', 'arousal', '--classification_objective', 'integrated_observational', '--lambda_obs', '0'])
        self.assertEqual(args.lambda_obs, 0.)
        for extra in ([], ['--lambda_obs', '0', '--fd_mode', 'none']):
            with self.assertRaises(SystemExit):
                get_args(argv=['--label', 'arousal', '--classification_objective', 'integrated_observational', *extra])

    def test_checkpoint_objective_change_requires_fresh_run(self):
        args = SimpleNamespace(lr=.00002, lambda_obs=0., fire_weight=.028,
                               factual_ce_mode='single', classification_objective='integrated_observational')
        state = dict(optimizer=dict(param_groups=[dict(lr=.00002)]),
                     arguments=dict(lambda_obs=0., fire_weight=.028))
        with self.assertRaisesRegex(ValueError, '主分类目标'):
            validate_resume_objective(state, args, 'DEAP')
        state['arguments']['classification_objective'] = 'integrated_observational'
        validate_resume_objective(state, args, 'DEAP')

    def test_fit_gate_requires_all_draws_nll_and_both_class_recalls(self):
        good = dict(balanced_accuracy=.9, min_class_recall=.8, integrated_observational_nll=.3)
        self.assertTrue(fit_gate([good] * 3, .68))
        for bad in (dict(good, balanced_accuracy=.5), dict(good, min_class_recall=0.),
                    dict(good, integrated_observational_nll=.67)):
            self.assertFalse(fit_gate([good, good, bad], .68))
        self.assertFalse(fit_gate([], .68))
        self.assertFalse(fit_gate([good], .68))

    def test_actual_training_entry_records_integrated_objective(self):
        from Datasets import PhysiologicalWindows
        def population(subject):
            data = object.__new__(PhysiologicalWindows)
            data.dataset, data.label, data.subjects = 'HCI', 'arousal', [subject]
            data.windows_per_trial, data.trials_per_subject = 1, 4
            data.x_c, data.x_a = torch.randn(4, 32, 768), torch.randn(4, 4, 768)
            data.Label = torch.tensor([0, 1, 0, 1])
            data.source_ids = torch.tensor([[subject, i, 0] for i in range(4)])
            return data
        with tempfile.TemporaryDirectory() as root:
            args = get_args(argv=['--dataset', 'HCI', '--label', 'arousal', '--device', 'cpu',
                '--output_dir', root, '--run_name', '方案A接口', '--held_out_subjects', '3', '--epochs', '1',
                '--batch_size', '2', '--hidden_dim', '8', '--lif_neurons', '8', '--mediator_dim', '4',
                '--ref_dim', '8', '--N_ref', '3', '--S_train', '2', '--K_train', '2', '--S_eval', '2', '--K_eval', '2',
                '--classification_objective', 'integrated_observational', '--lambda_obs', '0', '--fire_weight', '.028',
                '--loss_gradient_interval', '1'])
            run_fold(args, 3, population=population(1), test_dataset=population(3))
            fold = Path(root) / 'HCI/arousal/seed42/方案A接口/fold03'
            row = json.loads((fold / 'epoch_metrics.jsonl').read_text())
            self.assertEqual(row['optimization']['classification_objective'], 'integrated_observational')
            self.assertEqual(row['optimization']['weighted_obs_loss'], 0.)
            self.assertIn('integrated_observational_nll', row['train'])
            self.assertIn('macro_f1', row['test'])
            self.assertEqual(row['optimization']['loss_component_gradients']['reference_term'], 'integrated_observational_nll')
            state = torch.load(fold / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(state['arguments']['classification_objective'], 'integrated_observational')


if __name__ == '__main__':
    unittest.main()
