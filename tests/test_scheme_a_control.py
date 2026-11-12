"""服务器端KL单因素守卫、私有权重保存与零KL系数的CPU小测试。"""
import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scheme_a_control_evidence import BASELINE_COMMIT, check_arguments, check_sources, check_initial_nll, compare
from scheme_a_fit_check import save_private_checkpoint


def fixture():
    args = dict(dataset='DEAP', label='arousal', held_out_subjects=[2], seed=42,
        classification_objective='integrated_observational', beta=.001, fire_weight=.028,
        root_dir='/example', output_dir='/old-output', run_name='baseline', lr=.00002, batch_size=400)
    draw = dict(balanced_accuracy=.5, macro_f1=.38, integrated_observational_nll=.665,
                confusion_matrix=[[0, 153], [0, 247]], probability_std=[.001, .001], posterior={})
    record = dict(step=1, optimization_nll=.6838082075, draws=[draw] * 3)
    base = dict(source_version={'commit': BASELINE_COMMIT}, arguments=args, completed=True,
        passed=False, actual_steps=300, fit_controls={'fit_max_steps': 300, 'fit_check_every': 50},
        selected_source_ids=[[1, 1, 1]], reference_source_ids=[[3, 1, 1]],
        true_counts=[153, 247], constant_prior_nll=.6652747443, full_epoch_steps=31,
        gate={'balanced_accuracy_min': .8}, model={'beta': .001, 'S_train': 4},
        reference_report={'held_out_subject': 2}, records=[record, dict(record, step=300)])
    new = copy.deepcopy(base)
    new['arguments'].update(beta=0., output_dir='/new-output', run_name='kl-off')
    new['model']['beta'] = 0.
    return base, new


class SchemeAControlTests(unittest.TestCase):
    def test_only_beta_and_output_location_may_change(self):
        base, new = fixture()
        check_arguments(base, new['arguments'], new['fit_controls'])
        for key, value in (('lr', .0001), ('fire_weight', 0.), ('root_dir', '/other-data')):
            bad = dict(new['arguments'], **{key: value})
            with self.assertRaises(ValueError):
                check_arguments(base, bad, new['fit_controls'])

    def test_wrong_baseline_or_budget_is_rejected(self):
        base, new = fixture()
        for key, value in (('actual_steps', 37), ('passed', True), ('completed', False)):
            bad = dict(base, **{key: value})
            with self.assertRaises(ValueError):
                check_arguments(bad, new['arguments'], new['fit_controls'])
        with self.assertRaises(ValueError):
            check_arguments(base, new['arguments'], dict(fit_max_steps=500, fit_check_every=50))

    def test_changed_sample_reference_or_gate_is_rejected(self):
        base, new = fixture()
        check_sources(base, new)
        for key in ('selected_source_ids', 'reference_source_ids', 'gate'):
            bad = dict(new, **{key: []})
            with self.assertRaises(ValueError):
                check_sources(base, bad)

    def test_initial_task_nll_checks_reproduction_not_loss_weight(self):
        base, _ = fixture()
        check_initial_nll(base, base['records'][0]['optimization_nll'])
        for value in (float('nan'), .70):
            with self.assertRaises(ValueError):
                check_initial_nll(base, value)

    def test_valid_negative_result_remains_a_negative_result(self):
        base, new = fixture()
        before = copy.deepcopy(new)
        report = compare(base, new)
        self.assertTrue(report['protocol_checks_passed'])
        self.assertFalse(report['kl_off']['passed'])
        self.assertFalse(report['automatic_fold_training'])
        self.assertEqual(new, before)

    def test_private_checkpoint_is_atomic_exclusive_and_not_a_formal_resume(self):
        _, result = fixture()
        model = torch.nn.Linear(2, 2)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / 'private_fit_last.pt'
            save_private_checkpoint(model, result, Path(root))
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError):
                save_private_checkpoint(model, result, Path(root))
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), digest)
            self.assertEqual(list(Path(root).glob('.private-fit-*')), [])
            payload = torch.load(target, map_location='cpu', weights_only=False)
            self.assertEqual(payload['format_version'], 'neurofdan_private_fit_v1')
            self.assertNotIn('optimizer', payload)
            self.assertNotIn('epoch', payload)
            self.assertTrue(all(torch.equal(state[k], v) for k, v in model.state_dict().items()))

    def test_beta_zero_removes_only_kl_gradient_and_retains_task_and_fire(self):
        from classification_objectives import classification_terms
        from frontdoor_scm import FrontDoorConfig
        p = torch.tensor([[.4, .6], [.6, .4]], requires_grad=True)
        kl = torch.tensor([1., 2.], requires_grad=True)
        fire = torch.tensor(.5, requires_grad=True)
        outputs = dict(probs_do=p, logits_do=p.log(), logits_obs=torch.zeros(2, 2),
                       kl_c=kl, kl_a=kl, fire_loss=fire)
        terms = classification_terms(outputs, torch.tensor([1, 0]), FrontDoorConfig(beta=0.),
                                     objective='integrated_observational', lambda_obs=0.)
        total = terms['loss'] + .028 * fire
        gp, gkl, gf = torch.autograd.grad(total, (p, kl, fire))
        self.assertGreater(float(gp.abs().sum()), 0.)
        self.assertEqual(float(gkl.abs().sum()), 0.)
        self.assertAlmostEqual(float(gf), .028, places=6)


if __name__ == '__main__':
    unittest.main()
