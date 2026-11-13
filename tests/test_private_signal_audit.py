"""只读诊断的CPU统计/来源守卫测试，云端不执行。"""
import copy
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_private_signal import stats, variance_parts, validate


class SignalAuditTests(unittest.TestCase):
    def test_constant_mean_variable_noise_is_distinguished(self):
        x = torch.tensor([[[-1., 1.], [-1., 1.]], [[1., -1.], [1., -1.]]])
        d = variance_parts(x)
        self.assertEqual(d['finite_R_between_sample_mean_variance'], [0., 0.])
        self.assertEqual(d['mean_within_sample_MC_variance'], [1., 1.])

    def test_input_variation_and_variance_identity(self):
        x = torch.arange(24.).reshape(3, 4, 2)
        d = variance_parts(x)
        torch.testing.assert_close(torch.tensor(d['total_variance']),
            torch.tensor(d['finite_R_between_sample_mean_variance']) + torch.tensor(d['mean_within_sample_MC_variance']))

    def test_dimension_and_sample_variation_do_not_mix_and_no_mutation(self):
        x = torch.tensor([[0., 2.], [0., 2.]])
        before = x.clone()
        d = stats(x)
        self.assertEqual(d['between_sample_std_mean'], 0.)
        self.assertGreater(d['within_sample_dimension_std_mean'], 0.)
        self.assertTrue(torch.equal(x, before))
        with self.assertRaises(ValueError): stats(torch.full((2, 2), float('nan')))

    def test_checkpoint_identity_and_heldout_guard(self):
        p = dict(format_version='neurofdan_private_fit_v1', actual_steps=300,
            arguments=dict(dataset='DEAP', label='arousal', held_out_subjects=[2], seed=42, beta=0.),
            source_version=dict(commit='bfe108e703b5d089fc53d59d254b61d6fcbbb885'),
            reference_report=dict(train_subjects=[1, 3], held_out_subject=2),
            selected_source_ids=[[1, 0, 0]], reference_source_ids=[[3, 0, 0]])
        validate(p)
        bad = copy.deepcopy(p); bad['selected_source_ids'] = [[2, 0, 0]]
        with self.assertRaises(ValueError): validate(bad)
        bad = dict(p, format_version=2)
        with self.assertRaises(ValueError): validate(bad)
