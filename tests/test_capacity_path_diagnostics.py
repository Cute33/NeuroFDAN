"""服务器运行的路径诊断接口测试；人工张量不是真实训练或因果证据。"""
import unittest
from pathlib import Path

import torch

from scripts.capacity_path_diagnostics import original_reference_bank, outcome_path_sensitivity, probability_summary


class ToyEstimator:
    """用已知只读函数验证两种路径能被区分，不替代实际前门估计器。"""
    def __init__(self, ma_weight, ref_weight):
        self.ma_weight, self.ref_weight = ma_weight, ref_weight

    def marginalize(self, ma, ref):
        b, s, _ = ma.shape
        k = ref.shape[1]
        eta = (self.ma_weight*ma[:, :, None, 0].expand(b, s, k) +
               self.ref_weight*ref[:, None, :, 0].expand(b, s, k))
        logits = torch.stack((torch.zeros_like(eta), eta), -1)
        return logits.softmax(-1).mean(2).mean(1), logits


class CapacityPathTests(unittest.TestCase):
    def setUp(self):
        value = torch.tensor([-2., -1., 1., 2.])
        self.ma = value[:, None, None].expand(4, 2, 1).clone()
        self.own = value[:, None].clone()
        self.ref = torch.tensor([-2., 2.])[None, :, None].expand(4, 2, 1).clone()
        self.y = torch.tensor([0, 0, 1, 1])

    def test_mediator_only_dependence_survives_reference_integration(self):
        r = outcome_path_sensitivity(ToyEstimator(4., 0.), self.ma, self.own, self.ref, self.y)
        self.assertEqual(r['empirical_reference_do']['metrics']['acc'], 1.)
        self.assertEqual(r['sensitivity']['rotate_ref_fixed_ma']['mean_probability_l1'], 0.)
        self.assertEqual(r['sensitivity']['rotate_ma_own_ref']['opposite_label_donor_agreement'], 1.)

    def test_reference_only_shortcut_is_detected_and_inputs_unchanged(self):
        before = [v.clone() for v in (self.ma, self.own, self.ref, self.y)]
        r = outcome_path_sensitivity(ToyEstimator(0., 4.), self.ma, self.own, self.ref, self.y)
        self.assertEqual(r['own_reference_same_S']['metrics']['acc'], 1.)
        self.assertEqual(r['sensitivity']['rotate_ma_own_ref']['mean_probability_l1'], 0.)
        self.assertEqual(r['sensitivity']['rotate_ref_fixed_ma']['opposite_label_donor_agreement'], 1.)
        self.assertEqual(r['empirical_reference_do']['std_probability'], [0., 0.])
        for old, value in zip(before, (self.ma, self.own, self.ref, self.y)):
            torch.testing.assert_close(old, value)

    def test_invalid_shapes_and_non_normalized_probability_are_rejected(self):
        with self.assertRaises(ValueError):
            outcome_path_sensitivity(ToyEstimator(1., 1.), self.ma[:, 0], self.own, self.ref, self.y)
        with self.assertRaises(FloatingPointError):
            probability_summary(torch.full((4, 2), .2), self.y)

    def test_heldout_reference_is_rejected_before_reading_data(self):
        payload = dict(reference_report=dict(actual_N_ref=1, held_out_subject=1, train_subjects=[2]),
                       reference_source_ids=torch.tensor([[1, 0, 0]]))
        with self.assertRaises(ValueError):
            original_reference_bank(Path('/不存在的诊断数据路径'), payload)


if __name__ == '__main__':
    unittest.main()
