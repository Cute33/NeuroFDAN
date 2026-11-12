"""摘要器的服务器端纯标准库测试；不加载模型、不执行训练。"""
import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPT))
spec = importlib.util.spec_from_file_location('scheme_summary', SCRIPT / 'summarize_scheme_a_fit.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SchemeSummaryTests(unittest.TestCase):
    def test_zero_point_five_ba_is_not_assumed_single_class(self):
        self.assertFalse(summary.confusion_counts([[1, 1], [1, 1]])['single_class_prediction'])
        row = summary.confusion_counts([[0, 2], [0, 3]])
        self.assertTrue(row['single_class_prediction'])
        self.assertEqual(row['predicted_counts'], [0, 5])
        self.assertEqual(row['true_counts'], [2, 3])

    def test_recovers_prior_gap_and_preserves_source_without_id_arrays(self):
        probe = dict(passed=False, actual_steps=300, constant_prior_nll=.67,
            selected_source_ids=[[1, 2, 3]], records=[dict(step=300, passed=False,
            draws=[dict(balanced_accuracy=.5, integrated_observational_nll=.66,
                        confusion_matrix=[[0, 2], [0, 3]])])])
        before = copy.deepcopy(probe)
        result = summary.build_summary(probe, dict(stages={'fold20': {'status': 'not_started'}}))
        self.assertAlmostEqual(result['checks'][0]['draws'][0]['nll_improvement_over_constant'], .01)
        self.assertNotIn('selected_source_ids', result)
        self.assertEqual(result['workflow']['stages']['fold20']['status'], 'not_started')
        self.assertEqual(probe, before)
        json.dumps(result, allow_nan=False)

    def test_zero_task_norm_yields_null_ratio_and_weighted_terms_stay_visible(self):
        report = dict(reference_term='integrated_observational_nll',
            weighted_terms={'fire': .01}, gradients={
                'integrated_observational_nll': {'cns': {'l2': 0.}},
                'fire': {'cns': {'l2': 2., 'cosine_to_classification': None}}})
        result = summary.gradient_summary(report)
        self.assertIsNone(result['modules']['cns']['fire']['norm_over_task'])
        self.assertEqual(result['weighted_terms']['fire'], .01)
        json.dumps(result, allow_nan=False)

    def test_bad_matrix_is_rejected(self):
        for cm in ([[1, 2, 3]], [[-1, 2], [3, 4]], [[1., 2], [3, 4]]):
            with self.assertRaises(ValueError):
                summary.confusion_counts(cm)


if __name__ == '__main__':
    unittest.main()
