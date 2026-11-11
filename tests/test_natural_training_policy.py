"""服务器执行的自然训练默认值及来源选择回归测试；不运行正式训练。"""
import unittest
from types import SimpleNamespace

from mainNeuroFDAN import get_args
from ModelTrain import run_fold
from scripts.check_fixed_batch_capacity import source_grid_indices


class NaturalTrainingPolicyTests(unittest.TestCase):
    def test_cli_defaults_to_natural_population(self):
        args = get_args(argv=['--dataset', 'DEAP', '--label', 'arousal'])
        self.assertTrue(args.no_balance)

    def test_old_explicit_flag_remains_compatible(self):
        args = get_args(argv=['--dataset', 'HCI', '--label', 'arousal', '--no_balance'])
        self.assertTrue(args.no_balance)

    def test_direct_api_rejects_balancing_before_loading_data(self):
        with self.assertRaisesRegex(ValueError, '自然总体'):
            run_fold(SimpleNamespace(no_balance=False), 1)

    def test_grid_preserves_unique_sources_and_endpoints(self):
        ids = source_grid_indices(400, 16)
        self.assertEqual(len(ids), 16)
        self.assertEqual(len(set(ids)), 16)
        self.assertEqual((ids[0], ids[-1]), (0, 399))
        self.assertEqual(ids, sorted(ids))

    def test_grid_does_not_balance_class_counts(self):
        # 来源前100窗为0、后300窗为1，选择结果保留不平衡，而非强行每类8窗。
        labels = [0]*100+[1]*300
        chosen = [labels[i] for i in source_grid_indices(400, 16)]
        self.assertEqual((chosen.count(0), chosen.count(1)), (4, 12))

    def test_invalid_grid_size_is_rejected(self):
        for population, samples in ((1, 2), (400, 1), (8, 16)):
            with self.assertRaises(ValueError):
                source_grid_indices(population, samples)


if __name__ == '__main__':
    unittest.main()
