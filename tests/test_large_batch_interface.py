"""大训练batch与评估分批的CPU接口测试；不读取真实数据、不运行模型。"""
import unittest

import torch
from torch.utils.data import TensorDataset

from Datasets import make_loaders
from ModelTrain import validate_resume_batch_settings
from mainNeuroFDAN import get_args


class LargeBatchTests(unittest.TestCase):
    def setUp(self):
        self.data = TensorDataset(torch.arange(800))
        self.data.windows_per_trial = 10

    def test_train400_eval40_and_default_compatibility(self):
        train, test = make_loaders(self.data, self.data, batch_size=400,
                                   sampling_mode='window', eval_batch_size=40)
        self.assertEqual((len(train), len(test)), (2, 20))
        self.assertEqual((train.batch_size, test.batch_size), (400, 40))
        old_train, old_test = make_loaders(self.data, self.data, batch_size=40)
        self.assertEqual((old_train.batch_size, old_test.batch_size), (40, 40))

    def test_window_indices_once_per_epoch_and_same_test_order(self):
        a, _ = make_loaders(self.data, self.data, batch_size=400, sampling_mode='window')
        b, test = make_loaders(self.data, self.data, batch_size=40, sampling_mode='window', eval_batch_size=40)
        self.assertEqual(list(a.sampler), list(b.sampler))
        self.assertEqual(list(test.sampler), list(range(800)))

    def test_cli_and_invalid_eval_batch(self):
        args = get_args(argv=['--label', 'arousal', '--batch_size', '400', '--eval_batch_size', '40'])
        self.assertEqual((args.batch_size, args.eval_batch_size), (400, 40))
        self.assertIsNone(get_args(argv=['--label', 'arousal']).eval_batch_size)
        with self.assertRaises(SystemExit):
            get_args(argv=['--label', 'arousal', '--eval_batch_size', '0'])
        with self.assertRaises(ValueError):
            make_loaders(self.data, self.data, eval_batch_size=0)

    def test_resume_legacy_and_explicit_batches(self):
        legacy = {'arguments': {'batch_size': 40}}
        validate_resume_batch_settings(legacy, 40, 40)
        with self.assertRaises(ValueError):
            validate_resume_batch_settings(legacy, 400, 40)
        explicit = {'arguments': {'batch_size': 400, 'eval_batch_size': 40}}
        validate_resume_batch_settings(explicit, 400, 40)
        with self.assertRaises(ValueError):
            validate_resume_batch_settings(explicit, 400, 400)


if __name__ == '__main__':
    unittest.main()
