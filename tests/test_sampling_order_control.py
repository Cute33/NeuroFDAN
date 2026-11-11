"""采样顺序对照的CPU小测试；不读取生理数据，不训练模型。"""
import unittest

import torch
from torch.utils.data import TensorDataset

from Datasets import GroupedSampler, make_loaders
from ModelTrain import validate_resume_sampling_mode
from mainNeuroFDAN import get_args


class SamplingOrderTests(unittest.TestCase):
    def setUp(self):
        self.data = TensorDataset(torch.arange(400))
        self.data.windows_per_trial = 10

    def test_default_cli_and_original_sequence(self):
        self.assertEqual(get_args(argv=['--label', 'arousal']).sampling_mode, 'grouped')
        loader, _ = make_loaders(self.data, self.data)
        old = GroupedSampler(self.data, 10, seed=42)
        for _ in range(2):
            self.assertEqual(list(loader.sampler), list(old))

    def test_window_is_uniform_permutation_with_independent_generator(self):
        loader, _ = make_loaders(self.data, self.data, sampling_mode='window')
        expected = torch.Generator().manual_seed(42)
        before = torch.random.get_rng_state().clone()
        for _ in range(2):
            order = list(loader.sampler)
            self.assertEqual(order, torch.randperm(400, generator=expected).tolist())
            self.assertEqual(sorted(order), list(range(400)))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_sample_set_and_test_order_are_unchanged(self):
        grouped, test_grouped = make_loaders(self.data, self.data)
        window, test_window = make_loaders(self.data, self.data, sampling_mode='window')
        self.assertIs(grouped.dataset, window.dataset)
        self.assertEqual(list(test_grouped.sampler), list(test_window.sampler))
        self.assertEqual(len({i // 10 for i in list(grouped.sampler)[:40]}), 4)
        self.assertGreater(len({i // 10 for i in list(window.sampler)[:40]}), 4)

    def test_old_and_new_checkpoint_resume_modes(self):
        validate_resume_sampling_mode({'arguments': {}}, 'grouped')
        validate_resume_sampling_mode({'arguments': {'sampling_mode': 'window'}}, 'window')
        for saved, requested in (({}, 'window'), ({'sampling_mode': 'window'}, 'grouped')):
            with self.assertRaises(ValueError):
                validate_resume_sampling_mode({'arguments': saved}, requested)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            make_loaders(self.data, self.data, sampling_mode='invalid')


if __name__ == '__main__':
    unittest.main()
