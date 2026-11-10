"""第二阶段轻量级接口测试；临时数组文件不是真实数据实验。"""
import ast
import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from Datasets import (PhysiologicalWindows, WindowSelection, make_loaders,
                      _compat_dataset, normalize_subject, DATASET_SPECS)
from ModelTrain import (BestCheckpoint, build_reference_bank, cache_reference_features,
                        confusion_metrics, seed_everything, run_fold, save_checkpoint, load_checkpoint,
                        save_fold_confusion)
from NeuroFDAN import NeuroFDAN
from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator, frontdoor_loss
from mainNeuroFDAN import get_args
from CalculateResult import summarize


def tiny_config(**kwargs):
    return FrontDoorConfig(hidden_dim=8, mediator_dim=4, dropout=0., ref_dim=8,
                           S_train=2, K_train=2, S_eval=2, K_eval=3, reference_chunk_size=2, **kwargs)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        path = cls.root / 'Dataset/HCI-python'
        path.mkdir(parents=True)
        t = np.arange(20, dtype=np.float32)[:, None, None]
        channel = np.arange(36, dtype=np.float32)[None, :, None]
        times = np.arange(3840, dtype=np.float32)[None, None, :]
        raw = t + .01 * channel + .00001 * times
        labels = np.stack((np.where(np.arange(20) < 10, 7., 3.), np.where(np.arange(20) % 2, 8., 2.)), 1)
        for subject in (1, 2, 3):
            with (path / f'{subject}.npy').open('wb') as stream:
                pickle.dump({'data': raw + subject, 'labels': labels}, stream)
        cls.population = PhysiologicalWindows(cls.root, 'HCI', [1, 2], 'valence')
        cls.test_population = PhysiologicalWindows(cls.root, 'HCI', [3], 'valence')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_hci_windows_labels_metadata(self):
        population = self.population
        self.assertEqual(len(population), 200)
        self.assertEqual(population.x_c.shape, (200, 32, 768))
        self.assertEqual(population.x_a.shape, (200, 4, 768))
        for index in (0, 4, 5, 99, 100, 199):
            batch, returned = population[index]
            subject, trial, window = population.source_ids[index].tolist()
            self.assertEqual((subject, trial, window), (index // 100 + 1, index % 100 // 5, index % 5))
            self.assertEqual(batch['Labels'].item(), int(trial < 10))
            self.assertEqual(batch['dataset'], 'HCI')
            torch.testing.assert_close(batch['ECG'], batch['x_a'][:-1])
            torch.testing.assert_close(batch['GSR'], batch['x_a'][-1:])
            self.assertEqual(returned, index)

    def test_deap_dimensions_baseline_windows_and_threshold(self):
        spec = DATASET_SPECS['DEAP']
        trials = np.arange(40, dtype=np.float32)[:, None, None]
        raw = np.broadcast_to(trials, (40, 37, 8064))
        labels = np.stack((np.where(np.arange(40) % 2, 5., 6.), np.full(40, 5.)), 1)
        # 读取器仍走原pickle文件接口；只将文件内容替换为一个小型可控结构夹具。
        path = self.root / 'Dataset/DEAP-python'
        path.mkdir(exist_ok=True)
        (path / '1.npy').touch()
        with patch('Datasets.pickle.load', return_value={'data': raw, 'labels': labels}):
            population = PhysiologicalWindows(self.root, 'DEAP', [1], 'valence')
        self.assertEqual(len(population), 400)
        self.assertEqual(population.x_c.shape, (400, 32, 768))
        self.assertEqual(population.x_a.shape, (400, 3, 768))
        self.assertEqual(population.source_ids[-1].tolist(), [1, 39, 9])
        self.assertEqual(population[10][0]['Labels'].item(), 0)
        self.assertEqual(population[0][0]['Labels'].item(), 1)

    def test_valence_forwarded_to_compatibility_constructor(self):
        with patch('Datasets.PhysiologicalWindows') as constructor:
            _compat_dataset(self.root, ['1.npy'], 'valence', 'DEAP')
            constructor.assert_called_once_with(self.root, 'DEAP', [1], 'valence')

    def test_balance_keeps_pairing_and_unbalanced_population(self):
        population = self.population
        before = population.source_ids.clone()
        indices = population.balanced_indices()
        selected = WindowSelection(population, indices)
        self.assertEqual(int(selected.Label.sum()), len(selected) // 2)
        torch.testing.assert_close(population.source_ids, before)
        for i in (0, 10, len(selected) - 1):
            batch, source_index = selected[i]
            torch.testing.assert_close(batch['x_c'], population.x_c[source_index])
            torch.testing.assert_close(batch['source_ids'], population.source_ids[source_index])

    def test_deap_balance_uses_task_labels_not_arousal(self):
        population = object.__new__(PhysiologicalWindows)
        population.windows_per_trial, population.trials_per_subject = 10, 40
        # 四个被试中只有每人10个正类trial，正确valence平衡保留80个trial。
        population.Label = torch.tensor(([1] * 10 + [0] * 30) * 4).repeat_interleave(10)
        selected = population.balanced_indices()
        self.assertEqual(len(selected), 800)
        self.assertEqual(population.Label[selected].sum().item(), 400)

    def test_normalization_local_axis_and_constant(self):
        raw = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        expected = (raw - raw.mean(0, keepdims=True)) / raw.std(0, keepdims=True)
        np.testing.assert_allclose(normalize_subject(raw), expected)
        np.testing.assert_array_equal(normalize_subject(np.ones((3, 4, 5))), np.zeros((3, 4, 5)))
        np.testing.assert_allclose(normalize_subject(raw), normalize_subject(raw + 100), atol=1e-6)

    def test_reference_population_not_balanced_and_no_test(self):
        raw = build_reference_bank(self.population, [1, 2], 3, N_ref=999)
        self.assertEqual(raw.used_count, 200)
        self.assertTrue(raw.shortfall)
        self.assertEqual(raw.dataset, 'HCI')
        self.assertEqual(set(raw.source_ids[:, 0].tolist()), {1, 2})
        self.assertTrue(raw.report['source_check_passed'])
        with self.assertRaises(ValueError):
            build_reference_bank(self.population, [1], 2)
        with self.assertRaises(ValueError):
            build_reference_bank(self.population, [1, 2, 3], 3)
        with self.assertRaises(ValueError):
            build_reference_bank(WindowSelection(self.population, torch.arange(10)), [1, 2], 3)

    def test_loader_single_instance_sources_and_drop_last(self):
        selected = WindowSelection(self.population, torch.arange(199))
        train, test = make_loaders(selected, self.test_population, batch_size=6)
        self.assertIs(train.dataset, train.sampler.data_source)
        self.assertTrue(train.drop_last)
        self.assertFalse(test.drop_last)
        self.assertEqual(len(train) * 6, 198)
        self.assertEqual(len(test.dataset), 100)
        first, _ = next(iter(train))
        self.assertTrue(set(first['subject_id'].tolist()).issubset({1, 2}))
        self.assertTrue((first['source_ids'][:, 0] != 3).all())

    def test_one_batch_fold_cpu_checkpoint_no_formal_results(self):
        args = get_args(argv=['--dataset', 'HCI', '--label', 'valence', '--device', 'cpu',
             '--held_out_subject', '3', '--root_dir', str(self.root), '--output_dir', str(self.root / 'outputs'),
             '--run_name', '接口夹具', '--no_balance', '--epochs', '1', '--batch_size', '2',
             '--hidden_dim', '8', '--mediator_dim', '4', '--ref_dim', '8', '--N_ref', '6',
             '--S_train', '1', '--K_train', '2', '--S_eval', '1', '--K_eval', '2',
             '--max_train_batches', '1', '--max_test_batches', '1'])
        summary = run_fold(args, 3, population=self.population, test_dataset=self.test_population)
        self.assertTrue(summary['smoke'])
        self.assertEqual(summary['last_test']['count'], 2)
        self.assertFalse(summary['last_test']['complete_test'])
        folder = self.root / 'outputs/HCI/valence/seed42/接口夹具/fold03'
        self.assertTrue((folder / 'last.pt').is_file())
        sources = json.loads((folder / 'reference_sources.json').read_text())
        self.assertTrue(all(row['subject_id'] != 3 and row['dataset'] == 'HCI' for row in sources))
        state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        model = NeuroFDAN(frontdoor_config=FrontDoorConfig(**state['frontdoor_config']))
        load_checkpoint(folder / 'last.pt', model, map_location='cpu')
        self.assertGreater(model.feature_version, 0)
        with self.assertRaises(FileExistsError):
            run_fold(args, 3, population=self.population, test_dataset=self.test_population)
        args.resume = str(folder / 'last.pt')
        args.epochs = 2
        resumed = run_fold(args, 3, population=self.population, test_dataset=self.test_population)
        self.assertTrue(resumed['smoke'])
        state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(state['epoch'], 2)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_everything()
        self.model = NeuroFDAN(frontdoor_config=tiny_config())
        self.xc, self.xa = torch.randn(2, 32, 768), torch.randn(2, 4, 768)
        self.ids = torch.tensor([[1, 0, 0], [2, 0, 0]])

    def test_single_model_definition_and_no_prototype_path(self):
        tree = ast.parse(Path('NeuroFDAN.py').read_text())
        classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        self.assertEqual(len(classes), len(set(classes)))
        self.assertEqual(classes.count('NeuroFDAN'), 1)
        self.assertNotIn('CausalMemoryBank', classes)
        self.assertNotIn('FrontDoorInterventionModule', classes)

    def test_both_dataset_feature_dimensions_and_full_output(self):
        for channels, ref_dim in [(2, 13440), (3, 13824)]:
            cfg = tiny_config()
            cfg.ref_dim = 0
            model = NeuroFDAN(ecg_channels=channels, frontdoor_config=cfg)
            ans = self.xa[:, :channels + 1]
            out = model(self.xc, ans, source_ids=self.ids)
            self.assertEqual(out['probs_do'].shape, (2, 2))
            self.assertEqual(out['mu_a'].shape, (2, 2, 4))
            self.assertEqual(model.frontdoor.reference_encoder.output_dim, ref_dim)
            self.assertIn('logits_factual', out)
            self.assertIn('mediator_statistics', out)
            self.assertIn('reference_statistics', out)

    def test_backbone_preserved_and_batchnorm_frozen(self):
        model = self.model.train()
        for module in model.backbone.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                self.assertFalse(module.training)
                self.assertTrue(module.weight.requires_grad)
        model.eval()
        hc, ha = model.backbone(self.xc, self.xa)
        self.assertGreater(hc.abs().sum().item(), 0)
        self.assertGreater((hc[0] - hc[1]).abs().sum().item(), 0)
        first = model.backbone(self.xc[:1], self.xa[:1])
        torch.testing.assert_close(hc[:1], first[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(ha[:1], first[1], atol=1e-6, rtol=1e-5)

    def test_live_reference_gradient_and_conditional_path(self):
        bank = __import__('frontdoor_scm').ReferenceBank(self.xc, self.xa, self.ids, dataset='HCI')
        bank.raw_windows = True
        with patch.object(self.model.frontdoor, 'marginalize', wraps=self.model.frontdoor.marginalize) as marginal:
            output = self.model(self.xc, self.xa, reference_bank=bank, source_ids=self.ids)
        self.assertEqual(marginal.call_count, 1)
        objective = output['probs_do'][:, 1].sum()
        gradients = torch.autograd.grad(objective, (self.model.backbone.ans_enc[0].weight,
                                                   self.model.frontdoor.ans.cns_current.weight))
        for gradient in gradients:
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_live_reference_backbone_not_detached(self):
        bank = __import__('frontdoor_scm').ReferenceBank(self.xc, self.xa, self.ids, dataset='HCI')
        bank.raw_windows = True
        captured = []
        hook = self.model.backbone.register_forward_hook(lambda module, args, out: captured.append(out))
        output = self.model(self.xc, self.xa, reference_bank=bank, source_ids=self.ids)
        hook.remove()
        self.assertEqual(len(captured), 2)
        gradients = torch.autograd.grad(output['probs_do'][:, 1].sum(), captured[-1], allow_unused=True)
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_cached_reference_expiry_and_cpu_eval(self):
        class Bank:
            pass
        bank = Bank()
        bank.x_c, bank.x_a, bank.source_ids = self.xc, self.xa, self.ids
        bank.used_count, bank.shortfall, bank.population_count, bank.dataset = 2, False, 2, 'HCI'
        bank.report = {'held_out_subject': 3}
        cached = cache_reference_features(self.model, bank, torch.device('cpu'), 2)
        self.model.eval()
        output = self.model(self.xc, self.xa, reference_bank=cached)
        self.assertTrue(torch.isfinite(output['probs_do']).all())
        with torch.no_grad():
            self.model.backbone.dag_adj.add_(.1)
        with self.assertRaises(ValueError):
            self.model(self.xc, self.xa, reference_bank=cached)

    def test_forward_backward_optimizer_step_and_fact_loss(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.0001)
        output = self.model(self.xc, self.xa, source_ids=self.ids)
        with patch.dict(output, {'probs_do': torch.full((2, 2), float('nan'))}):
            terms = frontdoor_loss(output, torch.tensor([0, 1]), self.model.frontdoor_config)
            self.assertTrue(torch.isfinite(terms['loss']))
        before = self.model.backbone.cns_enc[0].weight.detach().clone()
        (terms['loss'] + output['fire_loss']).backward()
        optimizer.step()
        self.model.weights_updated()
        self.assertFalse(torch.equal(before, self.model.backbone.cns_enc[0].weight))

    def test_no_cns_path_and_none_modes_remain_explicit(self):
        model = NeuroFDAN(frontdoor_config=tiny_config(no_cns2ans=True))
        output = model(self.xc, self.xa, fd_mode='none')
        self.assertNotIn('probs_do', output)
        self.assertIsNone(model.frontdoor.ans.cns_current)
        self.assertEqual(output['reference_statistics']['K'], 0)
        self.assertEqual(output['reference_statistics']['bank_count'], 0)

    def test_reference_chunking_preserves_probability_formula(self):
        a = FrontDoorEstimator(5, 3, FrontDoorConfig(dropout=0.))
        b = copy.deepcopy(a)
        b.config.reference_chunk_size = 2
        ma, reference = torch.randn(2, 3, 64), torch.randn(2, 7, 8)
        torch.testing.assert_close(a.marginalize(ma, reference)[0], b.marginalize(ma, reference)[0])

    def test_seed_before_initialization_and_no_unconditional_cuda(self):
        seed_everything()
        first = NeuroFDAN(frontdoor_config=tiny_config())
        seed_everything()
        second = NeuroFDAN(frontdoor_config=tiny_config())
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b)
        with patch('torch.cuda.is_available', return_value=False), patch('torch.cuda.get_device_name', side_effect=AssertionError()):
            seed_everything()
            from ModelTrain import resolve_device
            self.assertEqual(resolve_device('0').type, 'cpu')

    def test_legacy_ce_constructor_fixed_without_weight_changes(self):
        from loss import CausalMultiLoss
        criterion = CausalMultiLoss(loss_type='ce')
        torch.testing.assert_close(criterion.criterion.weight, torch.tensor([.6, 1.5]))


class ProtocolTests(unittest.TestCase):
    def test_fold_matrix_updates_preserve_other_subjects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subject in (1, 2):
                folder = root / f'fold{subject:02d}'
                folder.mkdir()
                save_fold_confusion(root, folder, 'HCI', 'valence', subject, torch.eye(2, dtype=torch.long) * subject)
            matrices = np.load(root / 'NeuroFDAN-HCI_valence_cm.npy')
            np.testing.assert_array_equal(matrices[0], np.eye(2))
            np.testing.assert_array_equal(matrices[1], np.eye(2) * 2)

    def test_checkpoint_acc_gate_and_dataset_specific_f1_tie(self):
        tracker = BestCheckpoint()
        self.assertFalse(tracker.consider({'acc': .5, 'selection_f1': .8}, 1))
        self.assertTrue(tracker.consider({'acc': .6, 'selection_f1': .5}, 2))
        self.assertTrue(tracker.consider({'acc': .6, 'selection_f1': .6}, 3))
        self.assertFalse(tracker.consider({'acc': .59, 'selection_f1': .9}, 4))
        cm = torch.tensor([[3, 1], [2, 4]])
        self.assertEqual(confusion_metrics(cm, 'DEAP')['selection_f1'], 8 / 11)
        self.assertEqual(confusion_metrics(cm, 'HCI')['selection_f1'], 6 / 9)

    def test_results_ignore_zero_unrun_folds(self):
        result = summarize(np.array([[[8, 2], [1, 9]], [[0, 0], [0, 0]]]))
        self.assertEqual(result['completed_subjects'], 1)
        self.assertAlmostEqual(result['mean'][0], .85)


if __name__ == '__main__':
    unittest.main(verbosity=2)
