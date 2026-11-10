"""第三阶段：只测试结果产物夹具、日志退出码与数值守卫，不做真实数据实验。"""
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from CalculateResult import get_args as result_args
from Datasets import DATASET_SPECS
from ModelTrain import prediction_probabilities
from mainNeuroFDAN import get_args as train_args
from result_audit import SPECS, audit_run, export_report


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def make_run(root):
    """构造24个很小的HCI产物夹具；空权重占位仅用于验证不加载checkpoint。"""
    dataset, label, n, count = 'HCI', 'arousal', 24, 100
    matrices = np.zeros((n, 2, 2), dtype=np.int64)
    for subject in range(1, n + 1):
        folder = root / f'fold{subject:02d}'
        folder.mkdir()
        train = [s for s in range(1, n + 1) if s != subject]
        args = dict(dataset=dataset, label=label, seed=42, epochs=1, batch_size=40, lr=.0001,
                    fd_mode='full', no_balance=False, lambda_obs=1., fire_weight=None,
                    max_train_batches=0, max_test_batches=0)
        cfg = dict(dataset=dataset, held_out_subject=subject, seed=42, no_validation=True,
                   train_subjects=train, test_windows=count, population_windows=2300,
                   population_subjects=23, actual_N_ref=1, requested_N_ref=1,
                   actual_subject_count=1, shortfall=False, source_check_passed=True,
                   arguments=args, model={'num_classes': 2, 'mediator_family': 'normal', 'S_eval': 16}, smoke=False)
        save_json(folder / 'fold.json', cfg)
        save_json(folder / 'reference_sources.json', [dict(subject_id=train[0], trial_id=0, window_id=0, dataset=dataset)])
        metrics = dict(acc=1., selection_f1=1., count=count, complete_test=True)
        save_json(folder / 'summary.json', dict(held_out=subject, best={'acc': 1., 'f1': 1., 'epoch': 1},
                                              last_test=metrics, has_eligible_best=True, smoke=False))
        save_json(folder / 'history.jsonl', dict(epoch=1, train={'loss': .1}, test=metrics))
        for name in ('last.pt', 'best.pt'):
            (folder / name).write_bytes('仅文件存在性夹具，不是真实模型'.encode())
        matrices[subject - 1, 0, 0] = count
        np.save(folder / 'best_confusion.npy', matrices[subject - 1])
        save_json(folder / 'best_predictions.json', [dict(subject_id=subject, trial_id=i // 5,
                  window_id=i % 5, dataset=dataset, label=0, probabilities=[.8, .2]) for i in range(count)])
    np.save(root / 'NeuroFDAN-HCI_arousal_cm.npy', matrices)


class ResultAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        make_run(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_specs_match_loader_and_complete_export(self):
        for name, values in SPECS.items():
            source = DATASET_SPECS[name]
            self.assertEqual(values, (source['subjects'], source['trials'], source['windows']))
        report = audit_run(self.root, 'HCI', 'arousal', 0)
        self.assertTrue(report['complete'])
        self.assertEqual(report['eligible_subjects'], 24)
        self.assertFalse(report['checkpoint_contents_verified'])
        export_report(report, self.root / 'table')
        self.assertTrue((self.root / 'table/per_subject.csv').exists())
        self.assertTrue((self.root / 'table/summary.csv').exists())
        self.assertEqual(report['mean'][0], 1.)

    def test_no_best_not_silently_averaged_as_full(self):
        folder = self.root / 'fold01'
        summary = json.loads((folder / 'summary.json').read_text())
        summary.update(has_eligible_best=False, best={'epoch': 0, 'acc': 0., 'f1': 0.})
        summary['last_test'].update(acc=.5, selection_f1=2/3)
        save_json(folder / 'summary.json', summary)
        save_json(folder / 'history.jsonl', dict(epoch=1, train={'loss': .1}, test=summary['last_test']))
        for name in ('best.pt', 'best_confusion.npy', 'best_predictions.json'):
            (folder / name).unlink()
        matrices = np.load(self.root / 'NeuroFDAN-HCI_arousal_cm.npy')
        matrices[0] = 0
        np.save(self.root / 'NeuroFDAN-HCI_arousal_cm.npy', matrices)
        report = audit_run(self.root, 'HCI', 'arousal', 0)
        self.assertFalse(report['complete'])
        self.assertTrue(report['execution_complete'])
        self.assertEqual(report['folds'][0]['status'], 'completed_no_best')
        self.assertEqual(report['eligible_subjects'], 23)

    def test_missing_fold_and_incomplete_are_distinct(self):
        import shutil
        shutil.rmtree(self.root / 'fold01')
        matrices = np.load(self.root / 'NeuroFDAN-HCI_arousal_cm.npy'); matrices[0] = 0
        np.save(self.root / 'NeuroFDAN-HCI_arousal_cm.npy', matrices)
        (self.root / 'fold02/summary.json').unlink()
        report = audit_run(self.root, 'HCI', 'arousal', 0)
        self.assertEqual(report['folds'][0]['status'], 'not_started')
        self.assertEqual(report['folds'][1]['status'], 'incomplete')
        self.assertFalse(report['complete'])

    def test_smoke_and_reference_leak_excluded(self):
        cfg = json.loads((self.root / 'fold01/fold.json').read_text())
        cfg['smoke'] = True
        save_json(self.root / 'fold01/fold.json', cfg)
        save_json(self.root / 'fold02/reference_sources.json', [dict(subject_id=2, trial_id=0, window_id=0, dataset='HCI')])
        report = audit_run(self.root, 'HCI', 'arousal', 0)
        self.assertEqual(report['folds'][0]['status'], 'smoke')
        self.assertEqual(report['folds'][1]['status'], 'inconsistent')
        self.assertFalse(report['complete'])

    def test_bad_probability_and_configuration_mismatch(self):
        path = self.root / 'fold01/best_predictions.json'
        values = json.loads(path.read_text()); values[0]['probabilities'] = [.2, .2]
        save_json(path, values)
        cfg = json.loads((self.root / 'fold03/fold.json').read_text())
        cfg['model']['S_eval'] = 32
        save_json(self.root / 'fold03/fold.json', cfg)
        report = audit_run(self.root, 'HCI', 'arousal', 0)
        self.assertEqual(report['folds'][0]['status'], 'inconsistent')
        self.assertEqual(report['folds'][2]['status'], 'inconsistent')

    def test_resume_stale_summary_is_not_completed(self):
        cfg = json.loads((self.root / 'fold01/fold.json').read_text())
        cfg['arguments']['epochs'] = 2
        save_json(self.root / 'fold01/fold.json', cfg)
        report = audit_run(self.root, 'HCI', 'arousal', 0)
        self.assertEqual(report['folds'][0]['status'], 'incomplete')

    def test_cli_require_complete_exit_and_no_stale_export(self):
        command = [sys.executable, 'CalculateResult.py', '--dataset', 'HCI', '--label', 'arousal',
                   '--run_dir', str(self.root), '--export_dir', str(self.root / 'exports'), '--require_complete']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.root / 'fold01/summary.json').unlink()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        report = json.loads((self.root / 'exports/audit.json').read_text())
        self.assertFalse(report['complete'])
        self.assertEqual(len(report['folds']), 24)
        self.assertIn('False', (self.root / 'exports/summary.csv').read_text(encoding='utf-8-sig'))


class RuntimeSupportTests(unittest.TestCase):
    def test_primary_probability_guard_preserves_valid_gradient(self):
        p = torch.tensor([[.2, .8]], requires_grad=True)
        result = prediction_probabilities({'probs_do': p}, 'full')
        self.assertIs(result, p)
        result[:, 1].sum().backward()
        torch.testing.assert_close(p.grad, torch.tensor([[0., 1.]]))
        for bad in (torch.tensor([[float('nan'), .5]]), torch.tensor([[float('inf'), 0.]]),
                    torch.tensor([[.2, .2]]), torch.tensor([[-.1, 1.1]]), torch.tensor([.2, .8])):
            with self.assertRaises(FloatingPointError):
                prediction_probabilities({'probs_obs': bad}, 'none')

    def test_logger_preserves_python_and_tee_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / '带空格 日志.txt'
            for status in (0, 7):
                result = subprocess.run(['bash', 'scripts/phase3_run_logged.sh', str(log),
                                         sys.executable, '-c', f"print('临时子进程接口'); raise SystemExit({status})"], capture_output=True, text=True)
                self.assertEqual(result.returncode, status)
                self.assertIn(f'python_exit={status}', Path(str(log) + '.exit').read_text())
            self.assertEqual(log.read_text().count('临时子进程接口'), 2)
            directory = Path(tmp) / '目录不能作日志'
            directory.mkdir()
            result = subprocess.run(['bash', 'scripts/phase3_run_logged.sh', str(directory), sys.executable, '-c', "print('接口')"], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('python_exit=0', Path(str(directory) + '.exit').read_text())

    def test_documented_cli_commands_parse(self):
        guide = Path('docs/phase3_server_guide.md').read_text().replace('\\\n', ' ')
        counts = {'train': 0, 'result': 0}
        for line in guide.splitlines():
            for marker, parser, key in [('python mainNeuroFDAN.py', train_args, 'train'),
                                         ('python CalculateResult.py', result_args, 'result')]:
                if marker in line and '--dataset' in line:
                    argv = shlex.split(line.split(marker, 1)[1])
                    parser(argv=argv)
                    counts[key] += 1
        self.assertGreaterEqual(counts['train'], 9)
        self.assertGreaterEqual(counts['result'], 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
