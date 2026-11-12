"""自动退出收据与缺失/跳过项的纯标准库测试，云端不执行。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from upload_run_diagnostics import collect, log_exit_evidence


class AutoEvidenceTests(unittest.TestCase):
    def test_missing_receipt_does_not_invent_interruption(self):
        with tempfile.TemporaryDirectory() as root:
            row = log_exit_evidence(Path(root) / 'train.log', 'logs/train_01.log')
            self.assertIsNone(row['python_exit'])
            self.assertFalse(row['receipt_present'])

    def test_success_failure_and_malformed_receipts(self):
        with tempfile.TemporaryDirectory() as root:
            log = Path(root) / 'train.log'
            receipt = Path(str(log) + '.exit')
            for code in (0, 130, -9):
                receipt.write_text(f'python_exit={code}\ntee_exit=0\n')
                row = log_exit_evidence(log, 'logs/train_01.log')
                self.assertEqual(row['python_exit'], code)
                self.assertEqual(row['tee_exit'], 0)
            receipt.write_text('python_exit=0\ntee_exit=0\npassword=should_not_be_collected\n')
            row = log_exit_evidence(log, 'logs/train_01.log')
            self.assertIsNone(row['python_exit'])
            self.assertIn('issue', row)
            self.assertNotIn('should_not_be_collected', json.dumps(row))

    def test_symlink_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            log = Path(root) / 'train.log'
            secret = Path(root) / 'other'
            secret.write_text('python_exit=0\ntee_exit=0\n')
            Path(str(log) + '.exit').symlink_to(secret)
            self.assertIn('issue', log_exit_evidence(log, 'logs/train_01.log'))

    def test_collect_persists_receipt_and_skip_reasons_without_touching_run(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); run = root / 'run'; bundle = root / 'bundle'
            run.mkdir(); bundle.mkdir()
            (run / 'train.log').write_text('已有诊断输出\n')
            (run / 'train.log.exit').write_text('python_exit=130\ntee_exit=0\n')
            (run / 'config.json').write_text('{}')
            # 超限文件不得上传；原因由manifest自动回传。
            (run / 'probe_summary.json').write_bytes(b' ' * (1024**2 + 1))
            before = {p.name: p.read_bytes() for p in run.iterdir()}
            args = SimpleNamespace(max_file_mib=1, max_total_mib=2, log=[])
            collect(root, run, bundle, args, {})
            m = json.loads((bundle / 'manifest.json').read_text())
            s = json.loads((bundle / 'upload_summary.json').read_text())
            self.assertEqual(m['execution_evidence'][0]['python_exit'], 130)
            self.assertEqual(s['execution_evidence'], m['execution_evidence'])
            self.assertTrue(any(x['file'] == 'probe_summary.json' for x in m['skipped']))
            self.assertNotIn('train.log.exit', [x['file'] for x in m['uploaded']])
            self.assertEqual(before, {p.name: p.read_bytes() for p in run.iterdir()})


if __name__ == '__main__':
    unittest.main()
