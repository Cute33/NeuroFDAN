"""上传传输层的纯标准库模拟测试；不访问网络、不导入模型、不运行训练。"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import upload_run_diagnostics as uploader


class UploadTransportTests(unittest.TestCase):
    def test_network_and_local_timeouts_are_separate(self):
        completed = subprocess.CompletedProcess([], 0, '', '')
        with patch.object(uploader, 'GIT_NETWORK_TIMEOUT', 600), patch.object(uploader.subprocess, 'run', return_value=completed) as call, contextlib.redirect_stdout(io.StringIO()) as output:
            uploader.git(Path('/tmp'), 'fetch', 'origin', 'server-diagnostics')
            self.assertEqual(call.call_args.kwargs['timeout'], 600)
            uploader.git(Path('/tmp'), 'status', '--short')
            self.assertEqual(call.call_args.kwargs['timeout'], 60)
        self.assertIn('fetch', output.getvalue())
        self.assertIn('status', output.getvalue())

    def test_timeout_names_phase_and_sanitizes_partial_stderr(self):
        error = subprocess.TimeoutExpired('git', 300, stderr=b'https://name:password@github.com/repo')
        with patch.object(uploader.subprocess, 'run', side_effect=error), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError) as caught:
                uploader.git(Path('/tmp'), 'fetch', 'origin')
        self.assertIn('Git fetch', str(caught.exception))
        self.assertNotIn('name:password', str(caught.exception))

    def test_preserves_explicit_ssh_command(self):
        completed = subprocess.CompletedProcess([], 0, '', '')
        with patch.dict(os.environ, {'GIT_SSH_COMMAND': 'ssh -F /existing/config'}), patch.object(uploader.subprocess, 'run', return_value=completed) as call, contextlib.redirect_stdout(io.StringIO()):
            uploader.git(Path('/tmp'), 'ls-remote', 'origin')
        self.assertEqual(call.call_args.kwargs['env']['GIT_SSH_COMMAND'], 'ssh -F /existing/config')
        self.assertEqual(call.call_args.kwargs['env']['GIT_TERMINAL_PROMPT'], '0')

    def test_nonzero_push_is_returned_for_existing_retry_policy(self):
        completed = subprocess.CompletedProcess([], 1, '', 'non-fast-forward')
        with patch.object(uploader.subprocess, 'run', return_value=completed), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(uploader.git(Path('/tmp'), 'push', 'origin', check=False).returncode, 1)

    def test_receipt_keeps_candidate_if_push_times_out(self):
        candidate = 'a' * 40
        def run(command, **kwargs):
            operation = command[3]
            if operation == 'push':
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            stdout = ''
            if operation == 'config' and len(command) == 5:
                stdout = '测试作者' if command[-1] == 'user.name' else 'test@example.invalid'
            elif operation == 'ls-remote':
                stdout = 'b' * 40 + '\trefs/heads/server-diagnostics\n'
            elif operation == 'rev-parse':
                stdout = candidate + '\n'
            return subprocess.CompletedProcess(command, 0, stdout, '')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root/'bundle'
            bundle.mkdir()
            (bundle/'manifest.json').write_text(json.dumps({'uploaded': []}), encoding='utf-8')
            with patch.object(uploader.subprocess, 'run', side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, '推送结果不确定'):
                    uploader.publish(root, 'git@github.com:Cute33/NeuroFDAN.git', bundle, '独立记录')
            receipt = json.loads((bundle/'upload_state.json').read_text(encoding='utf-8'))
            self.assertEqual(receipt['candidate_sha'], candidate)
            self.assertEqual(receipt['run_id'], '独立记录')
            self.assertIn('尚未核验', receipt['phase'])
            self.assertNotIn('upload_state.json', uploader.WHITE)


if __name__ == '__main__':
    unittest.main()
