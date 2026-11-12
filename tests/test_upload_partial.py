"""服务器端本地file:// Git集成测试；没有GitHub网络、模型或训练。"""
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import upload_run_diagnostics as uploader


def local_git(directory, *args):
    return subprocess.run(['git', '-C', str(directory), *args], check=True,
                          capture_output=True, text=True, timeout=30).stdout.strip()


class PartialUploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.seed = self.root / 'seed'
        self.remote = self.root / 'remote.git'
        self.seed.mkdir()
        self.remote.mkdir()
        local_git(self.remote, 'init', '--bare')
        local_git(self.remote, 'config', 'uploadpack.allowFilter', 'true')
        local_git(self.seed, 'init')
        local_git(self.seed, 'config', 'user.name', 'Upload test')
        local_git(self.seed, 'config', 'user.email', 'upload-test@example.invalid')
        local_git(self.seed, 'config', 'commit.gpgsign', 'false')
        local_git(self.seed, 'remote', 'add', 'origin', self.remote.as_uri())
        old = self.seed / 'runs' / 'history' / 'old.log'
        old.parent.mkdir(parents=True)
        old.write_bytes(bytes(range(256)) * 4096)
        (self.seed / 'README.md').write_text('历史结果必须保留\n', encoding='utf-8')
        local_git(self.seed, 'add', '-f', '--', 'runs', 'README.md')
        local_git(self.seed, 'commit', '-m', 'historic results')
        self.parent = local_git(self.seed, 'rev-parse', 'HEAD')
        self.old_oid = local_git(self.seed, 'rev-parse', 'HEAD:runs/history/old.log')

    def bundle(self, name='bundle'):
        bundle = self.root / name
        bundle.mkdir()
        raw = b'{"completed": true, "passed": false}\n'
        (bundle / 'config.json').write_bytes(raw)
        manifest = {'uploaded': [{'file': 'config.json', 'bytes': len(raw),
                                 'sha256': hashlib.sha256(raw).hexdigest()}]}
        (bundle / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return bundle

    def publish(self, bundle, run_id, capture=None):
        real_git = uploader.git
        def transport(cwd, *args, **kwargs):
            result = real_git(cwd, *args, **kwargs)
            if capture is not None:
                capture['operations'].append(args)
                if args[0] in ('fetch', 'push'):
                    # --missing=print不触发对旧blob的惰性下载。
                    missing = local_git(cwd, 'rev-list', '--objects', '--missing=print', 'FETCH_HEAD')
                    capture[args[0] + '_old_blob_missing'] = '?' + self.old_oid in missing.splitlines()
                    capture['historical_worktree_exists'] = (cwd / 'runs').exists()
            return result
        with patch.object(uploader, 'git', side_effect=transport), contextlib.redirect_stdout(io.StringIO()):
            return uploader.publish(self.seed, self.remote.as_uri(), bundle, run_id)

    def seed_remote(self):
        local_git(self.seed, 'push', 'origin', 'HEAD:refs/heads/server-diagnostics')

    def test_appends_without_downloading_old_blobs_or_changing_old_tree(self):
        self.seed_remote()
        before = local_git(self.remote, 'ls-tree', '-r', self.parent)
        bundle = self.bundle()
        capture = {'operations': []}
        sha, observed = self.publish(bundle, 'new_snapshot', capture)
        self.assertEqual(sha, observed)
        self.assertEqual(local_git(self.remote, 'rev-parse', sha + '^'), self.parent)
        self.assertEqual(local_git(self.remote, 'rev-parse', 'refs/heads/server-diagnostics'), sha)
        after = local_git(self.remote, 'ls-tree', '-r', sha)
        self.assertTrue(set(before.splitlines()).issubset(after.splitlines()))
        self.assertEqual(local_git(self.remote, 'show', sha + ':runs/new_snapshot/config.json'),
                         (bundle / 'config.json').read_text().strip())
        self.assertTrue(capture['fetch_old_blob_missing'])
        self.assertTrue(capture['push_old_blob_missing'])
        self.assertFalse(capture['historical_worktree_exists'])
        self.assertNotIn('checkout', [args[0] for args in capture['operations']])
        self.assertFalse(any('--force' in args for args in capture['operations']))
        self.assertEqual(json.loads((bundle / 'upload_state.json').read_text())['parent_sha'], self.parent)

    def test_empty_remote_gets_new_root_commit(self):
        sha, observed = self.publish(self.bundle(), 'first_snapshot')
        self.assertEqual(sha, observed)
        self.assertEqual(local_git(self.remote, 'rev-list', '--count', sha), '1')
        self.assertIn('runs/first_snapshot/config.json', local_git(self.remote, 'ls-tree', '-r', '--name-only', sha))

    def test_existing_snapshot_is_never_overwritten(self):
        self.seed_remote()
        with self.assertRaisesRegex(ValueError, '拒绝覆盖历史'):
            self.publish(self.bundle(), 'history')
        self.assertEqual(local_git(self.remote, 'rev-parse', 'refs/heads/server-diagnostics'), self.parent)

    def test_changed_payload_is_rejected_before_push(self):
        self.seed_remote()
        bundle = self.bundle()
        (bundle / 'config.json').write_text('{"changed":true}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '快照在收集后发生变化'):
            self.publish(bundle, 'changed_payload')
        self.assertEqual(local_git(self.remote, 'rev-parse', 'refs/heads/server-diagnostics'), self.parent)
        self.assertIsNone(json.loads((bundle / 'upload_state.json').read_text())['candidate_sha'])

    def test_concurrent_append_rebases_snapshot_without_force(self):
        self.seed_remote()
        real_git = uploader.git
        pushes = []
        def transport(cwd, *args, **kwargs):
            if args[0] == 'push':
                pushes.append(args)
                if len(pushes) == 1:
                    (self.seed / 'concurrent.txt').write_text('并发结果必须保留', encoding='utf-8')
                    local_git(self.seed, 'add', '--', 'concurrent.txt')
                    local_git(self.seed, 'commit', '-m', 'concurrent append')
                    local_git(self.seed, 'push', 'origin', 'HEAD:refs/heads/server-diagnostics')
            return real_git(cwd, *args, **kwargs)
        with patch.object(uploader, 'git', side_effect=transport), contextlib.redirect_stdout(io.StringIO()):
            sha, observed = uploader.publish(self.seed, self.remote.as_uri(), self.bundle(), 'concurrent_snapshot')
        self.assertEqual(sha, observed)
        self.assertEqual(len(pushes), 2)
        self.assertFalse(any('--force' in args for args in pushes))
        self.assertEqual(local_git(self.remote, 'show', sha + ':concurrent.txt'), '并发结果必须保留')
        self.assertEqual(local_git(self.remote, 'rev-parse', sha + ':runs/history/old.log'), self.old_oid)
        self.assertEqual(local_git(self.remote, 'rev-parse', sha + '^'), local_git(self.seed, 'rev-parse', 'HEAD'))


if __name__ == '__main__':
    unittest.main()
