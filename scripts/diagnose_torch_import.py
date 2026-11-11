"""标准库收集PyTorch导入诊断；不运行模型、不安装依赖或修改Python环境。"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

WORKER = r'''
import enum
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

def path_info(value):
    if value is None:
        return None
    path = Path(value).resolve()
    return dict(path=str(path), tail='/'.join(path.parts[-3:]),
                path_id=hashlib.sha256(str(path).encode()).hexdigest()[:16],
                within_prefix=Path(sys.prefix).resolve() in path.parents)

def origin(name):
    spec = importlib.util.find_spec(name)
    return None if spec is None else path_info(spec.origin)

try:
    version = importlib.metadata.version('torch')
except importlib.metadata.PackageNotFoundError:
    version = None

print(json.dumps(dict(stage='导入前', python=sys.version, executable=sys.executable,
    isolated=sys.flags.isolated, prefix=sys.prefix, enum_file=path_info(enum.__file__),
    torch_distribution_version=version, torch_origin=origin('torch'),
    torchgen_origin=origin('torchgen')), ensure_ascii=False), flush=True)
import torch
import torchgen.model
print(json.dumps(dict(stage='导入成功', torch_version=str(torch.__version__),
    torch_file=path_info(torch.__file__), torchgen_file=path_info(torchgen.model.__file__),
    compiled_cuda_version=torch.version.cuda), ensure_ascii=False), flush=True)
'''


def as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value or ''


def main() -> None:
    parser = argparse.ArgumentParser(description='对比普通/隔离模式的纯PyTorch导入，不训练、不改环境', add_help=False)
    parser.add_argument('-h', '--help', action='help', help='显示中文帮助')
    parser.add_argument('--output_dir', type=Path, required=True, help='独立新诊断目录')
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if (destination / 'probe_summary.json').exists() or (destination / 'fold.json').exists():
        raise FileExistsError('不能覆盖已有诊断或训练fold，使用新编号')
    attempts = []
    for mode, options in (('normal', []), ('isolated', ['-I'])):
        for attempt in range(1, 4):
            command = [sys.executable, *options, '-u', '-c', WORKER]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
                row = dict(mode=mode, attempt=attempt, returncode=completed.returncode,
                           timed_out=False, stdout=completed.stdout, stderr=completed.stderr)
            except subprocess.TimeoutExpired as error:
                row = dict(mode=mode, attempt=attempt, returncode=None, timed_out=True,
                           stdout=as_text(error.stdout), stderr=as_text(error.stderr))
            for field in ('stdout', 'stderr'):
                row[field + '_truncated'] = len(row[field]) > 16000
                row[field] = row[field][-16000:]
            attempts.append(row)
            print(f'{mode} 第{attempt}次：退出码={row["returncode"]} 超时={row["timed_out"]}', flush=True)
            print(row['stdout'], end='', flush=True)
            if row['stderr']:
                print(row['stderr'], end='', flush=True)
    result = dict(operation='普通与隔离模式的纯PyTorch导入诊断',
        parent_executable=sys.executable, attempts=attempts,
        all_normal_imports_ok=all(r['returncode'] == 0 for r in attempts if r['mode'] == 'normal'),
        all_isolated_imports_ok=all(r['returncode'] == 0 for r in attempts if r['mode'] == 'isolated'),
        training_executed=False, model_loaded=False, packages_changed=False,
        note='-I仅作用于子进程，忽略PYTHONPATH/用户site/当前目录搜索；不禁用所有site定制。退出0表示收集完成，不等于PyTorch已修复。CUDA字符串不是GPU实测。')
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'probe_summary.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'收集完成：{destination / "probe_summary.json"}；未安装或修改依赖', flush=True)


if __name__ == '__main__':
    main()
