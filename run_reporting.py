"""仅从已有产物生成运行摘要；不加载模型、信号或checkpoint。"""
import json
import re
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path


def source_version(repository: Path) -> dict:
    """记录启动时源码身份，不收集远程凭证或环境变量。"""
    def git(*args):
        result = subprocess.run(['git', '-C', str(repository), *args], capture_output=True,
                                text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        status = git('status', '--porcelain', '--untracked-files=no')
        return dict(commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                    tracked_changes=None if status is None else bool(status))
    except (OSError, subprocess.TimeoutExpired):
        return dict(commit=None, branch=None, tracked_changes=None)


def _read_json(path: Path, issues: list) -> dict:
    if not path.is_file() or path.is_symlink():
        issues.append(f'缺少{path.name}')
        return {}
    try:
        if path.stat().st_size > 8 * 1024**2:
            raise ValueError('摘要JSON超过8MiB')
        value = json.loads(path.read_text(encoding='utf-8'))
        json.dumps(value, allow_nan=False)
        if not isinstance(value, dict):
            raise ValueError('需要JSON对象')
        return value
    except (OSError, ValueError) as error:
        issues.append(f'{path.name}无法解析：{type(error).__name__}')
        return {}


def _epoch_scan(path: Path, issues: list) -> tuple[dict, int, int]:
    """只保留最后记录；容忍正在写入的尾行，明确标记而不补零。"""
    last, count, single = {}, 0, 0
    if not path.is_file() or path.is_symlink():
        issues.append(f'缺少{path.name}')
        return last, count, single
    try:
        if path.stat().st_size > 64 * 1024**2:
            issues.append(f'{path.name}超过64MiB，不进行摘要扫描')
            return last, count, single
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    json.dumps(row, allow_nan=False)
                    if not isinstance(row, dict):
                        raise ValueError('需要JSON对象')
                except ValueError:
                    issues.append(f'{path.name}存在未解析行，可能正在写入')
                    continue
                last, count = row, count + 1
                entry = row.get('test', {}).get('prediction_diagnostics', {}).get('primary', {})
                single += int(bool(entry.get('single_class')))
    except OSError:
        issues.append(f'{path.name}读取失败')
    return last, count, single


def build_run_summary(run_dir: Path) -> dict:
    """摘要不是完整LOSO验收；缺文件/正在运行均不猜测成功或失败原因。"""
    folders = [run_dir] if (run_dir / 'fold.json').is_file() else sorted(
        p for p in run_dir.glob('fold[0-9]*') if re.fullmatch(r'fold\d+',p.name)
        and 1<=int(p.name[4:])<=32 and p.is_dir() and not p.is_symlink())
    records = []
    for folder in folders:
        issues = []
        cfg = _read_json(folder / 'fold.json', issues)
        summary = _read_json(folder / 'summary.json', issues)
        last, count, single = _epoch_scan(folder / 'epoch_metrics.jsonl', issues)
        historical, _, _ = _epoch_scan(folder / 'history.jsonl', issues)
        if not last:
            last = historical
        args = cfg.get('arguments', {})
        final = last.get('test', {})
        planned = args.get('epochs')
        ended = bool(summary and planned and last.get('epoch') == planned and
                     historical.get('epoch') == planned and summary.get('last_test') == historical.get('test'))
        records.append(dict(fold=cfg.get('held_out_subject'), dataset=cfg.get('dataset'),
            task=args.get('label'), seed=cfg.get('seed'), source_version=cfg.get('source_version'),
            planned_epochs=planned, last_recorded_epoch=last.get('epoch'), diagnostic_records=count,
            single_class_test_records=single if count else None, smoke=bool(cfg.get('smoke')),
            status='正常收尾' if ended else '未确认完整收尾，需日志/退出码判断',
            best=summary.get('best'), has_eligible_best=summary.get('has_eligible_best'),
            last_test={k: final.get(k) for k in ('acc', 'selection_f1', 'count', 'complete_test')},
            last_train={k: last.get('train', {}).get(k) for k in ('acc', 'selection_f1', 'count')},
            last_train_statistics_type='epoch末训练集评估' if count else '优化过程history统计，非epoch末完整评估',
            prediction=final.get('prediction_diagnostics'), issues=issues))
    return dict(format_version=1, generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        folds=records, observed_fold_count=len(records), formal_loso_verified=False,
        note='本摘要不加载权重、不重新选模；正式统计需CalculateResult.py --require_complete。旧Phase4未记录训练源码SHA时不能用上传时SHA代替。',
        selection_bias='held-out测试集参与checkpoint选择，存在测试集选模偏差')


def report_text(summary: dict) -> str:
    lines = ['NeuroFDAN运行摘要（非完整LOSO验收）', summary['note'], summary['selection_bias']]
    for row in summary['folds']:
        lines.append(f'fold={row["fold"]} {row["dataset"]}/{row["task"]} seed={row["seed"]} '
                     f'epoch={row["last_recorded_epoch"]}/{row["planned_epochs"]} '
                     f'状态={row["status"]} smoke={row["smoke"]} best={row["best"]}')
        lines.append(f'最后测试={row["last_test"]}；单类测试记录数={row["single_class_test_records"]}')
        if row['issues']:
            lines.append('注意：' + '；'.join(row['issues']))
    return '\n'.join(lines) + '\n'


def write_run_reports(run_dir: Path) -> None:
    """只写两个摘要文件，采用临时文件替换，不覆盖原训练产物。"""
    if not run_dir.is_dir():
        return
    summary = build_run_summary(run_dir)
    for name, content in [('run_summary.json', json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)),
                          ('run_report.txt', report_text(summary))]:
        temporary = run_dir / (name + '.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(run_dir / name)
