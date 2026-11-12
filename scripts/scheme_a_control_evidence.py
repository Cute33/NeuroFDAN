"""KL单因素拟合对照的只读协议守卫与结果比较；不运行模型。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from upload_run_diagnostics import read_snapshot, reject_constant, sanitize

BASELINE_COMMIT = '30152089b943ccf904d205810c72fb20161a483a'
LOCATION_KEYS = {'output_dir', 'run_name'}


def read_json(path):
    return json.loads(read_snapshot(Path(path), 16 * 1024**2).decode('utf-8'), parse_constant=reject_constant)


def summary_sha256(path):
    return hashlib.sha256(read_snapshot(Path(path), 16 * 1024**2)).hexdigest()


def check_arguments(baseline, arguments, controls):
    if (baseline.get('source_version', {}).get('commit') != BASELINE_COMMIT or
            not baseline.get('completed') or baseline.get('passed') or baseline.get('actual_steps') != 300):
        raise ValueError('需要已核验的30152089方案A、完成300步且未通过的原始摘要')
    old = baseline['arguments']
    if (old['dataset'], old['label'], old['held_out_subjects'], old['seed'],
            old['classification_objective'], old['beta'], old['fire_weight']) != (
            'DEAP', 'arousal', [2], 42, 'integrated_observational', .001, .028):
        raise ValueError('基线身份/损失配置不符')
    changed = {key for key in set(old) | set(arguments) if old.get(key) != arguments.get(key)}
    if changed - LOCATION_KEYS != {'beta'} or arguments['beta'] != 0.:
        raise ValueError('只允许beta由0.001改为0；其他训练参数须逐项相同')
    for key in ('fit_max_steps', 'fit_check_every'):
        if baseline['fit_controls'][key] != controls[key]:
            raise ValueError('拟合预算/检查间隔须与基线相同')


def check_sources(baseline, result):
    for key in ('selected_source_ids', 'reference_source_ids', 'true_counts', 'full_epoch_steps', 'gate'):
        if baseline[key] != result[key]:
            raise ValueError('对照来源/评价门槛与基线不一致：' + key)
    a, b = baseline['model'], result['model']
    changed = {key for key in set(a) | set(b) if a.get(key) != b.get(key)}
    if changed != {'beta'} or b['beta'] != 0.:
        raise ValueError('模型配置只允许改变beta，不允许改结构、采样或参考维度')


def check_initial_nll(baseline, value):
    expected = baseline['records'][0]['optimization_nll']
    if not math.isfinite(value) or not math.isclose(value, expected, abs_tol=2e-6, rel_tol=0.):
        raise ValueError('首步主NLL与基线不同，先检查输入/初始化/随机流，不继续训练')


def compare(baseline, result):
    check_arguments(baseline, result['arguments'], result['fit_controls'])
    check_sources(baseline, result)
    check_initial_nll(baseline, result['records'][0]['optimization_nll'])
    def last(probe):
        record = probe['records'][-1]
        return dict(step=record['step'], actual_steps=probe.get('actual_steps'),
            passed=probe['passed'], completed=probe['completed'],
            draws=[{key: d[key] for key in ('balanced_accuracy', 'macro_f1',
                'integrated_observational_nll', 'confusion_matrix', 'probability_std', 'posterior')}
                for d in record['draws']])
    return dict(format_version='scheme_a_kl_control_v1', role='training_subset_comparison_only',
        protocol_checks_passed=True, changed_training_factor=dict(beta=[.001, 0.]),
        true_counts=baseline['true_counts'], constant_prior_nll=baseline['constant_prior_nll'],
        baseline=last(baseline), kl_off=last(result),
        note='只比较固定自然训练子集拟合；KL关闭是诊断消融，不是最终方法或因果识别证据。',
        classification_gate_unchanged=True, automatic_fold_training=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--output', required=True, help='新的只读比较报告，拒绝覆盖')
    args = parser.parse_args()
    old, new = read_json(args.baseline), read_json(args.result)
    report = compare(old, new)
    report['baseline_summary_sha256'] = summary_sha256(args.baseline)
    encoded = json.dumps(sanitize(report), ensure_ascii=False, indent=2, allow_nan=False)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        stream.write(encoded + '\n')
    print(encoded, flush=True)


if __name__ == '__main__':
    main()
