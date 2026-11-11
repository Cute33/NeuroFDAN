"""只用标准库核验并整理两份固定诊断快照；不导入PyTorch、不读取权重或数据。"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


SNAPSHOTS = (
    ('DEAP', '8a368c15ad5e1e9fd5f909a99547dd90c3d11f98',
     'runs/2026-11-11_140435_unknown_unknown_run_4d0fa629'),
    ('HCI', 'cfcdbdff5696ed0061a5ef410dca5c1c6820da70',
     'runs/2026-11-11_140543_unknown_unknown_run_8e5b753f'),
)


def read_snapshot(repository: Path, dataset: str, sha: str, folder: str) -> tuple:
    """读取不可变Git对象并核对manifest；读取内容不作为代码执行。"""
    def read(name: str) -> bytes:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('快照文件路径不合法')
        return subprocess.check_output(['git', '-C', str(repository), 'show', f'{sha}:{folder}/{name}'])

    manifest = json.loads(read('manifest.json'))
    verified, logs = [], []
    for item in manifest['uploaded']:
        data = read(item['file'])
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'{dataset}载荷完整性失败：{item["file"]}')
        verified.append(item['file'])
        if item['file'].endswith('.log'):
            logs.append(data.decode('utf-8'))
    if 'probe_summary.json' not in verified:
        raise ValueError('快照缺少核心探针结果')
    result = json.loads(read('probe_summary.json'))
    if result['dataset'] != dataset or result['checkpoint_epoch'] != 20:
        raise ValueError('数据集或checkpoint轮次不匹配')
    if result['source_version']['commit'] != 'c6fbb61f41bb956a64ae2a8d8a4152cbfa420e49':
        raise ValueError('探针源码版本不符')
    if not all(result[key] for key in ('model_state_unchanged', 'checkpoint_file_unchanged', 'parameter_grad_fields_unchanged')):
        raise ValueError('原探针未通过状态保护检查')
    if result['optimizer_created'] or result['training_executed']:
        raise ValueError('快照不是预期只读操作')
    groups = {key: {tuple(x) for x in value} for key, value in result['sources'].items()}
    if any(row[0] == 1 for group in groups.values() for row in group):
        raise ValueError('探针来源包含测试被试1')
    keys = list(groups)
    if any(groups[a] & groups[b] for i, a in enumerate(keys) for b in keys[i+1:]):
        raise ValueError('探针来源组不满足排除自身约束')
    log = '\n'.join(logs)
    validation = dict(sha=sha, folder=folder, verified_files=verified,
        missing=manifest['missing'], skipped=manifest['skipped'],
        math_test_log_reports_5_passed=('Ran 5 tests in' in log and '\nOK\n' in log),
        sources_exclude_heldout=True, source_groups_disjoint=True,
        reference_draws=len(result['sources']['reference']), unique_reference_draws=len(groups['reference']))
    return result, validation


def extract(result: dict) -> dict:
    """提取可比较摘要；RMS跨输入变化不是全元素采样std。"""
    p = result['posterior']
    posterior = {}
    for name in ('cns', 'ans_with_common_cns_noise', 'ans_vary_xa_hold_mc'):
        item = p[name]
        posterior[name] = dict(rms_between_input_mu_std=math.sqrt(item['var_X_mu']['mean']),
            rms_between_input_sigma_std=math.sqrt(item['var_X_sigma']['mean']),
            mean_posterior_std=item['posterior_std']['mean'],
            mean_conditional_gaussian_kl=item['gaussian_conditional_kl_float64']['mean'])
    moments = p['ans_marginal_over_mc']
    posterior['ans_marginal'] = dict(
        rms_between_input_mean_std=math.sqrt(moments['var_X_E_Ma']['mean']),
        rms_between_input_total_std_variation=math.sqrt(moments['var_X_total_std']['mean']),
        mean_conditional_variance=moments['E_conditional_variance']['mean'],
        mean_variance_of_conditional_mean_over_mc=moments['Var_Mc_conditional_mean']['mean'])
    d, n = result['density_support'], result['joint_nearest_neighbor']
    return dict(selection=result['selection'], posterior=posterior,
        support=dict(own_low_density_fraction=d['own_below_own_5pct_fraction'],
            cross_low_density_fraction=d['cross_below_reference_5pct_fraction'],
            own_joint_distance_mean=n['query_factual_eval']['mean'],
            cross_joint_distance_mean=n['query_cross_eval']['mean'],
            cross_above_eval_p95_fraction=n['cross_above_eval_factual_p95_fraction']),
        head_standardized_gradient_mean={mode: {block: values['standardized_gradient_l2']['mean']
            for block, values in blocks.items()} for mode, blocks in result['local_head_sensitivity'].items()},
        paired_probability=result['paired_probability'],
        note='仅16个训练查询窗口；无真实中介/干预真值。概率std含MC噪声，不是纯输入信号或正式ACC。')


def main() -> None:
    parser = argparse.ArgumentParser(description='整理已上传的支持审计结果，不运行模型', add_help=False)
    parser.add_argument('-h', '--help', action='help', help='显示中文帮助')
    parser.add_argument('--repository', type=Path, default=Path('/home/fss/NeuroFDAN'), help='含已fetch诊断Git对象的仓库')
    parser.add_argument('--output_dir', type=Path, required=True, help='全新结果目录，不能覆盖历史整理')
    args = parser.parse_args()
    target = args.output_dir.resolve()
    if (target/'probe_summary.json').exists():
        raise FileExistsError('已有整理结果，停止，不覆盖')
    records = {}
    for dataset, sha, folder in SNAPSHOTS:
        original, validation = read_snapshot(args.repository, dataset, sha, folder)
        records[dataset] = dict(validation=validation, summary=extract(original))
        print(f'{dataset}：载荷完整，原探针状态保护通过；数学测试日志通过={validation["math_test_log_reports_5_passed"]}')
        print(json.dumps(records[dataset]['summary'], ensure_ascii=False, indent=2))
    result = dict(operation='既有支持诊断快照的标准库核验与汇总', records=records,
        model_loaded=False, training_executed=False, torch_imported=False,
        note='本操作未重复原数学测试或探针；测试状态来自服务器已上传日志。')
    target.mkdir(parents=True, exist_ok=True)
    with (target/'probe_summary.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(f'汇总已保存：{target/"probe_summary.json"}')


if __name__ == '__main__':
    main()
