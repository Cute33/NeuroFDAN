"""汇总本次已完成的实验；不训练模型、不填造缺失结果。"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


def read(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def aggregate(rows: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in groups.items():
        row = dict(zip(keys, key))
        row['count'] = len(values)
        for metric in metrics:
            numbers = np.array([float(v[metric]) for v in values])
            row[metric + '_mean'] = numbers.mean()
            row[metric + '_std'] = numbers.std(ddof=1) if len(numbers) > 1 else 0.
        output.append(row)
    return output


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'没有可汇总结果：{path}')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(root: Path, grouped: list[dict], convergence: list[dict],
                 references: list[dict], max_n: int) -> None:
    """生成独立科学图；英文仅保留变量名、模型标识和指标缩写。"""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for name in ('full', 'none', 'observational'):
        rows = sorted([r for r in grouped if int(r['n_train']) == max_n and r['model'] == name], key=lambda r: float(r['gamma']))
        axes[0].errorbar([float(r['gamma']) for r in rows], [r['mae_mean'] for r in rows],
                         yerr=[r['mae_std'] for r in rows], marker='o', capsize=3, label=name)
    axes[0].set(xlabel=r'$\gamma$', ylabel='MAE')
    axes[0].legend()
    for gamma in sorted({r['gamma'] for r in convergence}):
        rows = sorted([r for r in convergence if r['gamma'] == gamma], key=lambda r: int(r['S']))
        axes[1].errorbar([int(r['S']) for r in rows], [r['mae_to_direct_mean'] for r in rows],
                         yerr=[r['mae_to_direct_std'] for r in rows], marker='o', capsize=3, label=gamma)
    axes[1].set(xscale='log', xlabel='S = K', ylabel='MAE')
    axes[1].legend(title=r'$\gamma$')
    rows = sorted([r for r in references if float(r['gamma']) == 1 and int(r['n_train']) == max_n and r['no_cns2ans'] == 'False'], key=lambda r: int(r['ref_dim']))
    axes[2].bar([r['ref_dim'] for r in rows], [r['mae_mean'] for r in rows],
                 yerr=[r['mae_std'] for r in rows], capsize=3)
    axes[2].set(xlabel='ref_dim (0 = 24)', ylabel='MAE')
    figure.savefig(root / 'phase1_summary.png', dpi=180)
    figure.savefig(root / 'phase1_summary.pdf')
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description='将已完成的第一阶段 CSV 汇总成统计表与中文报告', add_help=False)
    parser.add_argument('-h', '--help', action='help', help='显示参数帮助并退出')
    parser.add_argument('--input', default='results/phase1_scm', help='已经生成 completed.json 的实验目录')
    parser.add_argument('--report', default='docs/phase1_implementation_report.md', help='追加实际结果的中文报告')
    parser.add_argument('--plots', action='store_true', help='额外生成需 matplotlib 的独立 PNG/PDF 科学图')
    args = parser.parse_args()
    root = Path(args.input)
    if not json.loads((root / 'completed.json').read_text())['completed']:
        raise RuntimeError('实验未完成，拒绝生成成功报告')
    learned = read(root / 'learned_vs_oracle.csv')
    metrics = ['mae', 'probability_mse', 'brier', 'nll', 'calibration_10bins', 'mae_to_oracle']
    standard = [r for r in learned if r['ref_dim'] == '0' and r['no_cns2ans'] == 'False'
                and (r['model'] != 'full' or (r['S'] == '16' and r['K'] == '64' and r['N_ref_requested'] == '2048'))]
    grouped = aggregate(standard, ['gamma', 'n_train', 'model'], metrics)
    write(root / 'metrics_summary.csv', grouped)
    sampling = [r for r in learned if r['model'] == 'full' and r['ref_dim'] == '0' and r['no_cns2ans'] == 'False']
    sample_stats = aggregate(sampling, ['gamma', 'n_train', 'S', 'K', 'N_ref_requested', 'N_ref_used'], metrics)
    write(root / 'sampling_sensitivity.csv', sample_stats)
    references = read(root / 'reference_sensitivity.csv')
    reference_stats = aggregate(references, ['gamma', 'n_train', 'ref_dim', 'no_cns2ans'], metrics)
    write(root / 'reference_sensitivity_summary.csv', reference_stats)
    oracle = aggregate(read(root / 'oracle_vs_direct.csv'), ['gamma'], ['absolute_error', 'relative_error', 'direct_se', 'oracle_se_approx'])
    write(root / 'oracle_summary.csv', oracle)
    convergence = aggregate(read(root / 'mc_convergence.csv'), ['gamma', 'S', 'K'], ['mae_to_direct'])
    write(root / 'mc_convergence_summary.csv', convergence)
    ci = read(root / 'conditional_independence.csv')
    config = json.loads((root / 'config.json').read_text())
    max_n = max(config['experiment']['train_sizes'])
    if args.plots:
        plot_results(root, grouped, convergence, reference_stats, max_n)
    text = ['\n### 直接干预与 Oracle\n', '| γ | 平均绝对误差 | 平均相对误差 |', '|---|---:|---:|']
    for row in oracle:
        text.append(f"| {row['gamma']} | {row['absolute_error_mean']:.6f} | {row['relative_error_mean']:.6f} |")
    text += ['\n### 学习估计器\n', f'以下为 {max_n} 个训练样本、各配置五个种子的实际干预 MAE（均值±样本标准差）；如修改种子配置，以 CSV 的 count 为准。',
             '\n| γ | full | none | 普通观察预测器 |', '|---|---:|---:|---:|']
    for gamma in config['experiment']['gammas']:
        entries = []
        for name in ('full', 'none', 'observational'):
            row = next(r for r in grouped if float(r['gamma']) == gamma and int(r['n_train']) == max_n and r['model'] == name)
            entries.append(f"{row['mae_mean']:.5f} ± {row['mae_std']:.5f}")
        text.append('| ' + str(gamma) + ' | ' + ' | '.join(entries) + ' |')
    text += ['\nfull 的有限样本学习误差不等于 oracle 数值积分误差。none 按约定使用一次事实路径，因此同时包含较大的单次中介抽样波动，不能把全部误差差异都归于 K 积分。Brier 是二分类单概率的期望平方误差，NLL 为真实干预概率的期望交叉熵，均不是用一次 Bernoulli 结果冒充概率。校准仅在有限干预点上诊断。',
             '\n### 参考表示和 CNS 条件路径敏感性\n', '| 参考维度 | 切断 CNS→ANS | MAE |', '|---:|---|---:|']
    for row in reference_stats:
        if float(row['gamma']) == 1 and int(row['n_train']) == max_n:
            text.append(f"| {row['ref_dim']} | {row['no_cns2ans']} | {row['mae_mean']:.5f} ± {row['mae_std']:.5f} |")
    values = [r for r in reference_stats if float(r['gamma']) == 1 and int(r['n_train']) == max_n and r['no_cns2ans'] == 'False']
    text.append('\n相对完整24维表示的配对 MAE 差值（种子一一配对；负数表示该配置误差降低）：')
    for row in values:
        dim = row['ref_dim']
        if dim == '0':
            continue
        pairs = []
        for seed in config['experiment']['seeds']:
            a = next(r for r in references if float(r['gamma']) == 1 and int(r['n_train']) == max_n and int(r['seed']) == seed and r['ref_dim'] == '0' and r['no_cns2ans'] == 'False')
            b = next(r for r in references if float(r['gamma']) == 1 and int(r['n_train']) == max_n and int(r['seed']) == seed and r['ref_dim'] == dim and r['no_cns2ans'] == 'False')
            pairs.append(float(b['mae']) - float(a['mae']))
        text.append(f"- ref_dim={dim}：{np.mean(pairs):+.6f}；种子间标准差 {np.std(pairs, ddof=1) if len(pairs)>1 else 0:.6f}。")
    text.append('\n这些差值反映本次有限样本和优化配置中的表示敏感性，不足以证明充分统计量或精确识别。64/128不是原始24维输入的有损压缩；8维才是压缩对照。')
    text += ['\n### S、K 与参考库规模\n', '以下为 gamma=1、训练样本2048、完整参考表示的五种子结果；其他配置见 sampling_sensitivity.csv。',
             '\n| S | K | 实际参考库规模 | MAE |', '|---:|---:|---:|---:|']
    for row in sample_stats:
        if float(row['gamma']) == 1 and int(row['n_train']) == max_n:
            text.append(f"| {row['S']} | {row['K']} | {row['N_ref_used']} | {row['mae_mean']:.5f} ± {row['mae_std']:.5f} |")
    text += ['\nOracle 收敛扫描（五随机流、四个固定干预点的平均绝对误差）：',
             '\n| gamma | S=K=32 | S=K=2048 |', '|---:|---:|---:|']
    for gamma in config['experiment']['gammas']:
        low = next(r for r in convergence if float(r['gamma']) == gamma and r['S'] == '32')
        high = next(r for r in convergence if float(r['gamma']) == gamma and r['S'] == '2048')
        text.append(f"| {gamma} | {low['mae_to_direct_mean']:.6f} | {high['mae_to_direct_mean']:.6f} |")
    text.append('\n增加 Monte Carlo 样本降低的是数值波动，不保证消除拟合偏差；学习误差不必随 S/K/N_ref 单调下降。完整收敛表还包括128和512档。')
    text += ['\n### 条件独立性诊断\n', f"已知条件分布残差的8个矩，在全部配置上的最大 |z| 为 {max(float(r['oracle_residual_max_abs_z_8moments']) for r in ci):.4f}。",
             f"无约束结果模型在固定 M_a,X、条件抽样 M_c 后的平均预测绝对变化为 {np.mean([float(r['unrestricted_conditional_mc_probability_change']) for r in ci]):.6f}。",
             '默认受约束结果模型没有直接 CNS 中介输入；无约束模型的残余变化是学习伪依赖的描述性证据，不能解释为 M_c→Y 的因果边。有限矩未拒绝和有限链诊断均不等于普遍证明条件独立性。',
             ]
    diagnostic_path = root / 'learning_diagnostics.csv'
    if diagnostic_path.exists():
        diagnostic = read(diagnostic_path)
        text += ['\n### 独立留出拟合诊断\n', 'gamma=1、2048训练样本、4096独立留出样本，重现五种子拟合；不使用留出样本更新模型。',
                 '\n| 切断 CNS→ANS | CNS均值RMSE | ANS条件均值RMSE | 结果条件概率MAE |', '|---|---:|---:|---:|']
        for disabled in ('False', 'True'):
            rows = [r for r in diagnostic if r['no_cns2ans'] == disabled]
            values = [np.mean([float(r[k]) for r in rows]) for k in ('cns_latent_mu_rmse', 'ans_teacher_conditional_mu_rmse', 'outcome_conditional_mae')]
            text.append(f'| {disabled} | {values[0]:.5f} | {values[1]:.5f} | {values[2]:.5f} |')
        differences = []
        for row in diagnostic:
            match = next(r for r in references if float(r['gamma']) == 1 and int(r['n_train']) == 2048 and r['ref_dim'] == '0' and r['seed'] == row['seed'] and r['no_cns2ans'] == row['no_cns2ans'])
            differences.append(abs(float(match['mae']) - float(row['reproduced_do_mae'])))
        text.append(f'\n独立重现与主实验 MAE 的最大差为 {max(differences):.12f}。完整条件密度超额NLL见 learning_diagnostics.csv。')
        text.append('保留 CNS 路径时，真实 ANS 条件均值和条件密度拟合更好，但最终干预 MAE 没有因此更低。结果条件模型仍有明显拟合误差，分层中介也有误差；这些误差可能相互抵消，加上仅16个评估点和有限采样，不能把本次消融差异解释为机制不存在，亦不能宣称已展示该路径的性能收益。')
    text.append('\n### 运行状态与复现\n')
    log = (root / 'test_log.txt').read_text()
    text.append('单元测试完整原始输出见 `results/phase1_scm/test_log.txt`。')
    text.append('```text\n' + '\n'.join(log.strip().splitlines()[-5:]) + '\n```')
    text.append('CPU 三层实验实际完成；CUDA 测试按实际可用性跳过或运行，不把跳过记作通过。命令行帮助与编译另见 validation_log.txt。')
    text.append('原始指标、每种子输出、数值收敛、支持距离和阶段耗时分别保存为 CSV。只有数值计算和有限样本合成验证完成；真实数据因果假设、DEAP/HCI分类表现和生理机制识别未获验证。')
    report = Path(args.report)
    original = report.read_text()
    marker = '\n<!-- 实际结果自动汇总 -->'
    report.write_text(original.split(marker)[0] + marker + '\n' + '\n'.join(text) + '\n')
    files = ['frontdoor_scm.py', 'synthetic_scm.py', 'synthetic_frontdoor_experiment.py',
             'tests/test_frontdoor_scm.py', 'phase1_results_summary.py', 'phase1_learning_diagnostics.py']
    (root / 'source_hashes.json').write_text(json.dumps({name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}, indent=2))
    print('实际结果统计表与中文技术报告已汇总。')


if __name__ == '__main__':
    main()
