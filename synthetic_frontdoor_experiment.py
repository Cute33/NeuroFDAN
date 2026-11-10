"""第一阶段 CPU 合成实验；已知条件机制与学习条件模型分别计时和保存。"""
import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator, ReferenceBank
from synthetic_scm import SCMConfig, SyntheticSCM, probability_metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    """每次写入完整当前结果，长实验中断时保留已经完成的配置。"""
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def tensor_data(data: dict) -> dict[str, torch.Tensor]:
    """float32 中 tanh 极端尾部作显式数值截断；次数另写实验配置。"""
    result = {}
    for key in ('x_c', 'x_a', 'm_c', 'm_a'):
        tensor = torch.tensor(data[key], dtype=torch.float32)
        result[key] = tensor.clamp(-1 + 1e-6, 1 - 1e-6) if key.startswith('m_') else tensor
    result['y'] = torch.tensor(data['y'], dtype=torch.long)
    return result


def make_bank(data: dict, size: int, seed: int) -> ReferenceBank:
    """合成来源均标为训练被试 0，trial 取样本唯一编号；测试来源另设。"""
    n = len(data['x_c'])
    sources = torch.stack((torch.zeros(n, dtype=torch.long), torch.arange(n), torch.zeros(n, dtype=torch.long)), 1)
    return ReferenceBank(data['x_c'], data['x_a'], sources, forbidden_subjects=(1, 2), max_size=size, seed=seed)


def fit(scm: SyntheticSCM, gamma: float, n: int, seed: int, ref_dim: int,
        epochs: int, no_cns2ans: bool = False) -> tuple:
    """仅使用训练样本及其合成中介真值；不使用真值干预概率训练。"""
    torch.manual_seed(seed)
    raw = scm.sample(n, gamma, seed + 1000)
    data = tensor_data(raw)
    c = scm.config
    cfg = FrontDoorConfig(ref_dim=ref_dim, mediator_dim=c.d_mc, mediator_family='tanh_normal',
                          no_cns2ans=no_cns2ans)
    if c.d_mc != c.d_ma:
        raise ValueError('本实验要求 CNS 与 ANS 中介维度相同')
    model = FrontDoorEstimator(c.d_xc, c.d_xa, cfg)
    observational = nn.Sequential(nn.Linear(c.d_xc + c.d_xa, 128), nn.SiLU(), nn.Linear(128, 2))
    unrestricted = nn.Sequential(nn.Linear(c.d_mc + c.d_ma + c.d_xc + c.d_xa, 128), nn.SiLU(), nn.Linear(128, 2))
    optimizer = torch.optim.Adam(list(model.parameters()) + list(observational.parameters()) + list(unrestricted.parameters()), lr=1e-3)
    start = time.perf_counter()
    for epoch in range(epochs):
        order = torch.randperm(n)
        for indices in order.split(128):
            xc, xa, mc, ma, y = (data[key][indices] for key in ('x_c', 'x_a', 'm_c', 'm_a', 'y'))
            mu_c, logvar_c = model.cns.distribution(xc)
            # ANS 条件密度用训练集中观测到的真实 CNS 中介作条件，不用测试中介。
            ans = model.ans(xa, mc[:, None])
            nll_c = model.cns.nll(mc, mu_c, logvar_c) / c.d_mc
            nll_a = model.ans.nll(ma, ans['mu_a'][:, 0], ans['logvar_a'][:, 0]) / c.d_ma
            reference = model.reference_encoder(xc, xa)
            outcome_loss = F.cross_entropy(model.outcome(ma, reference), y)
            joint = torch.cat((xc, xa), -1)
            loss = nll_c + nll_a + outcome_loss
            loss = loss + F.cross_entropy(observational(joint), y)
            loss = loss + F.cross_entropy(unrestricted(torch.cat((mc, ma, joint), -1)), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(observational.parameters()) + list(unrestricted.parameters()), 5.)
            optimizer.step()
    model.eval(); observational.eval(); unrestricted.eval()
    elapsed = time.perf_counter() - start
    clipping = int(sum(np.sum(np.abs(raw[key]) > 1 - 1e-6) for key in ('m_c', 'm_a')))
    return model, observational, unrestricted, data, elapsed, clipping


def conditional_mc_draw(scm: SyntheticSCM, data: dict, seed: int, burnin: int = 40) -> np.ndarray:
    """椭圆切片采样 M_c|M_a,X；固定 M_a,X，不进行全局置乱。"""
    rng = np.random.default_rng(seed)
    draws = []
    for xc, xa, ma in zip(data['x_c'], data['x_a'], data['m_a']):
        mean = xc @ scm.B_c.T
        target = np.arctanh(np.clip(ma, -1 + 1e-12, 1 - 1e-12)) - xa @ scm.B_a.T
        def log_likelihood(z):
            error = target - np.tanh(z) @ scm.C_ca.T
            return -.5 * (error @ error)
        z = mean + rng.normal(size=scm.config.d_mc)
        for _ in range(burnin):
            direction = rng.normal(size=z.shape)
            level = log_likelihood(z) + np.log(rng.uniform())
            angle = rng.uniform(0, 2 * np.pi)
            left, right = angle - 2 * np.pi, angle
            for attempt in range(1000):
                proposal = mean + (z - mean) * np.cos(angle) + direction * np.sin(angle)
                if log_likelihood(proposal) > level:
                    z = proposal
                    break
                if angle < 0:
                    left = angle
                else:
                    right = angle
                angle = rng.uniform(left, right)
            else:
                raise RuntimeError('条件中介采样未收敛，不能退化成全局打乱')
        draws.append(np.tanh(z))
    return np.array(draws)


def diagnostics(scm: SyntheticSCM, gamma: float, seed: int, model, unrestricted) -> dict:
    """已知条件概率残差矩与无约束学习对照分别报告，不宣称证明独立性。"""
    raw = scm.sample(4096, gamma, 80000 + seed)
    true_conditional = scm.conditional_outcome(raw['m_a'], raw['x_c'], raw['x_a'], gamma)
    score = (raw['y'] - true_conditional)[:, None] * raw['m_c'][:, :8]
    z = score.mean(0) / (score.std(0, ddof=1) / np.sqrt(len(score)))
    data = tensor_data(raw)
    with torch.no_grad():
        reference = model.reference_encoder(data['x_c'], data['x_a'])
        constrained = model.outcome(data['m_a'], reference)
        joint = torch.cat((data['m_a'], data['x_c'], data['x_a']), -1)
        extended = unrestricted(torch.cat((data['m_c'], joint), -1))
        subset = {k: raw[k][:128] for k in ('x_c', 'x_a', 'm_a')}
        draw1 = torch.tensor(conditional_mc_draw(scm, subset, seed + 90000), dtype=torch.float32)
        draw2 = torch.tensor(conditional_mc_draw(scm, subset, seed + 91000), dtype=torch.float32)
        first = unrestricted(torch.cat((draw1, joint[:128]), -1)).softmax(-1)[:, 1]
        second = unrestricted(torch.cat((draw2, joint[:128]), -1)).softmax(-1)[:, 1]
        # 两次条件 M_c 抽样均不能传入受约束结果模型；实际计算其相同输入预测。
        fixed_first = model.outcome(data['m_a'][:128], reference[:128]).softmax(-1)
        fixed_second = model.outcome(data['m_a'][:128], reference[:128]).softmax(-1)
    return {'gamma': gamma, 'seed': seed, 'oracle_residual_max_abs_z_8moments': float(abs(z).max()),
            'constrained_test_ce': F.cross_entropy(constrained, data['y']).item(),
            'unrestricted_test_ce': F.cross_entropy(extended, data['y']).item(),
            'unrestricted_conditional_mc_probability_change': (first - second).abs().mean().item(),
            'constrained_conditional_mc_probability_change': (fixed_first - fixed_second).abs().mean().item(),
            'conditional_sampler_burnin': 40, 'conditional_sampler_points': 128}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='独立合成 SCM 的三层前门验证，不调用真实数据训练器', add_help=False)
    p.add_argument('-h', '--help', action='help', help='显示参数帮助并退出')
    p.add_argument('--output', default='results/phase1_scm', help='结果目录')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='学习与数值诊断随机种子')
    p.add_argument('--gammas', type=float, nargs='+', default=[0., .5, 1., 2.], help='混杂强度扫描')
    p.add_argument('--train-sizes', type=int, nargs='+', default=[512, 2048], help='合成训练样本数扫描')
    p.add_argument('--epochs', type=int, default=20, help='每个学习模型拟合轮数')
    p.add_argument('--points', type=int, default=16, help='配对干预评估点数量')
    p.add_argument('--direct-n', type=int, default=65536, help='每个干预点直接积分样本数')
    p.add_argument('--oracle-s', type=int, default=2048, help='主 oracle 的中介积分次数')
    p.add_argument('--oracle-k', type=int, default=1024, help='主 oracle 的联合参考积分次数')
    p.add_argument('--reference-dims', type=int, nargs='+', default=[0, 8, 64, 128], help='参考维度敏感性；零保留全部 24 维，64 在此是扩维对照')
    p.add_argument('--threads', type=int, default=1, help='CPU 线程数，避免小矩阵过度并行')
    return p


def main() -> None:
    args = parser().parse_args()
    if min(args.epochs, args.points, args.direct_n, args.oracle_s, args.oracle_k, args.threads, *args.train_sizes) < 1:
        raise ValueError('实验规模和线程数必须为正')
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scm = SyntheticSCM()
    scm.save(out / 'scm_parameters.npz')
    cfg = {'experiment': vars(args), 'scm': asdict(scm.config), 'model_defaults': asdict(FrontDoorConfig()),
           'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available(),
           'learned_mediator_family': 'tanh_normal', 'learned_supervision': '仅合成训练集真实中介条件似然及事实结果 CE；无干预 oracle 监督',
           'reference_sampling': '训练经验配对样本，无标签选择；有放回抽取 K；库不足时使用全部',
           'train_batch_size': 128, 'learning_rate': .001, 'conditional_model_gh_nodes': 32,
           'truth_se_note': '直接积分独立样本标准误；前门为两向一阶近似标准误，不保证覆盖率',
           'evaluation_point_source': 'gamma=0 的独立输入分布，所有 gamma 使用相同点，避免只测混杂变化后的极端点'}
    (out / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    point = scm.sample(args.points, 0, 78901)
    points = tensor_data(point)
    np.savez(out / 'intervention_points.npz', x_c=point['x_c'], x_a=point['x_a'])
    oracle_rows, convergence_rows, learned_rows, diagnostic_rows, overlap_rows = [], [], [], [], []
    reference_rows = []
    for gamma in args.gammas:
        start = time.perf_counter()
        direct, dse = scm.direct_intervention(point['x_c'], point['x_a'], gamma, args.direct_n, 43210)
        direct_time = time.perf_counter() - start
        start = time.perf_counter()
        oracle, ose = scm.frontdoor_integral(point['x_c'], point['x_a'], gamma, args.oracle_s, args.oracle_k, 54321)
        oracle_time = time.perf_counter() - start
        for i in range(args.points):
            oracle_rows.append({'gamma': gamma, 'point': i, 'direct': direct[i], 'direct_se': dse[i],
                                'oracle_fd': oracle[i], 'oracle_se_approx': ose[i],
                                'absolute_error': abs(direct[i] - oracle[i]),
                                'relative_error': abs(direct[i] - oracle[i]) / max(direct[i], 1e-8),
                                'direct_seconds_all_points': direct_time, 'oracle_seconds_all_points': oracle_time})
        write_csv(out / 'oracle_vs_direct.csv', oracle_rows)
        print(f'混杂强度 {gamma}：直接与 oracle 平均绝对差 {abs(direct-oracle).mean():.6f}', flush=True)
        for size in (32, 128, 512, 2048):
            for seed in args.seeds:
                start = time.perf_counter()
                estimate, _ = scm.frontdoor_integral(point['x_c'][:4], point['x_a'][:4], gamma, size, size, 60000 + seed)
                convergence_rows.append({'gamma': gamma, 'seed': seed, 'S': size, 'K': size,
                                         'mae_to_direct': abs(estimate - direct[:4]).mean(),
                                         'seconds': time.perf_counter() - start})
        write_csv(out / 'mc_convergence.csv', convergence_rows)
        for n in args.train_sizes:
            for seed in args.seeds:
                # 参考维度、CNS 消融仅在 gamma=1、最大训练规模扫描，其他配置保持不变。
                variants = [(0, False)]
                if gamma == 1. and n == max(args.train_sizes):
                    variants += [(dim, False) for dim in args.reference_dims if dim != 0]
                    variants += [(0, True)]
                for ref_dim, no_cns in variants:
                    model, baseline, unrestricted, train, fit_time, clips = fit(scm, gamma, n, seed, ref_dim, args.epochs, no_cns)
                    bank = make_bank(train, 2048, seed)
                    # 每次使用相同随机流，便于估计器/表示维度的配对比较。
                    settings = [(16, 64, 2048)]
                    if ref_dim == 0 and not no_cns and n == max(args.train_sizes):
                        settings += [(4, 16, 2048), (16, 16, 2048), (16, 64, 128), (16, 64, 512)]
                    for S, K, capacity in settings:
                        bank = make_bank(train, capacity, seed)
                        torch.manual_seed(70000 + seed)
                        start = time.perf_counter()
                        with torch.no_grad():
                            estimate = model(points['x_c'], points['x_a'], reference_bank=bank, S=S, K=K)
                        elapsed = time.perf_counter() - start
                        q = estimate['probs_do'][:, 1].numpy()
                        row = {'gamma': gamma, 'seed': seed, 'n_train': n, 'ref_dim': ref_dim,
                               'no_cns2ans': no_cns, 'S': S, 'K': K, 'N_ref_requested': capacity,
                               'N_ref_used': bank.used_count, 'reference_shortfall': bank.shortfall,
                               'model': 'full', 'fit_seconds': fit_time, 'estimate_seconds': elapsed,
                               'mae_to_oracle': float(abs(q - oracle).mean()), 'numerical_clipped_mediator_values': clips,
                               **probability_metrics(q, direct)}
                        learned_rows.append(row)
                        if S == 16 and K == 64 and capacity == 2048:
                            reference_rows.append(row.copy())
                        # 支持诊断：交叉组合最近邻距离相对事实组合，非 positivity 证明。
                        if ref_dim == 0 and not no_cns and capacity == 2048 and S == 16 and K == 64:
                            factual = torch.cat((train['m_a'], train['x_c'], train['x_a']), -1)
                            scale = factual.std(0).clamp_min(.1)
                            cross = torch.cat((estimate['m_a'][:, 0], bank.x_c[:args.points], bank.x_a[:args.points]), -1)
                            near = torch.cdist(cross / scale, factual / scale).min(1).values
                            held = tensor_data(scm.sample(args.points, gamma, 33333))
                            factual_held = torch.cat((held['m_a'], held['x_c'], held['x_a']), -1)
                            normal = torch.cdist(factual_held / scale, factual / scale).min(1).values
                            overlap_rows.append({'gamma': gamma, 'seed': seed, 'n_train': n,
                                                 'cross_nearest_distance': near.mean().item(),
                                                 'factual_nearest_distance': normal.mean().item(),
                                                 'distance_ratio': (near.mean() / normal.mean()).item()})
                    if ref_dim == 0 and not no_cns:
                        torch.manual_seed(70000 + seed)
                        with torch.no_grad():
                            none = model(points['x_c'], points['x_a'], fd_mode='none')['probs_obs'][:, 1].numpy()
                            obs = baseline(torch.cat((points['x_c'], points['x_a']), -1)).softmax(-1)[:, 1].numpy()
                        for name, q in [('none', none), ('observational', obs)]:
                            learned_rows.append({'gamma': gamma, 'seed': seed, 'n_train': n, 'ref_dim': 0,
                                                 'no_cns2ans': False, 'model': name, 'fit_seconds': fit_time,
                                                 'mae_to_oracle': float(abs(q - oracle).mean()), **probability_metrics(q, direct)})
                        if n == max(args.train_sizes):
                            diagnostic_rows.append(diagnostics(scm, gamma, seed, model, unrestricted))
                    write_csv(out / 'learned_vs_oracle.csv', learned_rows)
                    write_csv(out / 'reference_sensitivity.csv', reference_rows)
                    write_csv(out / 'conditional_independence.csv', diagnostic_rows)
                    write_csv(out / 'overlap_diagnostics.csv', overlap_rows)
                    print(f'已完成 gamma={gamma} 样本数={n} 种子={seed} 参考维度={ref_dim} 切断CNS={no_cns}，拟合耗时 {fit_time:.1f}秒', flush=True)
    summary = []
    for gamma in args.gammas:
        for n in args.train_sizes:
            for name in ('full', 'none', 'observational'):
                group = [row for row in learned_rows if row['gamma'] == gamma and row['n_train'] == n and row['model'] == name and row['ref_dim'] == 0 and not row['no_cns2ans'] and (name != 'full' or (row['S'] == 16 and row['K'] == 64 and row['N_ref_requested'] == 2048))]
                values = np.array([row['mae'] for row in group])
                summary.append({'gamma': gamma, 'n_train': n, 'model': name, 'seeds': len(values),
                                'mae_mean': values.mean(), 'mae_std': values.std(ddof=1) if len(values) > 1 else 0})
    write_csv(out / 'gamma_sensitivity.csv', summary)
    (out / 'completed.json').write_text(json.dumps({'completed': True, 'learned_rows': len(learned_rows), 'oracle_points': len(oracle_rows)}, indent=2))
    print('三层实验及所有结果文件已写入；真实数据训练未执行。', flush=True)


if __name__ == '__main__':
    main()
