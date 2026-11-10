"""重现 gamma=1 的配对拟合，诊断条件模型误差；不修改主实验结果。"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from synthetic_frontdoor_experiment import fit, make_bank, tensor_data, write_csv
from synthetic_scm import SyntheticSCM, probability_metrics


def main() -> None:
    p = argparse.ArgumentParser(description='第一阶段留出样本条件拟合误差诊断', add_help=False)
    p.add_argument('-h', '--help', action='help', help='显示参数帮助并退出')
    p.add_argument('--output', default='results/phase1_scm', help='独立诊断结果目录')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='配对拟合随机种子')
    p.add_argument('--epochs', type=int, default=20, help='与主实验相同的拟合轮数')
    args = p.parse_args()
    torch.set_num_threads(1)
    scm = SyntheticSCM()
    point = scm.sample(16, 0, 78901)
    tensors = tensor_data(point)
    truth, _ = scm.direct_intervention(point['x_c'], point['x_a'], 1., 65536, 43210)
    rows = []
    for seed in args.seeds:
        raw = scm.sample(4096, 1., 80000 + seed)
        held = tensor_data(raw)
        expected_c = torch.tensor(raw['x_c'] @ scm.B_c.T, dtype=torch.float32)
        expected_a = torch.tensor(raw['x_a'] @ scm.B_a.T + raw['m_c'] @ scm.C_ca.T, dtype=torch.float32)
        conditional = scm.conditional_outcome(raw['m_a'], raw['x_c'], raw['x_a'], 1.)
        for disabled in (False, True):
            start = time.perf_counter()
            model, _, _, train, _, _ = fit(scm, 1., 2048, seed, 0, args.epochs, disabled)
            with torch.no_grad():
                mu_c, lv_c = model.cns.distribution(held['x_c'])
                ans = model.ans(held['x_a'], held['m_c'][:, None])
                q = model.outcome(held['m_a'], model.reference_encoder(held['x_c'], held['x_a'])).softmax(-1)[:, 1].numpy()
                learned_c = model.cns.nll(held['m_c'], mu_c, lv_c).item() / 64
                oracle_c = model.cns.nll(held['m_c'], expected_c, torch.zeros_like(expected_c)).item() / 64
                learned_a = model.ans.nll(held['m_a'], ans['mu_a'][:, 0], ans['logvar_a'][:, 0]).item() / 64
                oracle_a = model.ans.nll(held['m_a'], expected_a, torch.zeros_like(expected_a)).item() / 64
                torch.manual_seed(70000 + seed)
                estimate = model(tensors['x_c'], tensors['x_a'], reference_bank=make_bank(train, 2048, seed))['probs_do'][:, 1].numpy()
            rows.append({'seed': seed, 'no_cns2ans': disabled,
                         'cns_latent_mu_rmse': (mu_c - expected_c).square().mean().sqrt().item(),
                         'ans_teacher_conditional_mu_rmse': (ans['mu_a'][:, 0] - expected_a).square().mean().sqrt().item(),
                         'cns_conditional_nll_excess_per_dim': learned_c - oracle_c,
                         'ans_conditional_nll_excess_per_dim': learned_a - oracle_a,
                         'outcome_conditional_mae': float(abs(q - conditional).mean()),
                         'reproduced_do_mae': probability_metrics(estimate, truth)['mae'],
                         'seconds': time.perf_counter() - start})
            print(f'留出诊断完成：种子 {seed}，切断 CNS={disabled}', flush=True)
    write_csv(Path(args.output) / 'learning_diagnostics.csv', rows)


if __name__ == '__main__':
    main()
