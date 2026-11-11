"""仅合成中介真值监督的凸结果模型对照；冻结中介，不接入真实训练器。"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from torch import nn

from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator
from phase6_synthetic_experiment import class_metrics, data_digest, save_json, training_view
from run_reporting import source_version
from scripts.evaluate_phase6_components import (CELLS, TRAINING_COMMIT, evaluate_gamma,
    file_digest, prepare_output_directory, state_digest)
from synthetic_scm import SCMConfig, SyntheticSCM, probability_metrics


def logistic_objective(theta: np.ndarray, design: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """只用训练标签的平均二项负对数似然及解析梯度；不使用干预真值。"""
    eta = design @ theta
    return float(np.mean(np.logaddexp(0, eta)-y*eta)), design.T @ (expit(eta)-y)/len(y)


class LinearOutcomeControl(nn.Module):
    """双logit为[0,eta]；只接收Ma和联合参考，不作为新真实模型发布。"""
    def __init__(self, mediator_dim: int, reference_dim: int, weight: np.ndarray, bias: float):
        super().__init__()
        if weight.shape != (mediator_dim+reference_dim,) or not np.isfinite(weight).all() or not np.isfinite(bias):
            raise ValueError('线性条件结果参数维度或数值无效')
        self.mediator_dim, self.reference_dim = mediator_dim, reference_dim
        self.register_buffer('weight', torch.tensor(weight, dtype=torch.float32))
        self.register_buffer('bias', torch.tensor([bias], dtype=torch.float32))

    def forward(self, m_a: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """显式对齐的[...,D]、[...,d_ref]→[...,2]，无Mc输入。"""
        if m_a.shape[:-1] != reference.shape[:-1] or m_a.shape[-1] != self.mediator_dim or reference.shape[-1] != self.reference_dim:
            raise ValueError('Ma与参考的轴或特征维度不匹配')
        joint = torch.cat((m_a, reference), -1)
        eta = nn.functional.linear(joint, self.weight[None], self.bias).squeeze(-1)
        return torch.stack((torch.zeros_like(eta), eta), -1)


def fit_head(view: dict) -> tuple[LinearOutcomeControl, dict]:
    """权限为训练Ma/Xc/Xa/Y；不接收SCM、U、Mc、测试集或干预概率。"""
    if set(view) != {'m_a', 'x_c', 'x_a', 'y'}:
        raise ValueError('中介真值监督对照的训练视图只能含m_a/x_c/x_a/y')
    ma, xc, xa, y = (np.asarray(view[key], dtype=np.float64) for key in ('m_a', 'x_c', 'x_a', 'y'))
    if y.ndim != 1 or any(v.ndim != 2 or len(v) != len(y) for v in (ma, xc, xa)):
        raise ValueError('训练视图样本轴或特征维度错误')
    if not all(np.isfinite(v).all() for v in (ma, xc, xa, y)) or set(np.unique(y)) != {0., 1.}:
        raise ValueError('训练数据必须有限且包含两个标签')
    features = np.concatenate((ma, xc, xa), 1)
    if len(y) <= features.shape[1]+1:
        raise ValueError('本对照要求训练样本数超过参数数目')
    # 标准化只作求解器数值预处理；最后精确换回原特征坐标，不改原数据管线。
    mean, scale = features.mean(0), features.std(0)
    scale = np.where(scale < 1e-8, 1., scale)
    design = np.column_stack(((features-mean)/scale, np.ones(len(y))))
    started = time.perf_counter()
    result = minimize(logistic_objective, np.zeros(design.shape[1]), args=(design, y),
        jac=True, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12})
    loss, gradient = logistic_objective(result.x, design, y)
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(loss) or np.max(abs(gradient)) > 1e-4:
        raise RuntimeError(f'条件结果MLE未收敛：状态={result.status}，迭代={result.nit}，最大梯度={np.max(abs(gradient))}')
    weight = result.x[:-1]/scale
    bias = float(result.x[-1]-mean @ weight)
    model = LinearOutcomeControl(ma.shape[1], xc.shape[1]+xa.shape[1], weight, bias).eval()
    return model, dict(training_keys=sorted(view), n=len(y), parameter_count=len(result.x),
        train_label_counts=np.bincount(y.astype(np.int64), minlength=2).tolist(),
        optimizer='L-BFGS-B', maxiter=500, gtol=1e-8, ftol=1e-12, iterations=int(result.nit),
        converged=bool(result.success), final_mean_nll=loss, max_abs_gradient=float(np.max(abs(gradient))),
        fit_seconds=time.perf_counter()-started, penalty=0.,
        ma_weight_norm=float(np.linalg.norm(weight[:ma.shape[1]])),
        reference_weight_norm=float(np.linalg.norm(weight[ma.shape[1]:])),
        note='权重范数仅诊断；gamma0真实参考系数为零，但未将拟合系数强制设零。')


def head_training_view(scm: SyntheticSCM, gamma: float, base: dict,
                       size: int, seed: int) -> tuple[dict, dict]:
    """仅扩展合成真值中介监督的条件拟合样本；原参考库和已训中介不扩展。"""
    if set(base) != {'m_a', 'x_c', 'x_a', 'y'}:
        raise ValueError('条件拟合基础视图只能含m_a/x_c/x_a/y')
    n = len(base['y'])
    if n < 2 or size < n:
        raise ValueError('条件拟合样本不得少于原训练集')
    pieces = {key: [np.array(value, copy=True)] for key, value in base.items()}
    seeds, remaining, block = [], size-n, 0
    while remaining:
        # 块大小固定为原样本量；因此小规模样本是大规模样本的严格前缀。
        block_seed = seed+10000+block
        extra = training_view(scm.sample(n, gamma, block_seed), 'oracle_mediator')
        count = min(remaining, n)
        for key in base:
            pieces[key].append(extra[key][:count].numpy().copy())
        seeds.append(block_seed)
        remaining -= count
        block += 1
    view = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
    digest = hashlib.sha256()
    for key in sorted(view):
        digest.update(key.encode())
        digest.update(view[key].tobytes())
    return view, dict(n=size, original_prefix_n=n, block_size=n,
                     augmentation_seeds=seeds, training_view_sha256=digest.hexdigest(),
                     reference_population_unchanged=True)


def core_digest(model: FrontDoorEstimator) -> dict:
    """冻结模块状态指纹，不包含本轮允许拟合的条件结果对照。"""
    return {name: state_digest(getattr(model, name)) for name in ('cns', 'ans', 'reference_encoder')}


def own_pair_metrics(model: FrontDoorEstimator, raw: dict, scm: SyntheticSCM, gamma: float) -> dict:
    """拟合结束后的独立评分；真实条件概率不返回训练函数。"""
    with torch.no_grad():
        ma = torch.tensor(raw['m_a'], dtype=torch.float32).clamp(-1+1e-6, 1-1e-6)
        ref = model.reference_encoder(torch.tensor(raw['x_c'], dtype=torch.float32),
                                      torch.tensor(raw['x_a'], dtype=torch.float32))
        p = model.outcome(ma, ref).softmax(-1).numpy()
    true_conditional = scm.conditional_outcome(raw['m_a'], raw['x_c'], raw['x_a'], gamma)
    return dict(against_observed_labels=class_metrics(raw['y'], p),
                against_true_conditional=probability_metrics(p[:, 1], true_conditional))


def main() -> None:
    p = argparse.ArgumentParser(description='冻结合成真值监督对照的中介，仅拟合线性条件结果', add_help=False)
    p.add_argument('-h', '--help', action='help', help='显示中文帮助')
    p.add_argument('--run_dir', type=Path, required=True, help='原paired_tanh_normal_20ep_001目录，含本地模型和SCM参数')
    p.add_argument('--component_dir', type=Path, required=True, help='已成功的tanh_oracle_components_002目录')
    p.add_argument('--output_dir', type=Path, required=True, help='新的phase6结果目录，禁止覆盖')
    p.add_argument('--seed', type=int, choices=[42], default=42, help='沿用原合成训练数据种子')
    p.add_argument('--device', choices=['cpu'], default='cpu', help='只运行小型CPU合成对照')
    p.add_argument('--workers', type=int, choices=[0], default=0, help='不创建数据加载worker')
    p.add_argument('--diagnostic_level', choices=['full'], default='full', help='记录条件拟合、四格及MC误差')
    p.add_argument('--head_train_size', type=int, default=None,
                   help='仅扩展合成条件结果拟合样本；默认原训练数量；中介及参考仍用原2048样本版本')
    p.add_argument('--S', type=int, default=256, help='沿用上轮中介评价次数')
    p.add_argument('--K', type=int, default=128, help='沿用上轮参考评价次数')
    p.add_argument('--repeats', type=int, default=4, help='沿用上轮MC重复，非训练种子数')
    args = p.parse_args()
    if min(args.S, args.K, args.repeats) < 2:
        p.error('S/K/重复至少为2')
    run, component, out = (getattr(args, key).resolve() for key in ('run_dir', 'component_dir', 'output_dir'))
    if any(out == root or root in out.parents or out in root.parents for root in (run, component)):
        p.error('输出必须独立于两个输入目录')
    prepare_output_directory(out)
    result = dict(operation='合成中介真值监督的条件结果估计器对照', completed=False, training_executed=False,
        training_scope='仅拟合独立线性条件结果对照；原中介和原checkpoint冻结',
        real_data_training=False, source_version=source_version(PROJECT), gammas=[])
    save_json(out/'probe_summary.json', result)
    inputs = [run/'config.json', run/'probe_summary.json', run/'scm_parameters.npz',
              component/'config.json', component/'probe_summary.json']
    hashes = {str(path): file_digest(path) for path in inputs}
    cfg, saved, _, previous_cfg, previous = [json.loads(path.read_text(encoding='utf-8')) if path.suffix == '.json' else None for path in inputs]
    a = cfg['arguments']
    head_size = a['train_size'] if args.head_train_size is None else args.head_train_size
    if head_size < a['train_size']:
        raise ValueError('条件拟合数量不得少于原训练集；本轮只检验样本量增加')
    if cfg['source_version']['commit'] != TRAINING_COMMIT or not saved['completed'] or saved['smoke'] or a['mediator_family'] != 'tanh_normal' or a['no_cns2ans'] or a['seed'] != args.seed:
        raise ValueError('原合成实验状态或分布族不匹配')
    if not previous['completed'] or not previous['source_files_unchanged'] or previous['training_executed']:
        raise ValueError('前一轮四格评价未成功或状态异常')
    if [row['gamma'] for row in saved['gammas']] != a['gammas'] or [row['gamma'] for row in previous['gammas']] != a['gammas']:
        raise ValueError('两次输入的gamma记录不一致')
    if any(previous_cfg['arguments'][key] != getattr(args, key) for key in ('seed', 'S', 'K', 'repeats')):
        raise ValueError('必须沿用前次评价的seed/S/K/重复以便配对比较')
    scm = SyntheticSCM(SCMConfig(**cfg['scm']))
    with np.load(run/'scm_parameters.npz', allow_pickle=False) as params:
        for key in ('A_c', 'A_a', 'B_c', 'B_a', 'C_ca', 'w_a', 'w_u', 'b'):
            if not np.array_equal(params[key], np.asarray(getattr(scm, key))):
                raise ValueError('SCM参数重建不一致：'+key)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    points = scm.sample(a['points'], 0., 78901)
    save_json(out/'config.json', dict(source_version=result['source_version'],
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        original_arguments=a, scm=cfg['scm'],
        assumptions='线性logit为gamma0正确设定对照；gamma1因logistic-normal混合而非精确条件族。未用于真实数据。',
        selection='固定无惩罚MLE，无验证/真值调参；只用中介真值监督锚定的坐标。'))
    for target, base in zip(saved['gammas'], previous['gammas']):
        gamma = target['gamma']
        record = next(row for row in target['conditions'] if row['condition'] == 'oracle_mediator')
        checkpoint = run/'models'/f'gamma{gamma}_oracle_mediator_last.pt'
        hashes[str(checkpoint)] = file_digest(checkpoint)
        if hashes[str(checkpoint)] != base['checkpoint_sha256'] or not base['state_unchanged']:
            raise ValueError('checkpoint与前次已核验评价不一致')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['config'] != record['config'] or state['condition'] != 'oracle_mediator' or state['epoch'] != a['epochs'] or state['seed'] != args.seed:
            raise ValueError('中介真值监督checkpoint配置、轮数或监督条件不匹配')
        model_cfg = FrontDoorConfig(**state['config'])
        if model_cfg.ref_dim != 0 or model_cfg.mediator_family != 'tanh_normal' or model_cfg.no_cns2ans:
            raise ValueError('必须保持完整参考和原tanh条件中介路径')
        model = FrontDoorEstimator(scm.config.d_xc, scm.config.d_xa, model_cfg).cpu().eval()
        model.load_state_dict(state['model'], strict=True)
        model.requires_grad_(False)
        core_before = core_digest(model)
        raw = scm.sample(a['train_size'], gamma, args.seed+1000)
        observed = scm.sample(a['test_size'], gamma, args.seed+2000)
        train = training_view(raw, 'oracle_mediator')
        xy = {key: train[key] for key in ('x_c', 'x_a', 'y')}
        if data_digest(xy) != record['training_data_sha256']:
            raise ValueError('训练X/Y指纹不匹配')
        # 训练函数严格不接收SCM或测试数据；已知条件概率仅在拟合结束后评价。
        base_view = {key: train[key].numpy().copy() for key in ('m_a', 'x_c', 'x_a', 'y')}
        view, head_data = head_training_view(scm, gamma, base_view, head_size, args.seed)
        if result['training_executed'] is not True:
            result['training_executed'] = None
        result['training_status'] = f'gamma={gamma}条件结果拟合已启动；若失败请查日志，不代表已完成'
        save_json(out/'probe_summary.json', result)
        head, fit = fit_head(view)
        result['training_executed'] = True
        result['training_status'] = f'gamma={gamma}拟合完成，准备固定评价'
        baseline_scores = {name: own_pair_metrics(model, data, scm, gamma)
                           for name, data in (('train', raw), ('test', observed))}
        model.outcome = head
        current_scores = {name: own_pair_metrics(model, data, scm, gamma)
                          for name, data in (('train', raw), ('test', observed))}
        evaluation = evaluate_gamma(model, scm, xy, points, observed, gamma, target, args)
        for cell in (CELLS[0], CELLS[2]):
            if not np.allclose(evaluation['cells'][cell]['probability_per_repeat'],
                               base['cells'][cell]['probability_per_repeat'], rtol=0, atol=1e-10):
                raise RuntimeError('冻结中介或公共随机流与原评价不一致')
        if core_digest(model) != core_before or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('原中介/参考状态或只拟合Head的权限保护失败')
        models = out/'models'
        models.mkdir(exist_ok=True)
        np.savez(models/f'gamma{gamma}_linear_head.npz', weight=head.weight.numpy(), bias=head.bias.numpy())
        row = dict(gamma=gamma, head_fit=fit, head_training_data=head_data, baseline_own_pairs=baseline_scores,
            control_own_pairs=current_scores, baseline_cells=base['cells'], control_evaluation=evaluation,
            frozen_core_unchanged=True, shared_random_streams_verified=True,
            training_data_sha256=data_digest(xy), checkpoint_sha256=hashes[str(checkpoint)])
        result['gammas'].append(row)
        save_json(out/'probe_summary.json', result)
        print(f'gamma={gamma} 合成线性结果对照拟合完成，原中介未更新；原/新LL概率MSE='+
              str((base['cells'][CELLS[3]]['against_direct']['probability_mse'],
                   evaluation['cells'][CELLS[3]]['against_direct']['probability_mse'])), flush=True)
    if any(file_digest(Path(path)) != digest for path, digest in hashes.items()):
        raise RuntimeError('源文件在本次操作期间发生变化')
    result.update(completed=True, source_files_unchanged=True, source_sha256=hashes)
    save_json(out/'probe_summary.json', result)
    (out/'run_report.txt').write_text('合成中介真值监督条件结果对照完成；未训练中介或真实数据。\n'
        '线性logit仅为统计拟合对照，不是已获验证的真实模型或潜在中介识别方案。\n', encoding='utf-8')
    print('结果已保存：'+str(out), flush=True)


if __name__ == '__main__':
    main()
