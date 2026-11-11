"""Phase 6独立实验目标；不接入真实训练器，不宣称辅助重建提供因果识别。"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from frontdoor_scm import FrontDoorEstimator, gaussian_kl


CONDITIONS = ('oracle_mediator', 'observation_only', 'observation_input_fidelity')


def check_training_view(data: dict[str, Tensor], condition: str) -> None:
    """白名单隔离监督权限；无中介条件连字典中也不能持有真值/U/干预概率。"""
    if condition not in CONDITIONS:
        raise ValueError('未知监督条件')
    expected = {'x_c', 'x_a', 'y'} | ({'m_c', 'm_a'} if condition == 'oracle_mediator' else set())
    if set(data) != expected:
        raise ValueError(f'训练视图权限不符：{condition}只允许{sorted(expected)}')
    n = len(data['y'])
    if n < 2 or any(len(v) != n for v in data.values()):
        raise ValueError('训练张量样本轴不匹配或不足两个样本')


class ModalityReadouts(nn.Module):
    """训练期线性统计读出Mc→标准化Xc、Ma→标准化Xa；不是SCM反向边。"""
    def __init__(self, d_m: int, x_c: Tensor, x_a: Tensor):
        super().__init__()
        self.cns = nn.Linear(d_m, x_c.shape[1])
        self.ans = nn.Linear(d_m, x_a.shape[1])
        for name, value in (('c', x_c), ('a', x_a)):
            self.register_buffer('mean_'+name, value.mean(0))
            self.register_buffer('scale_'+name, value.std(0, unbiased=False).clamp_min(1e-6))

    def losses(self, outputs: dict[str, Tensor], x_c: Tensor, x_a: Tensor) -> tuple[Tensor, Tensor]:
        """B/S/观测特征均求平均；用实际随机中介，不只约束mu或人为设小sigma。"""
        b, s, _ = outputs['m_c'].shape
        target_c = ((x_c-self.mean_c)/self.scale_c)[:, None].expand(b, s, -1)
        target_a = ((x_a-self.mean_a)/self.scale_a)[:, None].expand(b, s, -1)
        return (.5*F.mse_loss(self.cns(outputs['m_c']), target_c),
                .5*F.mse_loss(self.ans(outputs['m_a']), target_a))


@dataclass
class LearningWeights:
    """仅独立合成实验配置；不改变真实数据默认权重。"""
    lambda_obs: float = 1.
    fidelity: float = 1.
    mediator_nll: float = 1.
    fire: float = 0.


def learning_terms(model: FrontDoorEstimator, readouts: ModalityReadouts,
                   batch: dict[str, Tensor], condition: str, epoch: int,
                   weights: LearningWeights) -> tuple[dict[str, Tensor], dict]:
    """同一核心估计器：真值中介仅oracle分支访问，干预标签从不进入训练。"""
    check_training_view(batch, condition)
    out = model(batch['x_c'], batch['x_a'], fd_mode='none')
    rates_c, rates_a = model.cns.snn.last_rates, model.ans.snn.last_rates
    cfg = model.config
    coefficient = cfg.beta*min(epoch/cfg.beta_warmup_epochs, 1.)
    kl_c = gaussian_kl(out['mu_c'], out['logvar_c']).mean()
    kl_a = gaussian_kl(out['mu_a'], out['logvar_a']).mean(1).mean()
    logits = out['logits_obs']
    # oracle增加训练期真实中介信息，其他组的代码路径不读取这些字段。
    zero = logits.new_zeros(())
    nll_c, nll_a = zero, zero
    if condition == 'oracle_mediator':
        conditional = model.ans(batch['x_a'], batch['m_c'][:, None])
        nll_c = model.cns.nll(batch['m_c'], out['mu_c'], out['logvar_c'])/cfg.mediator_dim
        nll_a = model.ans.nll(batch['m_a'], conditional['mu_a'][:, 0], conditional['logvar_a'][:, 0])/cfg.mediator_dim
        logits = model.outcome(batch['m_a'], model.reference_encoder(batch['x_c'], batch['x_a']))
    ce = F.cross_entropy(logits, batch['y'])
    # 与真实模型同形式的发放率项；合成默认关闭，显式报告，不声称等同HCI权重。
    fire = .5*((rates_c.mean(0)-.05).square().mean()+
                (rates_a.mean(0)-.05).square().mean())
    fidelity_c, fidelity_a = (readouts.losses(out, batch['x_c'], batch['x_a'])
        if condition == 'observation_input_fidelity' else (zero, zero))
    terms = dict(factual_ce=weights.lambda_obs*ce, kl_c=coefficient*kl_c,
        kl_a=coefficient*kl_a, fire=weights.fire*fire,
        mediator_nll=weights.mediator_nll*(nll_c+nll_a),
        fidelity_c=weights.fidelity*fidelity_c, fidelity_a=weights.fidelity*fidelity_a)
    terms['total'] = sum(terms.values())
    out['phase6_unweighted_losses'] = dict(factual_ce=ce,kl_c=kl_c,kl_a=kl_a,
        fire=fire,mediator_nll=nll_c+nll_a,fidelity_c=fidelity_c,fidelity_a=fidelity_a)
    return terms, out


def gradient_components(terms: dict[str, Tensor], named: list[tuple[str, nn.Parameter]]) -> dict:
    """autograd.grad分项范数和CE夹角；不写.grad，允许后续对total正常backward。"""
    parameters = [p for _, p in named]
    grouped = {}
    if 'factual_ce' not in terms:
        raise ValueError('分项夹角诊断必须包含事实CE')
    labels = ['factual_ce']+[key for key in terms if key not in ('total','factual_ce')]
    ce_gradients = None
    for label in labels:
        value = terms[label]
        gradients = (torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                     if value.requires_grad else (None,)*len(parameters))
        if label == 'factual_ce':
            ce_gradients = gradients
        accum = {}
        for i, ((name, _), g) in enumerate(zip(named, gradients)):
            local = name.removeprefix('frontdoor.')
            group = ('cns2ans' if local.startswith('ans.cns_current.') else
                     '.'.join(local.split('.')[:2]) if local.startswith(('backbone.','readouts.')) else
                     local.split('.')[0])
            row = accum.setdefault(group, dict(square=0., missing=0, zero=0, dot_ce=0., ce_square=0.))
            row['missing'] += int(g is None)
            c = ce_gradients[i] if ce_gradients is not None else None
            if c is not None:
                c = c.detach().double()
                row['ce_square'] += c.square().sum().item()
            if g is not None:
                g = g.detach().double()
                if not torch.isfinite(g).all():
                    raise FloatingPointError('分项梯度出现NaN/Inf')
                square = g.square().sum().item()
                row['square'] += square
                row['zero'] += int(square==0.)
                if c is not None:
                    row['dot_ce'] += (g*c).sum().item()
        grouped[label] = {}
        for group, row in accum.items():
            norm, ce_norm = row['square']**.5, row['ce_square']**.5
            grouped[label][group] = dict(l2=norm, missing_tensors=row['missing'],zero_tensors=row['zero'],
                cosine_to_ce=max(-1., min(1., row['dot_ce']/(norm*ce_norm))) if norm*ce_norm else None)
    return grouped
