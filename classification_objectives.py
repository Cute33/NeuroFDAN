"""观察标签预测目标；积分形式的NLL不是干预真值监督或因果识别证明。"""
import torch
from torch.nn import functional as F

from frontdoor_scm import frontdoor_loss


OBJECTIVES = ('factual', 'integrated_observational')


def integrated_observational_nll(outputs, labels):
    """复用唯一估计器已输出的log(mean softmax)，不重采样、不平均交叉配对CE。"""
    probs, log_probs = outputs.get('probs_do'), outputs.get('logits_do')
    if probs is None or log_probs is None:
        raise ValueError('积分观察风险需要fd_mode=full的probs_do/logits_do')
    if probs.ndim != 2 or probs.shape[1] != 2 or log_probs.shape != probs.shape or labels.shape != (len(probs),):
        raise ValueError('需要[B,2]积分概率和[B]观察标签')
    if not torch.isfinite(probs).all() or not torch.isfinite(log_probs).all():
        raise FloatingPointError('积分概率或其log含NaN/Inf')
    if ((probs < 0) | (probs > 1)).any() or not torch.allclose(probs.sum(-1), torch.ones_like(probs[:, 0]), atol=1e-5, rtol=1e-5):
        raise FloatingPointError('积分概率范围或归一化错误')
    if not torch.allclose(log_probs.exp(), probs, atol=1e-6, rtol=1e-5):
        raise ValueError('logits_do必须是积分概率的log，不是普通分类logits')
    target = labels.long()
    if ((target < 0) | (target > 1)).any():
        raise ValueError('观察标签必须为0/1')
    if (probs.gather(1, target[:, None]) < torch.finfo(probs.dtype).tiny).any():
        raise FloatingPointError('目标类概率低于正常浮点范围，停止以免log clamp掩盖零梯度')
    return F.nll_loss(log_probs, target)


def classification_terms(outputs, labels, config, epoch=1, *, objective='factual', lambda_obs=1.):
    """保留原事实模式；方案A替换主分类项，事实CE只记录，不叠加第二个任务损失。"""
    if objective not in OBJECTIVES:
        raise ValueError('未知分类目标')
    terms = frontdoor_loss(outputs, labels, config, epoch=epoch)
    coefficient = config.beta * min(max(epoch, 0) / config.beta_warmup_epochs, 1.)
    if objective == 'factual':
        # 与旧表达式相同的运算顺序，保持基线损失和梯度。
        selected = terms['loss'] + (lambda_obs - 1.) * terms['loss_obs']
        primary = terms['loss_obs']
        weighted = lambda_obs * primary
    else:
        if lambda_obs != 0:
            raise ValueError('方案A替换事实CE，需要显式--lambda_obs 0，不静默叠加分类项')
        primary = integrated_observational_nll(outputs, labels)
        weighted = primary
        selected = primary + coefficient * terms['loss_kl']
        terms['loss_integrated_observational'] = primary
    terms.update(loss=selected, classification_loss=primary,
                 weighted_classification_loss=weighted,
                 weighted_obs_loss=lambda_obs * terms['loss_obs'],
                 weighted_kl_loss=coefficient * terms['loss_kl'])
    return terms
