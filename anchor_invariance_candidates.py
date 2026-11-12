"""仅供只读筛查的幅值/偏置稳定候选；未接入训练或替换原锚点。"""
import torch

from physiological_anchors import extract_anchors


@torch.no_grad()
def extract_scale_stable(x_c, x_a, dataset, fs=128.):
    """配对768点窗口→CNS每通道5维，ANS每波形通道1维及GSR2维。

    对窗口内恒定的正增益及常量偏置保持不变，不覆盖时变滤波/归一化。
    候选只描述相对频谱和波形形态，不称HRV、SCR或真实中介。
    """
    original = extract_anchors(x_c, x_a, dataset, fs)
    b, channels, n = x_c.shape
    c = original['c'].reshape(b, channels, 6)[..., 1:].flatten(1)
    mask_c = original['mask_c'].reshape(b, channels, 6)[..., 1:].flatten(1)
    wave = x_a[:, :-1].double()
    rms = (wave-wave.mean(-1, keepdim=True)).square().mean(-1).sqrt()
    variation = wave.diff(dim=-1).abs().mean(-1)
    valid_wave = (rms > 1e-12) & (variation > 1e-12)
    # 用每采样间隔差分，不乘fs，使比值无量纲；无效值仅占位，mask明确禁用。
    denominator = torch.where(valid_wave, rms, torch.ones_like(rms))
    ratio = torch.where(valid_wave, variation/denominator, torch.ones_like(rms))
    wave_target = ratio.log()
    gsr = x_a[:, -1].double()
    centered = gsr-gsr.mean(-1, keepdim=True)
    std = centered.square().mean(-1).sqrt()
    valid_gsr = std > 1e-12
    scale = torch.where(valid_gsr, std, torch.ones_like(std))
    time = torch.arange(n, device=gsr.device, dtype=gsr.dtype)/fs
    time = time-time.mean()
    slope = (centered*time).sum(-1)/time.square().sum()
    trend = (n/fs)*slope/scale
    positive = gsr.diff(dim=-1).clamp_min(0).sum(-1)/scale
    gsr_target = torch.stack((trend, positive), -1)
    gsr_target = torch.where(valid_gsr[:, None], gsr_target, torch.zeros_like(gsr_target))
    a = torch.cat((wave_target, gsr_target), -1).float()
    if not torch.isfinite(a).all():
        raise FloatingPointError('比例锚点非有限，不静默截断')
    mode = 'EMG' if dataset == 'DEAP' else 'ECG'
    return dict(c=c, a=a, mask_c=mask_c,
                mask_a=torch.cat((valid_wave, valid_gsr[:, None].expand(-1, 2)), -1),
                names_c=[name for name in original['names_c'] if 'log_power' not in name],
                names_a=[f'{mode}{j}_log_mean_abs_diff_over_rms' for j in range(x_a.shape[1]-1)]+
                        ['GSR_normalized_linear_trend','GSR_positive_variation_over_std'])


@torch.no_grad()
def affine_perturbation(value, kind):
    """构造临时探针输入；偏置为各窗口标准差倍数，原输入和文件不改写。"""
    x = value.double()
    std = x.std(-1, unbiased=False, keepdim=True)
    if kind == 'gain':
        gain, offset = 2., 0.
    elif kind == 'offset':
        gain, offset = 1., 3*std
    elif kind == 'channel_affine':
        gain = torch.linspace(.5, 2., x.shape[1], device=x.device, dtype=x.dtype)[None, :, None]
        offset = 2*std
    else:
        raise ValueError('未知幅值/偏置探针')
    return gain*x+offset


def update_invariance(record, before, after, atol=1e-5, rtol=1e-5):
    """累计所有有效元素误差、mask变化和容差失败，不能只挑平均误差。"""
    for side in ('c','a'):
        x, y = before[side].double(), after[side].double()
        mask, other = before['mask_'+side], after['mask_'+side]
        if x.shape != y.shape or mask.shape != other.shape:
            raise ValueError('变换后锚点形状改变')
        valid = mask & other
        row = record.setdefault(side, dict(elements=0, violations=0, mask_changes=0, max_abs_delta=0., sum_abs_delta=0.))
        delta = (x-y).abs()[valid]
        row['mask_changes'] += int((mask != other).sum())
        row['elements'] += len(delta)
        row['violations'] += int((delta > atol+rtol*x[valid].abs()).sum())
        row['sum_abs_delta'] += float(delta.sum())
        if len(delta): row['max_abs_delta'] = max(row['max_abs_delta'], float(delta.max()))
