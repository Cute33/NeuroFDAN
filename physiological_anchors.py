"""观测生理统计锚点和训练期读出；不是中介真值或新增SCM结构边。"""
import torch
from torch import Tensor, nn


BANDS = ((1., 4.), (4., 8.), (8., 13.), (13., 30.), (30., 45.))


def anchor_names(cns_channels: int, peripheral_channels: int, dataset: str) -> tuple[list, list]:
    """CNS每通道6维；外周每波形通道2维，末通道GSR为4维。"""
    if dataset not in ('DEAP', 'HCI') or min(cns_channels, peripheral_channels-1) < 1:
        raise ValueError('需要DEAP/HCI及至少一个波形通道和末通道GSR')
    c = [f'EEG{ch}_{name}' for ch in range(cns_channels)
         for name in ('log_power_1_45', 'rel_delta', 'rel_theta', 'rel_alpha', 'rel_beta', 'rel_gamma')]
    mode = 'EMG' if dataset == 'DEAP' else 'ECG'
    a = [f'{mode}{ch}_{name}' for ch in range(peripheral_channels-1)
         for name in ('log_centered_rms', 'log_mean_abs_derivative')]
    return c, a+['GSR_mean', 'GSR_slope', 'GSR_log_std', 'GSR_mean_positive_derivative']


@torch.no_grad()
def extract_anchors(x_c: Tensor, x_a: Tensor, dataset: str, fs: float = 128.) -> dict:
    """配对原窗口[B,C,768]/[B,A,768]→targets及有效掩码；不读取标签。

    EEG去均值、Hann周期窗、单边periodogram；频带左闭右开，总带[1,45)。
    输入应来自既有文件的归一化前窗口；不解释为已校准物理单位或临床指标。
    """
    if x_c.ndim != 3 or x_a.ndim != 3 or x_c.shape[0] != x_a.shape[0] or x_c.shape[-1] != 768 or x_a.shape[-1] != 768:
        raise ValueError('锚点要求配对的[B,C,768]六秒窗口')
    if fs != 128. or not x_c.is_floating_point() or not x_a.is_floating_point():
        raise ValueError('当前锚点规范只接受128Hz浮点信号')
    if not torch.isfinite(x_c).all() or not torch.isfinite(x_a).all():
        raise ValueError('锚点源信号含NaN/Inf，不静默填充')
    names_c, names_a = anchor_names(x_c.shape[1], x_a.shape[1], dataset)
    c, a = x_c.double(), x_a.double()
    if c.device != a.device:
        raise ValueError('两模态设备不一致')
    eps = 1e-12
    window = torch.hann_window(768, dtype=c.dtype, device=c.device)
    freq = torch.fft.rfftfreq(768, d=1/fs, device=c.device)
    fft = torch.fft.rfft((c-c.mean(-1, keepdim=True))*window, dim=-1)
    psd = fft.abs().square()/(fs*window.square().sum())
    psd[..., 1:-1] *= 2
    power = torch.stack([psd[..., (freq >= lo) & (freq < hi)].sum(-1)*(fs/768) for lo, hi in BANDS], -1)
    total = power.sum(-1, keepdim=True)
    relative = power/total.clamp_min(eps)
    target_c = torch.cat((total.clamp_min(eps).log(), relative), -1).flatten(1)
    mask_c = (total > eps).expand(-1, -1, 6).flatten(1)
    wave = a[:, :-1]
    rms = (wave-wave.mean(-1, keepdim=True)).square().mean(-1).sqrt()
    derivative = wave.diff(dim=-1)*fs
    activity = derivative.abs().mean(-1)
    target_wave = torch.stack((rms.clamp_min(eps).log(), activity.clamp_min(eps).log()), -1).flatten(1)
    mask_wave = torch.stack((rms > eps, activity > eps), -1).flatten(1)
    gsr = a[:, -1]
    time = torch.arange(768, dtype=a.dtype, device=a.device)/fs
    time = time-time.mean()
    slope = ((gsr-gsr.mean(-1, keepdim=True))*time).sum(-1)/time.square().sum()
    std = gsr.std(-1, unbiased=False)
    target_gsr = torch.stack((gsr.mean(-1), slope, std.clamp_min(eps).log(),
                              (gsr.diff(dim=-1)*fs).clamp_min(0).mean(-1)), -1)
    mask_gsr = torch.ones_like(target_gsr, dtype=torch.bool)
    mask_gsr[:, 2] = std > eps
    target_a = torch.cat((target_wave, target_gsr), -1).float()
    if not torch.isfinite(target_c.float()).all() or not torch.isfinite(target_a).all():
        raise FloatingPointError('锚点计算超出有限数值范围，不静默截断')
    return dict(c=target_c.float(), a=target_a,
                mask_c=mask_c, mask_a=torch.cat((mask_wave, mask_gsr), -1),
                names_c=names_c, names_a=names_a)


def training_statistics(value: Tensor, mask: Tensor) -> dict[str, Tensor]:
    """只由调用方传入训练窗口；有效计数不足2或std过小的维度禁用。"""
    if value.ndim != 2 or value.shape != mask.shape or mask.dtype != torch.bool or not torch.isfinite(value).all():
        raise ValueError('统计量需有限二维target及同形bool掩码')
    count = mask.sum(0)
    mean = (value.double()*mask).sum(0)/count.clamp_min(1)
    var = ((value.double()-mean).square()*mask).sum(0)/count.clamp_min(1)
    active = (count >= 2) & (var.sqrt() > 1e-6)
    return dict(mean=mean.float(), scale=var.sqrt().clamp_min(1e-6).float(), active=active, count=count)


class PhysiologicalAnchorReadouts(nn.Module):
    """仅训练期Mc→CNS锚点、Ma→ANS锚点；Head和前门积分不调用本模块。"""
    def __init__(self, mediator_dim: int, targets: dict):
        super().__init__()
        self.cns = nn.Linear(mediator_dim, targets['c'].shape[1])
        self.ans = nn.Linear(mediator_dim, targets['a'].shape[1])
        for side in ('c', 'a'):
            stats = training_statistics(targets[side], targets['mask_'+side])
            if not stats['active'].any():
                raise ValueError(f'{side}没有可用训练锚点，停止而非零占位')
            for key, value in stats.items():
                self.register_buffer(side+'_'+key, value)

    def standardized(self, side: str, value: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """测试/训练均使用已保存的训练mean/scale；不重新拟合。"""
        mean, scale = getattr(self, side+'_mean'), getattr(self, side+'_scale')
        if value.ndim != 2 or value.shape != mask.shape or value.shape[1] != len(mean) or mask.dtype != torch.bool:
            raise ValueError('锚点与训练统计维度不符')
        return (value.detach()-mean)/scale, mask & getattr(self, side+'_active')

    def losses(self, outputs: dict, targets: dict) -> tuple[Tensor, Tensor]:
        """每模态0.5均方误差；有效B×S×锚点求平均，两个模态分别归约。

        对实际采样m而非只对mu读出；不改变sigma或额外读取另一分支特征。
        """
        losses = []
        for side, mediator, head in (('c', outputs['m_c'], self.cns), ('a', outputs['m_a'], self.ans)):
            target, valid = self.standardized(side, targets[side], targets['mask_'+side])
            if mediator.ndim != 3 or mediator.shape[0] != len(target):
                raise ValueError('中介必须为与target配对的[B,S,D]')
            prediction = head(mediator)
            target = target[:, None, :].expand_as(prediction)
            valid = valid[:, None, :].expand_as(prediction)
            if not valid.any():
                raise ValueError('本批没有有效锚点，不用零损失掩盖')
            losses.append(.5*(prediction-target).square()[valid].mean())
        return tuple(losses)
