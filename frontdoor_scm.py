"""第一阶段前门核心；独立合成验证与第二阶段真实模型共同复用。"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class FrontDoorConfig:
    """采样数仅控制数值积分；参考维度为零表示不额外投影。"""
    mediator_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.3
    S_train: int = 4
    K_train: int = 16
    S_eval: int = 16
    K_eval: int = 64
    N_ref: int = 2048
    beta: float = 1e-3
    beta_warmup_epochs: int = 10
    logvar_min: float = -6.0
    logvar_max: float = 2.0
    ref_dim: int = 0
    num_classes: int = 2
    time_steps: int = 4
    no_cns2ans: bool = False
    mediator_family: str = 'normal'
    reference_chunk_size: int = 0
    lif_neurons: Optional[int] = None
    cns_input_norm: str = 'none'
    temporal_mediator: bool = False   # 新增：中介直接读取96步时间序列
    ref_pool: str = 'flat'            # 新增：flat完整展平；meanstd为时间维均值+标准差
    seq_len: int = 96                 # 新增：骨干输出的时间步数

    @property
    def lif_width(self) -> int:
        """旧配置未指定时沿用hidden_dim；真实CLI可独立指定LIF神经元数。"""
        return self.hidden_dim if self.lif_neurons is None else self.lif_neurons

    def __post_init__(self):
        if self.cns_input_norm not in ('none', 'layernorm'):
            raise ValueError('CNS入口归一化仅支持none或layernorm')
        if self.ref_pool not in ('flat', 'meanstd'):          # 新增
            raise ValueError('ref_pool仅支持flat或meanstd')
        for key in ('mediator_dim', 'hidden_dim', 'S_train', 'K_train', 'S_eval',
                    'K_eval', 'N_ref', 'beta_warmup_epochs', 'time_steps', 'seq_len'):   # 新增seq_len
            if getattr(self, key) < 1:
                raise ValueError(f'{key} 必须为正整数')
        if self.ref_dim < 0 or self.num_classes < 2 or self.reference_chunk_size < 0:
            raise ValueError('参考维度不得为负；softmax 类别数至少为二')
        if not 0 <= self.dropout < 1 or self.logvar_min >= self.logvar_max or self.beta < 0:
            raise ValueError('dropout、方差边界或 KL 系数无效')
        if self.mediator_family not in ('normal', 'tanh_normal'):
            raise ValueError('中介分布仅支持 normal 和 tanh_normal')
        if self.lif_neurons is not None and (not isinstance(self.lif_neurons, int) or self.lif_neurons < 1):
            raise ValueError('LIF神经元数必须为正整数或旧配置的None')


def flatten_input(x: Tensor, dim: int) -> Tensor:
    """保留 batch 轴，完整展平其余轴；不进行池化。"""
    if x.ndim < 2 or x[0:1].numel() != dim:
        raise ValueError(f'每个输入应有 {dim} 个特征，实际形状为 {tuple(x.shape)}')
    return x.reshape(x.shape[0], dim)


class ReferenceEncoder(nn.Module):
    """输入配对 h_c、h_a；flat完整拼接；meanstd取时间维均值+标准差（HCI约288维）。"""
    def __init__(self, cns_dim: int, ans_dim: int, ref_dim: int = 0,
                 pool: str = 'flat', seq_len: int = 96):
        super().__init__()
        if min(cns_dim, ans_dim) < 1 or ref_dim < 0:
            raise ValueError('输入维度必须为正，参考维度不得为负')
        if pool not in ('flat', 'meanstd'):
            raise ValueError('pool仅支持flat或meanstd')
        if pool == 'meanstd' and (cns_dim % seq_len or ans_dim % seq_len):
            raise ValueError('meanstd要求输入维度能被seq_len整除')
        self.cns_dim, self.ans_dim, self.pool, self.seq_len = cns_dim, ans_dim, pool, seq_len
        self.joint_dim = (2 * (cns_dim // seq_len) + 2 * (ans_dim // seq_len)
                          if pool == 'meanstd' else cns_dim + ans_dim)
        self.output_dim = ref_dim or self.joint_dim
        self.projection = nn.Linear(self.joint_dim, ref_dim) if ref_dim else nn.Identity()

    def _encode(self, x: Tensor, dim: int) -> Tensor:
        if self.pool == 'meanstd':
            if x[0:1].numel() != dim:
                raise ValueError(f'每个输入应有 {dim} 个特征，实际形状为 {tuple(x.shape)}')
            x = x.reshape(x.shape[0], self.seq_len, dim // self.seq_len)
            return torch.cat((x.mean(1), (x.var(1, unbiased=False) + 1e-6).sqrt()), -1)
        return flatten_input(x, dim)

    def joint(self, x_c: Tensor, x_a: Tensor) -> Tensor:
        """输出 [R,joint_dim]，投影前的联合表示始终可用。"""
        if x_c.shape[0] != x_a.shape[0]:
            raise ValueError('CNS 和 ANS 的参考样本数必须相同')
        return torch.cat((self._encode(x_c, self.cns_dim), self._encode(x_a, self.ans_dim)), -1)

    def forward(self, x_c: Tensor, x_a: Tensor) -> Tensor:
        return self.projection(self.joint(x_c, x_a))


class CurrentLIF(nn.Module):
    """复用原 LIF 参数和 surrogate；附加电流在神经元电流处相加。"""
    def __init__(self, dim: int):
        super().__init__()
        from NeuroFDAN import VectorizedLIFLayer
        self.cell = VectorizedLIFLayer(dim, dim)

    def forward(self, drive: Tensor, extra: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """drive 为 [Q,T,H]，extra 为 [Q,H]；返回发放率和平均总电流。"""
        from NeuroFDAN import surrogate_spike
        if drive.ndim != 3:
            raise ValueError('LIF 驱动必须为 [Q,T,H]')
        current = self.cell.input_proj(drive) + self.cell.bias
        if extra is not None:
            if extra.shape != (drive.shape[0], drive.shape[2]):
                raise ValueError('附加电流必须为 [Q,H]')
            current = current + extra[:, None, :].expand_as(current)
        voltage = torch.zeros_like(current[:, 0])
        alpha = self.cell.alpha.to(current)
        spikes = []
        for t in range(drive.shape[1]):
            voltage = alpha * voltage + (1 - alpha) * current[:, t]
            threshold = self.cell.thresh[None, :].expand_as(voltage)
            spike = surrogate_spike(voltage, threshold, torch.ones_like(voltage))
            voltage = voltage - spike * threshold
            spikes.append(spike)
        self.last_rates = torch.stack(spikes, 1).mean(1)
        return self.last_rates, current.mean(1)


class GaussianMediator(nn.Module):
    """局部分支生成高斯参数；temporal_mediator=True时逐时间步驱动LIF。"""
    def __init__(self, input_dim: int, config: FrontDoorConfig):
        super().__init__()
        self.input_dim, self.config = input_dim, config
        h, neurons = config.hidden_dim, config.lif_width
        self.temporal = config.temporal_mediator
        if self.temporal:
            if input_dim % config.seq_len:
                raise ValueError('temporal_mediator要求输入维度能被seq_len整除')
            # 每个时间步共享一个线性投影到神经元电流；不做逐步LayerNorm，保留幅度信息
            self.step_proj = nn.Linear(input_dim // config.seq_len, neurons)
        else:
            self.encoder = nn.Sequential(nn.Linear(input_dim, h), nn.SiLU(),
                                         nn.Linear(h, config.time_steps * neurons))
        self.snn = CurrentLIF(neurons)
        self.parameters_head = nn.Linear(2 * neurons, 2 * config.mediator_dim)

    def distribution(self, x: Tensor, extra: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        flat = flatten_input(x, self.input_dim)
        if self.temporal:
            drive = self.step_proj(flat.reshape(flat.shape[0], self.config.seq_len, -1))   # [Q,96,neurons]
        else:
            drive = self.encoder(flat)
            drive = drive.reshape(x.shape[0], self.config.time_steps, self.config.lif_width)
        rates, current = self.snn(drive, extra)
        mu, logvar = self.parameters_head(torch.cat((rates, current), -1)).chunk(2, -1)
        return mu, logvar.clamp(self.config.logvar_min, self.config.logvar_max)

    def draw(self, mu: Tensor, logvar: Tensor) -> Tensor:
        value = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return value.tanh() if self.config.mediator_family == 'tanh_normal' else value

    def nll(self, value: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        """对观测中介的条件负对数密度：特征求和，其他轴平均。"""
        if value.shape != mu.shape or mu.shape != logvar.shape:
            raise ValueError('中介真值和条件参数形状不一致')
        if self.config.mediator_family == 'tanh_normal':
            if torch.any(value.abs() >= 1):
                raise ValueError('tanh 中介必须严格位于 (-1,1)')
            latent = torch.atanh(value)
            jacobian = torch.log1p(-value.square())
        else:
            latent, jacobian = value, torch.zeros_like(value)
        term = 0.5 * ((latent - mu).square() * (-logvar).exp() + logvar + 1.8378770664093453)
        return (term + jacobian).sum(-1).mean()


class CNSMediator(GaussianMediator):
    """仅接收 CNS 输入；参数 [B,D]，独立噪声采样 [B,S,D]。"""
    def __init__(self, input_dim: int, config: FrontDoorConfig):
        super().__init__(input_dim, config)
        if config.cns_input_norm == 'layernorm' and input_dim < 2:
            raise ValueError('标量CNS输入不能使用LayerNorm：会退化为常数')
        self.input_norm = (nn.LayerNorm(input_dim, eps=1e-5, elementwise_affine=False)
                           if config.cns_input_norm == 'layernorm' else None)

    def distribution(self, x: Tensor, extra: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        # oracle条件似然与forward统一走同一入口；ANS和参考输入保持原路径。
        if self.input_norm is not None:
            x = self.input_norm(flatten_input(x, self.input_dim))
        return super().distribution(x, extra)

    def forward(self, x_c: Tensor, S: int = 4) -> dict[str, Tensor]:
        if S < 1:
            raise ValueError('S 必须为正整数')
        mu, logvar = self.distribution(x_c)
        return {'mu_c': mu, 'logvar_c': logvar,
                'm_c': self.draw(mu[:, None].expand(-1, S, -1), logvar[:, None].expand(-1, S, -1))}


class ANSMediator(GaussianMediator):
    """只接收 ANS 和采样 CNS 中介；每次采样分别生成 [B,S,D] 参数。"""
    def __init__(self, input_dim: int, config: FrontDoorConfig):
        super().__init__(input_dim, config)
        self.cns_current = None if config.no_cns2ans else nn.Linear(config.mediator_dim, config.lif_width)

    def forward(self, x_a: Tensor, m_c: Tensor) -> dict[str, Tensor]:
        if m_c.ndim != 3 or m_c.shape[0] != x_a.shape[0] or m_c.shape[-1] != self.config.mediator_dim:
            raise ValueError('CNS 中介必须为 [B,S,D] 并与 ANS 对齐')
        b, s, d = m_c.shape
        inputs = flatten_input(x_a, self.input_dim)[:, None, :].expand(b, s, self.input_dim)
        # 消融只读取形状，完全不读取 CNS 数值，且不构造梯度路径。
        extra = None if self.cns_current is None else self.cns_current(m_c).reshape(b * s, self.config.lif_width)
        mu, logvar = self.distribution(inputs.reshape(b * s, self.input_dim), extra)
        mu, logvar = mu.reshape(b, s, d), logvar.reshape(b, s, d)
        return {'mu_a': mu, 'logvar_a': logvar, 'm_a': self.draw(mu, logvar)}


class OutcomeHead(nn.Module):
    """仅由 m_a 和参考表示生成结果 logits；没有 m_c 或当前输入接口。"""
    def __init__(self, mediator_dim: int, reference_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.3, num_classes: int = 2):
        super().__init__()
        self.mediator_dim, self.reference_dim = mediator_dim, reference_dim
        self.net = nn.Sequential(nn.Linear(mediator_dim + reference_dim, hidden_dim),
                                 nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, m_a: Tensor, reference: Tensor) -> Tensor:
        """接收已经显式对齐的 [...,D] 与 [...,d_ref]，不执行隐式广播。"""
        if m_a.shape[:-1] != reference.shape[:-1] or m_a.shape[-1] != self.mediator_dim or reference.shape[-1] != self.reference_dim:
            raise ValueError('结果模型的中介与参考形状不匹配')
        return self.net(torch.cat((m_a, reference), -1))


class ReferenceBank(nn.Module):
    """经验配对输入、[被试,trial,window]来源和数据集标识；不是学习原型。"""
    def __init__(self, x_c: Tensor, x_a: Tensor, source_ids: Tensor,
                 forbidden_subjects: tuple[int, ...] = (), max_size: Optional[int] = None,
                 seed: int = 0, detach_inputs: bool = True, dataset: Optional[str] = None):
        super().__init__()
        n = x_c.shape[0]
        if n < 1 or x_a.shape[0] != n or source_ids.shape != (n, 3):
            raise ValueError('参考输入必须配对，来源必须为 [N,3]')
        if source_ids.dtype != torch.long or torch.unique(source_ids, dim=0).shape[0] != n:
            raise ValueError('来源必须为互不重复的整数三元组')
        if any(torch.any(source_ids[:, 0] == subject) for subject in forbidden_subjects):
            raise ValueError('参考库含 validation/test 被试')
        if max_size is not None and max_size < 1:
            raise ValueError('参考库容量必须为正整数')
        limit = n if max_size is None else min(max_size, n)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:limit].to(x_c.device)
        self.available_count, self.used_count = n, limit
        self.shortfall = max_size is not None and n < max_size
        self.dataset = dataset
        c, a = x_c[indices], x_a[indices]
        # 训练 batch 的骨干特征保留参考路径梯度；评估缓存仍采用独立数据快照。
        self.register_buffer('x_c', (c.detach() if detach_inputs else c).clone())
        self.register_buffer('x_a', (a.detach() if detach_inputs else a).clone())
        self.register_buffer('source_ids', source_ids.to(x_c.device)[indices].detach().clone())

    def sample(self, batch_size: int, K: int, exclude_ids: Optional[Tensor] = None,
               excluded_subjects: tuple[int, ...] = ()) -> tuple[Tensor, Tensor, Tensor]:
        """逐当前样本有放回抽 K 次；可排除自身及指定来源被试。"""
        if batch_size < 1 or K < 1:
            raise ValueError('batch 和 K 必须为正整数')
        valid = torch.ones(batch_size, self.used_count, dtype=torch.bool, device=self.source_ids.device)
        if exclude_ids is not None:
            if exclude_ids.shape != (batch_size, 3):
                raise ValueError('待排除来源必须为 [B,3]')
            valid &= ~(exclude_ids.to(self.source_ids.device)[:, None, :] == self.source_ids[None]).all(-1)
        for subject in excluded_subjects:
            valid &= self.source_ids[None, :, 0] != subject
        if not valid.any(1).all():
            raise ValueError('排除来源后某个样本没有合法参考')
        indices = torch.multinomial(valid.float(), K, replacement=True)
        return self.x_c[indices], self.x_a[indices], self.source_ids[indices]


class FrontDoorEstimator(nn.Module):
    """对分层中介和经验联合参考分别边缘化；真实/合成共享唯一积分实现。"""
    def __init__(self, cns_dim: int, ans_dim: int, config: Optional[FrontDoorConfig] = None):
        super().__init__()
        self.config = config or FrontDoorConfig()
        cfg = self.config
        self.cns = CNSMediator(cns_dim, cfg)
        self.ans = ANSMediator(ans_dim, cfg)
        self.reference_encoder = ReferenceEncoder(cns_dim, ans_dim, cfg.ref_dim, cfg.ref_pool, cfg.seq_len)
        self.outcome = OutcomeHead(cfg.mediator_dim, self.reference_encoder.output_dim,
                                   cfg.hidden_dim, cfg.dropout, cfg.num_classes)

    def marginalize(self, m_a: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        """[B,S,D] 与 [B,K,d_ref] → [B,S,K,C] → [B,C]。"""
        if m_a.ndim != 3 or reference.ndim != 3 or m_a.shape[0] != reference.shape[0]:
            raise ValueError('中介与参考必须分别为 [B,S,D] 和 [B,K,d_ref]')
        b, s, d = m_a.shape
        k, r = reference.shape[1:]
        chunk = self.config.reference_chunk_size or k
        pieces = []
        for start in range(0, k, chunk):
            count = min(chunk, k - start)
            pieces.append(self.outcome(m_a[:, :, None, :].expand(b, s, count, d),
                          reference[:, None, start:start + count, :].expand(b, s, count, r)))
        logits = torch.cat(pieces, dim=2)
        probs = logits.softmax(-1).mean(2).mean(1)
        return probs, logits

    def forward(self, x_c: Tensor, x_a: Tensor, reference_bank: Optional[ReferenceBank] = None,
                source_ids: Optional[Tensor] = None, fd_mode: str = 'full',
                S: Optional[int] = None, K: Optional[int] = None,
                factual_ce_mode: str = 'single') -> dict[str, Tensor]:
        cfg = self.config
        S = (cfg.S_train if self.training else cfg.S_eval) if S is None else S
        K = (cfg.K_train if self.training else cfg.K_eval) if K is None else K
        if fd_mode not in ('full', 'none') or S < 1 or K < 1 or x_c.shape[0] != x_a.shape[0]:
            raise ValueError('模式、采样数或 batch 配对不合法')
        if factual_ce_mode not in ('single', 'all_samples'):
            raise ValueError('事实CE模式仅允许single或all_samples')
        result = self.cns(x_c, S)
        result.update(self.ans(x_a, result['m_c']))
        b = x_c.shape[0]
        own_reference = self.reference_encoder(x_c, x_a)
        # 默认保留原调用和随机流；可选对现有S组中介计算同一个事实CE期望。
        if factual_ce_mode == 'all_samples':
            logits_samples = self.outcome(result['m_a'], own_reference[:, None, :].expand(b, S, -1))
            result['logits_obs_samples'] = logits_samples
            logits_obs = logits_samples[:, 0]
        else:
            logits_obs = self.outcome(result['m_a'][:, 0], own_reference)
        result.update(logits_obs=logits_obs, probs_obs=logits_obs.softmax(-1))
        result['kl_c'] = gaussian_kl(result['mu_c'], result['logvar_c'])
        result['kl_a'] = gaussian_kl(result['mu_a'], result['logvar_a']).mean(1)
        if fd_mode == 'none':
            return result
        if self.training and (b < 2 or source_ids is None):
            raise ValueError('训练前门估计至少需要两个样本及真实来源索引')
        bank = reference_bank
        if bank is None:
            if not self.training or source_ids is None:
                raise ValueError('评估必须提供仅含训练数据的参考库')
            bank = ReferenceBank(x_c, x_a, source_ids, detach_inputs=False)
        rc, ra, ids = bank.sample(b, K, exclude_ids=source_ids)
        # 参考原始输入仅存数据；可学习 ReferenceEncoder 的梯度不会 detach。
        reference = self.reference_encoder(rc.reshape(b * K, *rc.shape[2:]),
                                            ra.reshape(b * K, *ra.shape[2:])).reshape(b, K, -1)
        probs, logits = self.marginalize(result['m_a'], reference)
        result.update(probs_do=probs, logits_do=probs.clamp_min(torch.finfo(probs.dtype).tiny).log(),
                      logits_combinations=logits, reference_indices=ids)
        return result


def gaussian_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    """特征 D 求和；保留 batch/采样轴，不在 S 轴重复 CNS KL。"""
    return 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1)


def frontdoor_loss(outputs: dict[str, Tensor], y: Tensor, config: FrontDoorConfig,
                   epoch: int = 1, oracle_probs: Optional[Tensor] = None) -> dict[str, Tensor]:
    """观察 CE + warmup KL；干预监督仅由显式 oracle 概率开启。"""
    if config.mediator_family != 'normal':
        raise ValueError('此标准正态 KL 接口仅用于高斯中介；tanh 合成拟合使用条件似然')
    logits_samples = outputs.get('logits_obs_samples')
    if logits_samples is None:
        obs = F.cross_entropy(outputs['logits_obs'], y.long())
    else:
        if logits_samples.ndim != 3 or min(logits_samples.shape) < 1:
            raise ValueError('事实采样logits必须为非空[B,S,C]')
        b, s, c = logits_samples.shape
        if c != config.num_classes or y.shape != (b,) or outputs['logits_obs'].shape != (b, c):
            raise ValueError('事实采样logits必须为[B,S,C]，标签为[B]')
        obs = F.cross_entropy(logits_samples.reshape(b * s, c), y.long().repeat_interleave(s))
    kl = (outputs['kl_c'] + outputs['kl_a']).mean()
    coefficient = config.beta * min(max(epoch, 0) / config.beta_warmup_epochs, 1.0)
    total = obs + coefficient * kl
    result = {'loss_obs': obs, 'loss_kl': kl}
    if oracle_probs is not None:
        if oracle_probs.shape != outputs['probs_do'].shape or torch.any(oracle_probs < 0) or not torch.allclose(oracle_probs.sum(-1), torch.ones_like(oracle_probs[:, 0])):
            raise ValueError('oracle 必须为归一化真实干预概率，不能是随机标签')
        oracle = -(oracle_probs * outputs['logits_do']).sum(-1).mean()
        result['loss_do_oracle'] = oracle
        total = total + oracle
    result['loss'] = total
    return result
