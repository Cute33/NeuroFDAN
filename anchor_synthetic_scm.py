"""锚点机制对齐/遗漏信息的配对SCM；保持原有节点、边和独立外生噪声。"""
import numpy as np
import torch
from scipy.special import expit, ndtr

from physiological_anchors import extract_anchors
from synthetic_scm import SyntheticSCM, SCMConfig


def render_coefficients(x_c: np.ndarray, x_a: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """16/8个有符号高斯系数→1个EEG、2个波形及1个GSR通道。

    各系数对应独立基函数，映射单射；只是合成信号，不声称模拟真实EEG/EMG。
    模型仍读取全部系数而非锚点，故错配机制不被锚点预先截断。
    """
    if x_c.ndim != 2 or x_a.ndim != 2 or len(x_c) != len(x_a) or x_c.shape[1] != 16 or x_a.shape[1] != 8:
        raise ValueError('合成渲染只接受配对的16维CNS和8维ANS系数')
    t = np.arange(768)/128
    frequencies = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 18, 22, 30, 38])
    basis_c = np.sqrt(2)*np.cos(2*np.pi*frequencies[:, None]*t)
    basis_a = np.zeros((8, 3, 768))
    for j, f in enumerate((10, 20, 35)):
        basis_a[j, 0] = np.sqrt(2)*np.cos(2*np.pi*f*t)
        basis_a[j+3, 1] = np.sqrt(2)*np.cos(2*np.pi*(f+1)*t)
    basis_a[6, 2] = 1
    basis_a[7, 2] = np.sqrt(2)*np.cos(2*np.pi*(1/3)*t)
    c = np.asarray(x_c @ basis_c, dtype=np.float32)[:, None]
    a = np.asarray(np.einsum('nd,dct->nct', x_a, basis_a), dtype=np.float32)
    return torch.from_numpy(c), torch.from_numpy(a)


def coefficient_anchors(x_c: np.ndarray, x_a: np.ndarray, variant: str = 'original') -> dict:
    """实际调用同一个频谱/波形提取器；合成目标不读取真实M、U或Y。"""
    if variant == 'original':
        return extract_anchors(*render_coefficients(x_c, x_a), 'DEAP')
    if variant == 'scale_stable':
        from anchor_invariance_candidates import extract_scale_stable
        return extract_scale_stable(*render_coefficients(x_c, x_a), 'DEAP')
    raise ValueError('未知合成锚点版本')


class AnchorSyntheticSCM(SyntheticSCM):
    """对齐组用观测锚点决定中介位置；错配组沿用原有有符号系数机制。

    两组都遵守S1，但观测Y分布不同；只允许在各机制内部比较三种监督。
    对齐组是有利的工作假设，不是对DEAP/HCI真实中介的事实断言。
    """
    def __init__(self, mechanism: str = 'aligned', seed: int = 2026, anchor_variant: str = 'original'):
        if mechanism not in ('aligned', 'misaligned'):
            raise ValueError('未知锚点机制')
        super().__init__(SCMConfig(seed=seed))
        self.mechanism = mechanism
        if anchor_variant not in ('original','scale_stable'):
            raise ValueError('未知锚点版本')
        self.anchor_variant = anchor_variant

    def mediator_inputs(self, xc, xa):
        if self.mechanism == 'misaligned':
            return xc, xa
        anchors = coefficient_anchors(xc, xa, self.anchor_variant)
        # 固定的生成结构映射，不使用任何训练/测试统计或标签。
        # 复制到原维度使Bc/Ba形状不变；锚点维度不是新SCM节点。
        def expand(value, dim):
            value = np.tanh(value.numpy().astype(np.float64)/3)
            return np.tile(value, (1, (dim+value.shape[1]-1)//value.shape[1]))[:, :dim]
        return expand(anchors['c'], 16), expand(anchors['a'], 8)

    def sample(self, n, gamma, seed):
        c, rng = self.config, np.random.default_rng(seed)
        u = rng.normal(size=(n, c.d_u))
        xc = gamma*u@self.A_c.T+rng.normal(size=(n, c.d_xc))
        xa = gamma*u@self.A_a.T+rng.normal(size=(n, c.d_xa))
        zc, za = self.mediator_inputs(xc, xa)
        mc = np.tanh(zc@self.B_c.T+rng.normal(size=(n, c.d_mc)))
        ma = np.tanh(za@self.B_a.T+mc@self.C_ca.T+rng.normal(size=(n, c.d_ma)))
        p = self.outcome_given_u(ma, u, gamma)
        return dict(x_c=xc, x_a=xa, m_c=mc, m_a=ma, u=u,
                    y=(ndtr(rng.normal(size=n)) < p).astype(np.int64))

    def direct_intervention(self, xc, xa, gamma, n, seed, chunk=4096):
        """独立直接生成：固定联合X、重采边缘U和中介噪声。"""
        self.validate_points(xc, xa)
        if n < 2 or chunk < 1:
            raise ValueError('直接积分规模无效')
        zc, za = self.mediator_inputs(xc, xa)
        rng, c = np.random.default_rng(seed), self.config
        means, ses = [], []
        for pc, pa in zip(zc, za):
            total = square = 0.
            for start in range(0, n, chunk):
                size = min(chunk, n-start)
                u = rng.normal(size=(size, c.d_u))
                mc = np.tanh(pc@self.B_c.T+rng.normal(size=(size, c.d_mc)))
                ma = np.tanh(pa@self.B_a.T+mc@self.C_ca.T+rng.normal(size=(size, c.d_ma)))
                p = self.outcome_given_u(ma, u, gamma)
                total += p.sum(); square += p@p
            mean = total/n
            means.append(mean); ses.append(np.sqrt(max(0, square-n*mean**2)/(n-1)/n))
        return np.array(means), np.array(ses)

    def frontdoor_integral(self, xc, xa, gamma, S, K, seed, chunk=64):
        """独立S9：采参考X'，由U|X'计算条件结果；不调用直接干预函数。"""
        self.validate_points(xc, xa)
        if min(S, K, chunk) < 1:
            raise ValueError('前门积分规模无效')
        rng, c = np.random.default_rng(seed), self.config
        u = rng.normal(size=(K, c.d_u))
        rc = gamma*u@self.A_c.T+rng.normal(size=(K, c.d_xc))
        ra = gamma*u@self.A_a.T+rng.normal(size=(K, c.d_xa))
        posterior, covariance = self.posterior_u(rc, ra, gamma)
        offset = gamma*posterior@self.w_u+self.b
        spread = np.sqrt(2*gamma**2*(self.w_u@covariance@self.w_u))
        zc, za = self.mediator_inputs(xc, xa)
        estimates, errors = [], []
        for pc, pa in zip(zc, za):
            mc = np.tanh(pc@self.B_c.T+rng.normal(size=(S, c.d_mc)))
            ma = np.tanh(pa@self.B_a.T+mc@self.C_ca.T+rng.normal(size=(S, c.d_ma)))
            rows, col = [], np.zeros(K)
            for start in range(0, S, chunk):
                eta = (ma[start:start+chunk]@self.w_a)[:, None]+offset[None]
                p = expit(eta[:, :, None]+spread*self.nodes)@self.weights
                rows.extend(p.mean(1)); col += p.sum(0)
            rows = np.asarray(rows)
            variance = rows.var(ddof=1)/S if S > 1 else 0.
            variance += (col/S).var(ddof=1)/K if K > 1 else 0.
            estimates.append(rows.mean()); errors.append(np.sqrt(variance))
        return np.array(estimates), np.array(errors)
