"""独立 SCM 生成器；直接干预与前门积分使用不同随机流和积分路径。"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

Array = NDArray[np.float64]


@dataclass
class SCMConfig:
    """矩阵按 N(0,scale²/输入维度) 初始化，外生高斯方差均为一。"""
    d_u: int = 8
    d_xc: int = 16
    d_xa: int = 8
    d_mc: int = 64
    d_ma: int = 64
    matrix_scale: float = 0.7
    cns_ans_scale: float = 0.7
    outcome_scale: float = 1.0
    bias: float = 0.2
    seed: int = 2026


class SyntheticSCM:
    """唯一图：U→X,Y；X_c→M_c；X_a,M_c→M_a；M_a→Y。"""
    def __init__(self, config: SCMConfig = SCMConfig()):
        self.config = config
        c = config
        if min(c.d_u, c.d_xc, c.d_xa, c.d_mc, c.d_ma) < 1 or min(c.matrix_scale, c.cns_ans_scale, c.outcome_scale) <= 0:
            raise ValueError('维度和结构系数尺度必须为正')
        rng = np.random.default_rng(c.seed)
        def matrix(rows: int, cols: int, scale: float) -> Array:
            return rng.normal(0, scale / np.sqrt(cols), (rows, cols))
        self.A_c = matrix(c.d_xc, c.d_u, c.matrix_scale)
        self.A_a = matrix(c.d_xa, c.d_u, c.matrix_scale)
        self.B_c = matrix(c.d_mc, c.d_xc, c.matrix_scale)
        self.B_a = matrix(c.d_ma, c.d_xa, c.matrix_scale)
        self.C_ca = matrix(c.d_ma, c.d_mc, c.cns_ans_scale)
        self.w_a = matrix(1, c.d_ma, c.outcome_scale).ravel()
        self.w_u = matrix(1, c.d_u, c.outcome_scale).ravel()
        self.b = c.bias
        self.nodes, self.weights = np.polynomial.hermite.hermgauss(32)
        self.weights /= np.sqrt(np.pi)

    def save(self, path: Path) -> None:
        """保存全部实际生成参数，而不仅保存种子。"""
        np.savez(path, **{k: getattr(self, k) for k in ('A_c', 'A_a', 'B_c', 'B_a', 'C_ca', 'w_a', 'w_u', 'b')})

    def sample(self, n: int, gamma: float, seed: int) -> dict[str, Array]:
        """每个外生量独立抽样；eps_y 的正态 CDF 提供 Bernoulli 均匀噪声。"""
        from scipy.special import ndtr
        c, rng = self.config, np.random.default_rng(seed)
        u = rng.normal(size=(n, c.d_u))
        xc = gamma * (u @ self.A_c.T) + rng.normal(size=(n, c.d_xc))
        xa = gamma * (u @ self.A_a.T) + rng.normal(size=(n, c.d_xa))
        mc = np.tanh(xc @ self.B_c.T + rng.normal(size=(n, c.d_mc)))
        ma = np.tanh(xa @ self.B_a.T + mc @ self.C_ca.T + rng.normal(size=(n, c.d_ma)))
        probability = expit(ma @ self.w_a + gamma * (u @ self.w_u) + self.b)
        y = (ndtr(rng.normal(size=n)) < probability).astype(np.int64)
        return {'x_c': xc, 'x_a': xa, 'm_c': mc, 'm_a': ma, 'u': u, 'y': y}

    def outcome_given_u(self, ma: Array, u: Array, gamma: float) -> Array:
        """结构结果方程只读取 M_a、U；无 M_c 输入。"""
        return expit(ma @ self.w_a + gamma * (u @ self.w_u) + self.b)

    def posterior_u(self, xc: Array, xa: Array, gamma: float) -> tuple[Array, Array]:
        """联合高斯观测模型给出的 U|X 后验，不使用中介或结果标签。"""
        a = np.concatenate((self.A_c, self.A_a), axis=0)
        covariance = np.linalg.inv(np.eye(self.config.d_u) + gamma ** 2 * a.T @ a)
        mean = np.concatenate((xc, xa), axis=-1) @ (gamma * a) @ covariance
        return mean, covariance

    def conditional_outcome(self, ma: Array, xc: Array, xa: Array, gamma: float) -> Array:
        """p(Y=1|M_a,X)；对后验 U 的一维结果投影作高斯求积。"""
        mean, covariance = self.posterior_u(xc, xa, gamma)
        location = ma @ self.w_a + gamma * (mean @ self.w_u) + self.b
        variance = gamma ** 2 * (self.w_u @ covariance @ self.w_u)
        return expit(location[..., None] + np.sqrt(2 * variance) * self.nodes) @ self.weights

    def direct_intervention(self, xc: Array, xa: Array, gamma: float,
                            n: int, seed: int, chunk: int = 4096) -> tuple[Array, Array]:
        """直接 do 真值：独立重采边缘 U；返回均值与 Monte Carlo 标准误。"""
        self.validate_points(xc, xa)
        c, rng = self.config, np.random.default_rng(seed)
        if n < 2 or chunk < 1:
            raise ValueError('直接积分至少需要两次采样，分块大小必须为正')
        estimates, errors = [], []
        for point_c, point_a in zip(xc, xa):
            total = total_square = 0.0
            for start in range(0, n, chunk):
                size = min(chunk, n - start)
                u = rng.normal(size=(size, c.d_u))
                mc = np.tanh(point_c @ self.B_c.T + rng.normal(size=(size, c.d_mc)))
                ma = np.tanh(point_a @ self.B_a.T + mc @ self.C_ca.T + rng.normal(size=(size, c.d_ma)))
                # 对结果 Bernoulli 噪声解析取期望，降低真值方差。
                values = self.outcome_given_u(ma, u, gamma)
                total += values.sum()
                total_square += values @ values
            mean = total / n
            estimates.append(mean)
            errors.append(np.sqrt(max(0, total_square - n * mean ** 2) / (n - 1) / n))
        return np.array(estimates), np.array(errors)

    def frontdoor_integral(self, xc: Array, xa: Array, gamma: float, S: int, K: int,
                           seed: int, chunk: int = 64) -> tuple[Array, Array]:
        """独立 S9 积分：中介 S 次、联合参考 K 次，内层使用 U|X' 后验。"""
        self.validate_points(xc, xa)
        if min(S, K, chunk) < 1:
            raise ValueError('S、K 和分块大小必须为正整数')
        c, rng = self.config, np.random.default_rng(seed)
        # 内层只生成观测 X'，不从边缘 U 直接计算结果。
        u_ref = rng.normal(size=(K, c.d_u))
        rc = gamma * (u_ref @ self.A_c.T) + rng.normal(size=(K, c.d_xc))
        ra = gamma * (u_ref @ self.A_a.T) + rng.normal(size=(K, c.d_xa))
        posterior, covariance = self.posterior_u(rc, ra, gamma)
        offsets = gamma * (posterior @ self.w_u) + self.b
        spread = np.sqrt(2 * gamma ** 2 * (self.w_u @ covariance @ self.w_u))
        estimates, errors = [], []
        for point_c, point_a in zip(xc, xa):
            # 与 direct_intervention 独立实现和独立随机流，保留 tanh 条件机制。
            mc = np.tanh(point_c @ self.B_c.T + rng.normal(size=(S, c.d_mc)))
            ma = np.tanh(point_a @ self.B_a.T + mc @ self.C_ca.T + rng.normal(size=(S, c.d_ma)))
            rows, col_sum = [], np.zeros(K)
            for start in range(0, S, chunk):
                eta = (ma[start:start + chunk] @ self.w_a)[:, None] + offsets[None, :]
                probabilities = expit(eta[:, :, None] + spread * self.nodes) @ self.weights
                rows.extend(probabilities.mean(1))
                col_sum += probabilities.sum(0)
            rows = np.array(rows)
            estimates.append(rows.mean())
            # 行/列方差给出保守的一阶两向积分误差估计，非严格覆盖率保证。
            variance = rows.var(ddof=1) / S if S > 1 else 0
            variance += (col_sum / S).var(ddof=1) / K if K > 1 else 0
            errors.append(np.sqrt(variance))
        return np.array(estimates), np.array(errors)

    def validate_points(self, xc: Array, xa: Array) -> None:
        """防止 zip 截断或广播悄悄改变联合干预点。"""
        if xc.ndim != 2 or xa.ndim != 2 or xc.shape[0] < 1 or xc.shape[0] != xa.shape[0]:
            raise ValueError('干预点必须为样本数相同的两个非空二维矩阵')
        if xc.shape[1] != self.config.d_xc or xa.shape[1] != self.config.d_xa:
            raise ValueError('干预点维度与 SCM 不一致')
        if not np.isfinite(xc).all() or not np.isfinite(xa).all():
            raise ValueError('干预点必须为有限数值')


def probability_metrics(prediction: Array, truth: Array) -> dict[str, float]:
    """Brier 为真实干预概率下的期望二分类 Brier；另报可约概率 MSE。"""
    q = np.clip(prediction, 1e-7, 1 - 1e-7)
    bins = np.minimum((q * 10).astype(int), 9)
    calibration = sum(np.mean(bins == k) * abs(q[bins == k].mean() - truth[bins == k].mean())
                      for k in np.unique(bins))
    return {'mae': float(np.mean(abs(q - truth))),
            'probability_mse': float(np.mean((q - truth) ** 2)),
            'brier': float(np.mean((q - truth) ** 2 + truth * (1 - truth))),
            'nll': float(np.mean(-truth * np.log(q) - (1 - truth) * np.log1p(-q))),
            'calibration_10bins': float(calibration)}
