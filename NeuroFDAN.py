import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def intermediate_forward(self, x, edge_index=None, layer_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature_list(self, x, edge_index=None):
        out_list = []
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out_list.append(x)
        x = self.lins[-1](x)
        return x, out_list


# ==================== 1. 数据预处理模块 ====================
class DilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, bias=True):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )
    def forward(self, x):
        """
            Input x:    [B, C_in, T]
            Output:     [B, C_out, T]
        """
        # Manually apply left padding
        x = F.pad(x, (self.padding, 0))
        # Perform convolution with no padding
        conv_output = self.conv1d(x)
        return conv_output


class ResidualCausalTemporalBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size, dilation_rate):
        super().__init__()
        # 如果输入和输出通道数不同，使用 1x1 卷积进行匹配
        self.downsample = nn.Conv1d(in_channels, hidden_dim, kernel_size=1) if in_channels != hidden_dim else None
        # 定义主路径
        self.main_path = nn.Sequential(
            DilatedCausalConv1d(in_channels, hidden_dim, kernel_size, dilation_rate),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.5)
        )
        
    def forward(self, x):  # x: [B, in_channels, T]
        residual = x
        # 通过主路径进行卷积操作
        out = self.main_path(x)
        # 如果需要，调整残差的维度
        if self.downsample is not None:
            residual = self.downsample(residual)
        # 将主路径输出与残差相加
        out += residual
        return out


class DeepTemporalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, kernel_size, dilation_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation_rate = 2 ** i
            in_c = in_channels if i == 0 else hidden_dim
            layers.append(
                ResidualCausalTemporalBlock(
                    in_channels=in_c,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate
                )
            )
        self.encoder_blocks = nn.Sequential(*layers)
        
    def forward(self, x):  # x: [B, in_channels, T]
        # 经过多个因果空洞残差块
        x = self.encoder_blocks(x)
        return x


# ==============================================================
# ==================== 2. GNN 因果图学习模块 ====================
# ==============================================================
class GNNGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # 定义 W_query, W_key, W_value
        self.W = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(1, num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.zeros(1, num_heads, out_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        # 拟合非线性结构方程 f 的 MLP
        self.structural_func = nn.Sequential(
            nn.Linear(num_heads * out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

        # 初始化参数
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        """0
            h: [B, N, in_dim]
            adj: [N, N] 或 [B, N, N] (学习到的 DAG 矩阵，值在 0-1 之间)
        """
        B, N, _ = h.shape

        # 1. 线性变换并分头 [B, N, heads, out_dim]
        h_prime = self.W(h).view(B, N, self.num_heads, self.out_dim)

        # 2. 计算注意力分数 (GAT 逻辑)
        # score_src: [B, N, heads, 1]
        score_src = (h_prime * self.a_src).sum(dim=-1, keepdim=True)
        score_dst = (h_prime * self.a_dst).sum(dim=-1, keepdim=True)

        # scores: [B, N, N, heads] -> 表示 i 对 j 的原始注意力
        score_src_reshaped = score_src.permute(0, 1, 3, 2)  # [B, N, 1, heads]
        score_dst_reshaped = score_dst.permute(0, 3, 1, 2)  # [B, 1, N, heads]
        # 广播相加: [B, N, 1, heads] + [B, 1, N, heads] -> [B, N, N, heads]
        scores = score_src_reshaped + score_dst_reshaped
        scores = self.leaky_relu(scores)

        # 3. 【关键步骤】引入 DAG 结构约束 (Masking)
        # adj 形状调整为 [B, N, N, 1] 以广播到所有 head
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)  # [1, N, N]
        adj_mask = adj.unsqueeze(-1)  # [B, N, N, 1]

        # 如果 adj 值接近 0 (无边)，则将注意力分数设为负无穷
        # 这样 Softmax 后权重为 0，实现了 "DAG 约束"
        zero_vec = -1e9 * torch.ones_like(scores)
        # 使用 where: 如果 adj > 0.01 (有边), 保留 scores, 否则设为负无穷
        attention = torch.where(adj_mask > 0.01, scores, zero_vec)

        # 4. Softmax 归一化得到动态因果强度
        attention = F.softmax(attention, dim=2)  # 对邻居维度归一化
        attention = self.dropout(attention)

        # 5. 加权聚合: H_new = Attention * H_old
        # [B, N, N, heads] * [B, N, heads, out_dim] -> [B, N, heads, out_dim]
        # 注意维度匹配，这里需要一点 trick
        h_prime = h_prime.unsqueeze(1)  # [B, 1, N, heads, out_dim]
        attention = attention.unsqueeze(-1)  # [B, N, N, heads, 1]
        # 聚合：sum over neighbor dimension (dim 2)
        out = (attention * h_prime).sum(dim=2)  # [B, N, heads, out_dim]

        # 6. 拼接多头并经过 MLP (模拟任意函数 f)
        out = out.view(B, N, self.num_heads * self.out_dim)
        out = self.structural_func(out)

        # 残差连接
        if self.in_dim == self.out_dim:
            out = out + h

        return out


class DAGGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 可学习的邻接矩阵，代表因果图
        self.W = nn.Parameter(torch.rand(num_nodes, num_nodes))

        self.dag_layer1 = GNNGATLayer(in_dim, hidden_dim)
        self.dag_layer2 = GNNGATLayer(hidden_dim, out_dim)
        
        self.out_layer = nn.Linear(out_dim, out_dim)

    def _get_adj(self):
        """生成有向无环图的邻接矩阵 A"""
        # gumbel_softmax 可以生成一个稀疏的近似二值化的邻接矩阵
        A = F.gumbel_softmax(self.W, tau=1.0, hard=False, dim=-1)
        A = A * (1 - torch.eye(self.num_nodes, device=A.device))
        return A
    
    def _dag_penalty_robust(self, A):
        """
            Args:
                A (Tensor): 邻接矩阵，形状 [N, N]
            Returns:
                Tensor: DAG 惩罚项的值
        """
        N = A.size(0)
        # 确保对角线元素为0，因为节点不能指向自己
        A = A * (1 - torch.eye(N, device=A.device))
        # 一个更平滑的矩阵指数近似
        # 使用多项式逼近 exp(x) ≈ 1 + x + x^2/2! + ... + x^k/k!
        A_power = A.clone()
        # 迹的惩罚项: sum_{k=1}^N tr((I + A)^k) - N^2
        # 我们只关注 A 的幂，因为对角线元素已经为0
        penalty = torch.trace(A_power)
        for k in range(2, N + 1):
            A_power = torch.matmul(A_power, A)
            penalty += torch.trace(A_power)
        return penalty

    def forward(self, x):
        """
            Input Shape: [B, N, in_dim]
        """
        # Step 1: 学习结构 G (DAG)
        adj = self._get_adj()

        # Step 2: 学习机制 f (SCM Structural Equations)
        # 在 GNN 层上进行特征传播, 这里的传播严格受限于 adj 定义的路径
        h1 = F.relu(self.dag_layer1(x, adj))    # Shape [B, C, hidden_dim]
        h2 = F.relu(self.dag_layer2(h1, adj))   # Shape [B, C, out_dim]
                        
        gcn_out = h2
        
        dag_loss = self._dag_penalty_robust(adj)

        # return gcn_out, adj, torch.tensor(0.0, requires_grad=True)
        return gcn_out, adj, dag_loss

# ==============================================================
# ==============================================================
# ==============================================================



# ------------------ Surrogate spike autograd ------------------
# ------------------------ 代理尖峰自 梯度 -----------------------
class SurrogateSpike(torch.autograd.Function):
    """
        Forward: hard threshold 阶跃函数
        Backward: surrogate gradient scaled by causal_confidence (optional)
    """
    @staticmethod
    def forward(ctx, input, thresh, scale):
        # 输入 阈值 缩放因子 保存在context对象中
        ctx.save_for_backward(input, thresh, scale)
        # 尖峰神经元的硬阈值激活 前向传播中真实生物行为
        out = (input > thresh).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # 从context中恢复之前保存的张量
        input, thresh, scale = ctx.saved_tensors
        # fast sigmoid surrogate derivative
        beta = 5.0  # temperature param, can be tuned   温度参数，用于调整代理梯度的斜率，beta越大，梯度越陡峭，越接近阶跃函数行为
        # 调整输入，使其以阈值为中心，并由beta缩放
        x = (input - thresh) * beta
        # 快速计算simoid代理梯度。 在x接近0时（即膜电位接近阈值时）有较大的梯度，而在远离阈值时梯度迅速减小，模拟了膜电位越接近阈值越荣日产生尖峰的“易感性”
        surrogate_grad = beta / ((1.0 + x.abs()) ** 2)  # shape same as input
        # apply causal scaling (scale broadcastable to input)
        # 计算最终的输入梯度，将来自上一层的梯度 grad_output 与代理梯度和可选的 scale 因子相乘。
        # scale 参数在这里至关重要，它允许外部模块（例如 SpikingMediator）根据某些“置信度”来调整梯度，这被称为 因果代理（causal-surrogate）学习。
        grad_input = grad_output * surrogate_grad * scale
        # 返回每个前向传播输入对应的梯度，None表示 thresh 和 scale 不需要梯度计算，它们被视为常数
        return grad_input, None, None


surrogate_spike = SurrogateSpike.apply

# ------------------ Vectorized LIF layer ------------------
# ---------------------- 向量化LIF层 ------------------------
# 实现了向量化的LIF神经元模型，能同时处理一个batch中所有样本在每个时间步上所有神经元
# 模拟了膜电位的 整合 Integrate 和 泄露 Leak 过程，并根据 阈值 产生尖峰
class VectorizedLIFLayer(nn.Module):
    def __init__(self, num_neurons, in_dim, tau=2.0, thresh=1.0, dt=1.0, device='cpu'):
        """
            num_neurons: number of LIF units in this population (could be equal to C or multiple per channel)
            in_dim: dimensionality of input embedding (D)
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.in_dim = in_dim
        self.tau = tau
        self.alpha = torch.exp(torch.tensor(-dt / tau))   # 膜电位的衰减因子； tau 时间常数 dt 时间步长 alpha 膜电位每一步衰减多少
        self.thresh = nn.Parameter(torch.ones(num_neurons) * thresh * 0.2)  # learnable thresholds 可学习神经元阈值
        # map input embeddings to currents for each neuron
        # 将输入特征向量投影到与神经元数量相匹配的维度，生成 输入电流
        self.input_proj = nn.Linear(in_dim, num_neurons, bias=True)
        # optional neuron-specific bias
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        # reset mode implemented via subtracting thresh on spike
        # energy regularizer parameters
        self.device = device

    def forward(self, input_seq, hidden_v=None, adj_scale=None):
        """
            input_seq: [B, T, in_dim]  (if static embedding, T could be 1 and then broadcast)
            adj_scale: [B, T, num_neurons] or scalar for surrogate scaling (causal-surrogate)
            returns:
                spikes: [B, T, num_neurons] (0/1)
                V: [B, T, num_neurons] membrane potentials
                rates: [B, num_neurons] average firing rates over T
        """
        B, T, D = input_seq.shape
        device = input_seq.device
        V = torch.zeros(B, self.num_neurons, device=device)     # 初始化所有神经元的膜电位为 0
        spikes = []
        Vs = []
        for t in range(T):
            x_t = input_seq[:, t, :]            # [B, in_dim]
            I_t = self.input_proj(x_t) + self.bias  # [B, num_neurons]  计算当前时间步的输入电流
            # membrane update: V = alpha * V + (1 - alpha) * I  LIF膜电位更新方程：当前膜电位是上一刻膜电位的衰减(alpha*V)和新输入电流的累计((1-alpha)*I_t)
            V = self.alpha * V + (1.0 - self.alpha) * I_t
            # spike generation
            # prepare thresholds broadcast
            thresh = self.thresh.unsqueeze(0).expand(B, -1)  # [B, num_neurons]
            # causal scaling default 1.0
            if adj_scale is None:
                scale = torch.ones_like(V)
            else:
                # adj_scale shape flexible, ensure [B, num_neurons]
                scale = adj_scale[:, t, :] if adj_scale.dim() == 3 else adj_scale
            # 使用SurrogateSpike函数生成尖峰。在前向传播中使用硬阈值处理，同时为反向传播设置了代理梯度
            s_t = surrogate_spike(V, thresh, scale)  # [B, num_neurons], forward is hard threshold
            # reset by subtracting thresh on spikes
            # 尖峰后处置：如果神经元产生尖峰(s_t为1)，其膜电位会减去阈值，模拟LIF模型的硬重置
            V = V - s_t * thresh
            spikes.append(s_t)
            Vs.append(V)
        spikes = torch.stack(spikes, dim=1)  # [B, T, num_neurons]
        # 加入残差模块，缓解梯度消失
        # spikes += input_seq
        Vs = torch.stack(Vs, dim=1)          # [B, T, num_neurons]
        rates = spikes.mean(dim=1)           # [B, num_neurons] 计算每个神经元在整个时间序列上的平均放电率，这通常是用于下游任务的输出。        
        return spikes, Vs, rates

# ------------------ Spiking Mediator that uses adjacency confidence for causal surrogate scaling ------------------
# ------------------------------------------------ 因果前门型脉冲中介器 ----------------------------------------------
# 将来自GNN的静态特征(Embedding)转换为一个时间序列的 尖峰模式。
# 通过 VectorizedLIF 模拟神经元群体的动态行为，并利用 adj_confidience (相邻通道的连接置信度)作为 因果代理缩放因子
# 旨在让训练过程更加关注那些“可靠”的连接，增强模型的可解释性和性能
class SpikingMediator(nn.Module):
    def __init__(self, C_in, D_in, num_neurons, time_steps=32, pool_mode='mean', device='cpu'):
        """
            C_in: number of channels (e.g., EEG channels 32)
            D_in: embedding dim output by GNN per channel
            num_neurons: number of neurons in mediator population (can be >= C_in)
        """
        super().__init__()
        self.C_in = C_in
        self.D_in = D_in
        self.num_neurons = num_neurons
        self.time_steps = time_steps
        # project per-channel features to a time-sequence of currents
        # Option A: if gnn outputs per-time embedding [B,C,D,T], we can use directly
        # Here assume gnn_out is [B, C, D] per-window; we expand to time_steps
        self.channel_to_current = nn.Linear(D_in, num_neurons)
        # optionally a small temporal generator to expand static embedding into a sequence
        # 门控循环单元 用于将静态的输入特征扩展到一个时间序列。(将静态信息转换为动态输入以驱动SNN的常见方法)
        self.temporal_generator = nn.GRU(input_size=num_neurons, hidden_size=num_neurons, batch_first=True)
        self.lif = VectorizedLIFLayer(num_neurons=num_neurons, in_dim=num_neurons, tau=2.0, thresh=1.0, dt=1.0, device=device)
        self.pool_mode = pool_mode

    def forward(self, gnn_feats, adj_confidence=None):
        """
            gnn_feats: [B, C, D]  (per-channel embeddings)
            adj_confidence: adjacency confidence matrix [B, C, C] or [C,C]; used to build surrogate scale for neurons
            returns:
                spikes: [B, T, num_neurons]
                rates: [B, num_neurons]
                pooled_rate: [B, K] (aggregated groups as M)
        """
        B, C, D = gnn_feats.shape
        device = gnn_feats.device
        # aggregate channels into a population drive (simple sum or learned)
        # Option: sum across channels after projecting to neuron space
        # compute per-channel contribution to each neuron
        # 将每个通道的GNN特征映射到神经元空间
        per_chan = self.channel_to_current(gnn_feats.view(B*C, D))      # [B*C, num_neurons]
        per_chan = per_chan.view(B, C, self.num_neurons)                # [B, C, num_neurons]
        # combine channels: simple sum across channel dim -> drive per sample
        # 聚合所有通道的贡献，形成一个驱动整个神经元群体的总输入。
        drive = per_chan.sum(dim=1)   # [B, num_neurons]
        # expand drive into sequence via small GRU (repeat as initial input)
        # prepare seq_in: replicate drive T times as initial sequence input
        # 将聚合后的驱动输入复制 time_steps 次，作为GRU的初始输入序列
        seq_in = drive.unsqueeze(1).repeat(1, self.time_steps, 1)       # [B, T, num_neurons]
        # GRU将静态驱动转换为一个动态的、随时间变化的序列，以驱动LIF层
        seq_out, _ = self.temporal_generator(seq_in)                    # [B, T, num_neurons]
        # adj_confidence -> surrogate scaling: derive scale per neuron per timestep
        if adj_confidence is None:
            adj_scale = None
        else:
            # 只是占位符
            # reduce adjacency into neuron-level scale:
            # for example, take max incoming confidence per channel, map channels->neurons via per_chan weights
            # Here use simple heuristic: channel importance -> neuron scale
            chan_importance = adj_confidence.mean(dim=-1) if adj_confidence.dim()==3 else adj_confidence.mean(dim=0)
            # shape adjustments: make [B,T,num_neurons] full ones for now
            adj_scale = torch.ones(B, self.time_steps, self.num_neurons, device=device)
        spikes, Vs, rates = self.lif(seq_out, hidden_v=None, adj_scale=adj_scale)
        # pooled M: reduce neurons into K mediators if needed (e.g., K small)
        # Here simply return rates as M (you ca
        # 平均放电率
        pooled = rates  # [B, num_neurons]
        # shape [B, num_neurons, num_neurons], [B, num_neurons, num_neurons], [B, num_neurons]
        return spikes, Vs, rates, pooled



# 第二阶段唯一有效模型；上方历史基础组件保留供已有研究接口使用。
from frontdoor_scm import (
    FrontDoorConfig, CNSMediator, ANSMediator, ReferenceEncoder, OutcomeHead,
    ReferenceBank, FrontDoorEstimator, frontdoor_loss,
)


class PhysiologicalBackbone(nn.Module):
    """复用原实际执行的分组时序卷积及 CNS 通道图，两个模态不交叉读取。"""
    def __init__(self, eeg_channels=32, ans_channels=4, input_len=768):
        super().__init__()
        if eeg_channels != 32 or ans_channels not in (3, 4) or input_len != 768:
            raise ValueError('真实数据骨干要求32个EEG通道、3/4个ANS通道和768个时间点')
        self.ans_channels = ans_channels
        def encoder(channels):
            return nn.Sequential(
                nn.Conv1d(channels, channels * 4, 5, stride=2, padding=2, groups=channels),
                nn.BatchNorm1d(channels * 4), nn.SiLU(),
                nn.Conv1d(channels * 4, channels * 4, 3, stride=2, padding=1, groups=channels),
                nn.BatchNorm1d(channels * 4), nn.SiLU(),
                nn.Conv1d(channels * 4, channels * 4, 3, stride=2, padding=1, groups=channels),
                nn.BatchNorm1d(channels * 4), nn.SiLU())
        self.cns_enc, self.ans_enc = encoder(32), encoder(ans_channels)
        self.dag_adj = nn.Parameter(torch.eye(32) + .1 * torch.randn(32, 32))
        uniform = torch.rand(32, 32, generator=torch.Generator().manual_seed(42))
        self.register_buffer('eval_gumbel', -torch.log(-torch.log(uniform)))
        self.cns_dim, self.ans_dim = 96 * 128, 96 * ans_channels * 4
        self.train(True)

    def train(self, mode=True):
        """冻结既有 BN 运行统计，保留仿射参数，避免 batch 内样本相互泄漏。"""
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
        return self

    def forward(self, x_c, x_a):
        """[B,32,768]、[B,A,768] → [B,96,128]、[B,96,4A]，完整保留时序轴。"""
        b = x_c.shape[0]
        if x_c.shape != (b, 32, 768) or x_a.shape != (b, self.ans_channels, 768):
            raise ValueError('CNS/ANS窗口形状或配对关系错误')
        cns = self.cns_enc(x_c).reshape(b, 32, 4, 96)
        ans = self.ans_enc(x_a).reshape(b, self.ans_channels, 4, 96)
        if self.training:
            adjacency = F.gumbel_softmax(self.dag_adj, tau=1., hard=True)
        else:
            # 固定seed42的Gumbel扰动，保持原硬图采样形式且可缓存；不宣称必为DAG。
            # 不直接取权重argmax，否则初始单位阵去自环后会使全部CNS特征归零。
            adjacency = F.one_hot((self.dag_adj + self.eval_gumbel).argmax(-1), num_classes=32).to(cns)
        adjacency = adjacency * (1 - torch.eye(32, device=cns.device))
        cns = torch.matmul(cns.permute(0, 3, 2, 1), adjacency.t()).reshape(b, 96, 128)
        ans = ans.permute(0, 3, 1, 2).reshape(b, 96, self.ans_channels * 4)
        return cns, ans


class _LiveReferenceView:
    """只适配特征，不另写抽样：原始训练参考经当前骨干编码并保留梯度。"""
    def __init__(self, bank, backbone, device):
        self.bank, self.backbone, self.device = bank, backbone, device

    def sample(self, batch_size, K, exclude_ids=None):
        c, a, ids = self.bank.sample(batch_size, K, exclude_ids=exclude_ids)
        hc, ha = self.backbone(c.reshape(-1, *c.shape[2:]).to(self.device),
                               a.reshape(-1, *a.shape[2:]).to(self.device))
        return hc.reshape(batch_size, K, *hc.shape[1:]), ha.reshape(batch_size, K, *ha.shape[1:]), ids.to(self.device)


class NeuroFDAN(nn.Module):
    """真实骨干 + 唯一 Phase 1 前门估计器；不接受真实数据 oracle 标签。"""
    def __init__(self, eeg_channels=32, ecg_channels=3, gsr_channels=1, input_len=768,
                 hidden_dim=32, num_classes=2, frontdoor_config=None):
        super().__init__()
        if num_classes != 2:
            raise ValueError('第二阶段使用两个logits；不能对单个logit执行类别softmax')
        self.backbone = PhysiologicalBackbone(eeg_channels, ecg_channels + gsr_channels, input_len)
        self.frontdoor_config = frontdoor_config or FrontDoorConfig()
        if self.frontdoor_config.num_classes != 2 or self.frontdoor_config.mediator_family != 'normal':
            raise ValueError('真实数据模式使用双类别高斯潜在中介，不使用合成真值分布接口')
        # hidden_dim 保留旧构造参数兼容；旧实际骨干同样未使用它决定通道数。
        self.frontdoor = FrontDoorEstimator(self.backbone.cns_dim, self.backbone.ans_dim, self.frontdoor_config)
        self.feature_version = 0

    def forward(self, x_c=None, x_a=None, gsr=None, *, reference_bank=None,
                source_ids=None, fd_mode='full', S=None, K=None):
        """支持旧的 EEG/ECG/GSR 三位置参数；统一返回概率、事实logits和统计字典。"""
        if x_c is None or x_a is None:
            raise ValueError('必须提供配对的CNS和ANS输入')
        if gsr is not None:
            x_a = torch.cat((x_a, gsr), dim=1)
        raw_reference = reference_bank is not None and getattr(reference_bank, 'raw_windows', False)
        if reference_bank is not None and not raw_reference:
            if getattr(reference_bank, 'feature_version', None) != self.feature_version:
                raise ValueError('参考特征缓存已过期或未由当前模型构建，需按当前权重重建')
            if getattr(reference_bank, 'model_identity', None) != id(self):
                raise ValueError('参考库不属于当前fold模型')
            if reference_bank.parameter_versions != tuple(p._version for p in self.parameters()):
                raise ValueError('模型权重已更新，参考特征缓存必须重建')
        h_c, h_a = self.backbone(x_c, x_a)
        effective_bank = _LiveReferenceView(reference_bank, self.backbone, x_c.device) if raw_reference else reference_bank
        outputs = self.frontdoor(h_c, h_a, effective_bank, source_ids, fd_mode, S, K)
        rates_c = self.frontdoor.cns.snn.last_rates
        rates_a = self.frontdoor.ans.snn.last_rates
        fire_loss = .5 * ((rates_c.mean(0) - .05).square().mean()
                         + (rates_a.mean(0) - .05).square().mean())
        outputs['fire_loss'] = fire_loss
        outputs['logits_factual'] = outputs['logits_obs']
        outputs['mediator_statistics'] = {
            'mu_c': outputs['mu_c'].detach().mean(), 'var_c': outputs['logvar_c'].detach().exp().mean(),
            'mu_a': outputs['mu_a'].detach().mean(), 'var_a': outputs['logvar_a'].detach().exp().mean(),
            'kl_c': outputs['kl_c'].detach().mean(), 'kl_a': outputs['kl_a'].detach().mean(),
            'mu_c_spread': outputs['mu_c'].detach().std(dim=0, unbiased=False).mean(),
            'mu_a_spread': outputs['mu_a'].detach().flatten(0, 1).std(dim=0, unbiased=False).mean(),
            'firing_rate_c': rates_c.detach().mean(), 'firing_rate_a': rates_a.detach().mean(),
        }
        outputs['reference_statistics'] = {
            'S': outputs['m_a'].shape[1],
            'K': outputs['reference_indices'].shape[1] if 'reference_indices' in outputs else 0,
            'bank_count': 0 if fd_mode == 'none' else (reference_bank.used_count if reference_bank is not None else x_c.shape[0]),
            'factual_reference_count': x_c.shape[0],
            'shortfall': reference_bank.shortfall if reference_bank is not None else False,
            'dataset': reference_bank.dataset if reference_bank is not None else None,
        }
        return outputs

    def weights_updated(self):
        """优化器step或加载权重后显式失效旧缓存。"""
        self.feature_version += 1
