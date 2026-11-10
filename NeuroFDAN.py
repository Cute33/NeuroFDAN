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



# ========== 5. 前门调整模块 ==========
class FrontDoorInterventionModule(nn.Module):
    def __init__(self, x_dim, m_dim, proj_dim):
        super().__init__()
        # 添加投影层
        #self.x_proj = nn.Linear(x_dim, proj_dim)
        #self.m_proj = nn.Linear(m_dim, proj_dim)
        self.x_proj = nn.Sequential(
            nn.Linear(x_dim, x_dim // 2),
            nn.SiLU(),
            nn.Linear(x_dim // 2, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.Dropout(0.1)
        )
        self.m_proj = nn.Sequential(
            nn.Linear(m_dim, m_dim // 2),
            nn.SiLU(),
            nn.Linear(m_dim // 2, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.Dropout(0.1)
        )
        self.proj_dim = proj_dim

    def forward(self, x, m):
        """
        Args:
            x: 输入特征 X [batch_size, x_dim]
            m: 中介变量 M [batch_size, m_dim]
        """
        # 投影到相同维度
        x_proj = self.x_proj(x)  # [batch_size, proj_dim]
        m_proj = self.m_proj(m)  # [batch_size, proj_dim]

        # 扩展维度以便进行注意力计算
        x_proj = x_proj.unsqueeze(0)  # [1, batch_size, proj_dim]
        m_proj = m_proj.unsqueeze(0)  # [1, batch_size, proj_dim]

        # 第一步: 计算 P(M=m|X=x)
        attn_weights = torch.matmul(m_proj, x_proj.transpose(-2, -1)) / math.sqrt(self.proj_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        first_fusion = torch.matmul(attn_weights, x_proj)   # shape: [1, B, B]

        # 第二步: 计算 ∑_x' P(Y|X=x', M=m)P(X=x')
        mediator_attn = torch.matmul(x_proj, m_proj.transpose(-2, -1)) / math.sqrt(self.proj_dim)
        mediator_attn = F.softmax(mediator_attn, dim=-1)
        deconfounded_features = torch.matmul(mediator_attn, first_fusion)

        # 压缩维度
        deconfounded_features = deconfounded_features.squeeze(0)  # [batch_size, proj_dim]

        return deconfounded_features


# ==================== 5. 前门干预模块 ====================
class FrontDoorInterventionModule(nn.Module):
    def __init__(self, x_dim, m_dim, proj_dim):
        super().__init__()
        # 路径一：建模 P(M|X)
        self.x_to_m_path = nn.Sequential(
            nn.Linear(x_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, m_dim * 2),  # 输出均值和log_方差
        )

        # 路径二：建模 P(Y|M)
        self.m_to_y_proj = nn.Sequential(
            nn.Linear(m_dim, m_dim // 2),
            nn.SiLU(),
            nn.Linear(m_dim // 2, proj_dim),
        )
        self.proj_dim = proj_dim

    def forward(self, x, m_real):
        batch_size = x.size(0)
        
        # ----------------- Step 1: 建模 P(M|X) -----------------
        # 预测 M 的分布参数 （假设为高斯分布）
        m_params = self.x_to_m_path(x)
        mu, log_var = m_params.chunk(2, dim=1)
        log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var)
        
        # 从该分布中采样 M 来模拟干预 do(X)
        # 即前门机制中的 do(M)
        epsilon = torch.randn_like(mu)
        m_do = mu + epsilon * std

        # ----------------- Step 2: 建模 P(Y|M) 并进行求和 -----------------
        # 根据前门公式，需要对所有可能的 M 值求和，但在神经网络中使用采样来逼近
        # 用两个不同的 M 值来预测，然后取平均或注意力融合
        
        # 1. 原始路径的特征 （来自真实的 M）
        deconfounded_real = self.m_to_y_proj(m_real)
        # 2. 干预路径的特征 （来子采样的 M）
        deconfounded_do = self.m_to_y_proj(m_do)
        # 简单的求和/平均操作来逼近前门公式中的求和
        deconfounded_features = (deconfounded_real + deconfounded_do) / 2
        
        return deconfounded_features, m_do


class FrontDoorInterventionModule(nn.Module):
    def __init__(self, x_dim, m_dim, proj_dim):
        super().__init__()
        # 第1步: 建模 P(M|X) -> 预测M的分布参数
        self.x_to_m = nn.Sequential(
            nn.Linear(x_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, m_dim * 2),  # 输出均值和log_方差
        )
        
        # 第2步: 建模 P(Y|M) -> 从M到Y的映射，使用一个共享的投影层
        self.m_to_y = nn.Sequential(
            nn.Linear(m_dim, proj_dim),
            nn.SiLU(),
        )
        
        # 用一个简单的注意力机制来模拟求和/积分
        self.attention_net = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=4, batch_first=True
        )

    def forward(self, x, m_real):
        # ''
        #   x       [B, C_cns+C_ans]
        #   m_real  [B, C_cns+C_ans]
        # ''
        # --- Step 1: 建模 P(M|X) 和采样 ---
        # x_to_m 返回 mu 和 log_var
        m_params = self.x_to_m(x)
        mu, log_var = m_params.chunk(2, dim=1)
        log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var)
        
        # 重参数化采样
        epsilon = torch.randn_like(mu)
        m_do = mu + epsilon * std
        
        # --- Step 2: 建模 P(Y|M) 并使用注意力机制融合 ---
        # 1. 原始路径: 使用真实的M
        h_m_real = self.m_to_y(m_real)
        
        # 2. 干预路径: 使用采样的M
        h_m_do = self.m_to_y(m_do)
        
        # 3. 使用注意力机制模拟前门公式中的求和/积分
        # 这里，h_m_do 作为 Query，h_m_real 作为 Key 和 Value
        # 目标是让采样的 M 特征去“关注”真实的 M 特征
        h_m_do = h_m_do.unsqueeze(1)  # [B, 1, proj_dim]
        h_m_real = h_m_real.unsqueeze(1) # [B, 1, proj_dim]

        # 注意力机制的输入需要是 [batch_size, sequence_length, feature_dim]
        # 但我们这里只有1个序列长度，所以需要更巧妙地使用
        # 我们可以把 h_m_real 当作 keys 和 values，然后让 h_m_do 
        # 当作 query 去查询整个批次的 h_m_real
        
        # 让我们换一种更直观的注意力方式：
        # 计算 m_do 对所有 m_real 的关注度
        attn_scores = torch.matmul(h_m_do, h_m_real.transpose(-2, -1)) / math.sqrt(h_m_real.size(-1))
        
        # softmax 归一化注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 加权求和
        deconfounded_features = torch.matmul(attn_weights, h_m_real).squeeze(1) # [B, proj_dim]

        return deconfounded_features, m_do
   
 
# ========== 7. 主模型 ==========
class NeuroFDAN(nn.Module):
    def __init__(self, eeg_channels, ecg_channels, gsr_channels, input_len, hidden_dim, num_classes, device=device):
        super().__init__()
        self.cns_channel = eeg_channels
        self.ans_channel = ecg_channels + gsr_channels
        self.device = device
        
        # 编码器
        self.eeg_temporal_encoder = DeepTemporalEncoder(eeg_channels, eeg_channels, num_layers=1, kernel_size=5, dilation_rate=5)
        self.ecg_temporal_encoder = DeepTemporalEncoder(ecg_channels, ecg_channels, num_layers=1, kernel_size=3, dilation_rate=5)
        self.gsr_temporal_encoder = DeepTemporalEncoder(gsr_channels, gsr_channels, num_layers=1, kernel_size=3, dilation_rate=5)
        
        # GNN 因果图学习模块
        self.eeg_dag_gnn = DAGGNN(
            in_dim=input_len, 
            hidden_dim=hidden_dim, 
            out_dim=hidden_dim, 
            num_nodes=eeg_channels
        )
        self.ans_dag_gnn = DAGGNN(
            in_dim=input_len, 
            hidden_dim=hidden_dim, 
            out_dim=hidden_dim, 
            num_nodes=ecg_channels + gsr_channels
        )
        
        self.cns = SpikingMediator(self.cns_channel, hidden_dim, self.cns_channel, time_steps=128, device=self.device)
        self.ans = SpikingMediator(self.ans_channel, hidden_dim, self.ans_channel, time_steps=128, device=self.device)

        # 前门干预模块
        self.fim = FrontDoorInterventionModule(
            x_dim=self.cns_channel + self.ans_channel,
            m_dim=self.cns_channel + self.ans_channel,
            proj_dim=hidden_dim
        )
        
        def init_last_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # 情绪分类器
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.emotion_classifier.apply(init_last_layer)

        
        self.mlp = nn.Sequential(
            nn.Linear(96, 96 // 2),
            nn.SiLU(),
            nn.BatchNorm1d(96 // 2),
            nn.Dropout(p=0.5),
            nn.Linear(96 // 2, num_classes),
        )
        self.emotion_classifier.apply(init_last_layer)


    def forward(self, eeg, ecg, gsr):
        # Inpute Shape: [B, C, T]

        # 1. 通道时序编码
        h_eeg = self.eeg_temporal_encoder(eeg) # [B, C_eeg, T]
        h_ecg = self.ecg_temporal_encoder(ecg) # [B, C_ecg, T]
        h_gsr = self.gsr_temporal_encoder(gsr) # [B, C_gsr, T]

        # 2. GNN 因果图学习
        eeg_gcn_out, cns_adj, eeg_dag_penalty = self.eeg_dag_gnn(h_eeg)             # [B, C_cns, D], [C_cns, C_cns], loss
        h_ans = torch.cat([h_ecg, h_gsr], dim=1)                                    # [B, C_ans, T]
        ans_gcn_out, ans_adj, ans_dag_penalty = self.ans_dag_gnn(h_ans)             # [B, C_ans, D], [C_ans, C_ans], loss
        
        X = torch.cat([eeg_gcn_out, ans_gcn_out], dim=1)                            # [B, C_cns+C_ans, D]
        X_mean = X.mean(dim=1)  # [B, D] 对通道平均
        emotion_logits = self.emotion_classifier(X_mean)
        return emotion_logits, eeg_dag_penalty, ans_dag_penalty

        # 3. 神经系统潜变量
        cns_spikes, cns_Vs, cns_rates, cns_pooled = self.cns(eeg_gcn_out, None)     # [B, C_cns, C_cns], [B, C_cns, C_cns], [B, C_cns]
        ans_spikes, ans_Vs, ans_rates, ans_pooled = self.ans(ans_gcn_out, None)     # [B, C_ans, C_ans], [B, C_ans, C_ans], [B, C_ans]
        M = torch.cat([cns_pooled, ans_pooled], dim=1)                              # [B, C_cns+C_ans]
        # return self.mlp(M), eeg_dag_penalty, ans_dag_penalty
        # return self.mlp(X), eeg_dag_penalty, ans_dag_penalty

        # 4. 前门干预
        X = X.mean(dim=2)
        
        deconfounded_features, m_do = self.fim(X, M)
        
        # 情绪分类
        emotion_logits = self.emotion_classifier(deconfounded_features)

        return emotion_logits, eeg_dag_penalty, ans_dag_penalty









class CausalMemoryBank(nn.Module):
    def __init__(self, input_dim, num_prototypes=64, tau=0.1):
        """
        Args:
            input_dim: 输入特征维度 (D)
            num_prototypes: 记忆库大小 (K), 代表 K 种典型的神经活动模式
            tau: 温度系数，控制 Attention 的稀疏度
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.input_dim = input_dim
        self.tau = tau

        # Memory G: [K, D] -> 对应前门公式中的 P(X) 离散化表示
        # 这是一个可学习的“字典”，存储跨被试共享的神经原型
        self.memory = nn.Parameter(torch.randn(num_prototypes, input_dim))

        # 投影层
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        x: SNN 输出的特征 [B, D] (即 Query)
        Returns:
            m_causal: 经过前门干预后的特征
            attn: P(M|X) 的概率分布
        """
        # 1. 计算 P(M|X) -> Attention Weights
        # Query:    当前样本的特征
        Q = self.q_proj(x)  # [B, D]
        # Key:      记忆库中的原型
        K = self.k_proj(self.memory)  # [K, D]

        # 计算相似度 logits: [B, K]
        scores = torch.matmul(Q, K.t()) / self.tau

        # Softmax 归一化得到概率分布 P(M|X)
        attn_weights = F.softmax(scores, dim=-1)  # [B, K]

        # 2. 前门干预 (NWGM 近似)
        # Value: 记忆库中的原型值
        V = self.v_proj(self.memory)  # [K, D]

        # 重构 M: 使用“纯净”的记忆库原型重组特征
        # 这阻断了 X -> Y 的直接路径 (及其背后的 U -> X -> Y)
        # 强制模型使用通用原型来表达当前样本
        m_causal = torch.matmul(attn_weights, V)  # [B, D]

        # 辅助Loss
        rec_loss = F.mse_loss(m_causal, x)
        mean_usage = attn_weights.mean(dim=0)
        div_loss = -torch.sum(mean_usage * torch.log(mean_usage + 1e-8))

        return m_causal, attn_weights, rec_loss, div_loss
        # 这是之前71ACC版本，没有别的损失的
        return m_causal, attn_weights


class NeuroFDAN(nn.Module):
    def __init__(self, eeg_channels, ecg_channels, gsr_channels, input_len, hidden_dim, num_classes=1):
        super().__init__()
        self.eeg_channels, self.ans_channels = eeg_channels, ecg_channels + gsr_channels

        # 1. Efficient Encoders (Time: 768 -> 96, Feat: 1 -> 4)
        # Grouped conv ensures channel independence for DAG
        self.cns_enc = nn.Sequential(
            nn.Conv1d(self.eeg_channels, self.eeg_channels * 4, kernel_size=5, stride=2, padding=2, groups=32),  # 384
            nn.BatchNorm1d(32 * 4), nn.SiLU(),
            nn.Conv1d(32 * 4, 32 * 4, kernel_size=3, stride=2, padding=1, groups=32),  # 192
            nn.BatchNorm1d(32 * 4), nn.SiLU(),
            nn.Conv1d(32 * 4, 32 * 4, kernel_size=3, stride=2, padding=1, groups=32),  # 96
            nn.BatchNorm1d(32 * 4), nn.SiLU()
        )
        # ANS (4 channels)
        self.ans_enc = nn.Sequential(
            nn.Conv1d(self.ans_channels, self.ans_channels * 4, kernel_size=5, stride=2, padding=2, groups=self.ans_channels),
            nn.BatchNorm1d(self.ans_channels * 4), nn.SiLU(),
            nn.Conv1d(self.ans_channels * 4, self.ans_channels * 4, kernel_size=3, stride=2, padding=1, groups=self.ans_channels),
            nn.BatchNorm1d(self.ans_channels * 4), nn.SiLU(),
            nn.Conv1d(self.ans_channels * 4, self.ans_channels * 4, kernel_size=3, stride=2, padding=1, groups=self.ans_channels),
            nn.BatchNorm1d(self.ans_channels * 4), nn.SiLU()
        )

        # 2. Spatial DAG (Micro-SCM for CNS)
        # Learn adjacency [32, 32] to mix spatial info
        initial_adj = torch.eye(32) * 1.0 + torch.randn(32, 32) * 0.1
        self.dag_adj = nn.Parameter(initial_adj)

        # 3. SNN Dynamics
        # Input dim = (32*4) + (4*4) = 128 + 16 = 144
        self.snn_input_proj = nn.Linear((self.eeg_channels+self.ans_channels)*4, 64)
        self.lif = VectorizedLIFLayer(num_neurons=64, in_dim=64)

        # 4. Causal Memory (Front-Door)
        self.memory = CausalMemoryBank(input_dim=64, num_prototypes=64)

        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 64 // 2),
            nn.SiLU(),
            nn.BatchNorm1d(64 // 2),
            nn.Dropout(p=0.5),
            nn.Linear(64 // 2, num_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(144, 64 // 2),
            nn.SiLU(),
            nn.BatchNorm1d(64 // 2),
            nn.Dropout(p=0.5),
            nn.Linear(64 // 2, num_classes),
        )

    def forward(self, eeg, ecg, gsr):
        # eeg: [B, 32, 768]
        # ans: [B, 4, 768]
        B = eeg.shape[0]
        ans = torch.cat([ecg, gsr], dim=1)

        # x_cns = self.cns_enc(eeg).mean(dim=1)
        # return self.mlp(x_cns), torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)

        # --- Step 1: Encoding & Temporal Abstraction ---
        # Output: [B, 128, 96] -> view -> [B, 32, 4, 96]
        x_cns = self.cns_enc(eeg).view(B, self.eeg_channels, 4, -1) # [B, 32, 4, 96]
        x_ans = self.ans_enc(ans).view(B, self.ans_channels, 4, -1)  # [B, 4, 4, 96]

        # --- Step 2: Spatial DAG (CNS Only) ---
        # DAG Constraint: H_new = A @ H_old
        # x_cns: [B, N, D, T] -> permute -> [B, T, D, N]
        adj = F.gumbel_softmax(self.dag_adj, tau=1, hard=True)
        adj = adj * (1 - torch.eye(32, device=adj.device))  # No self-loop

        x_cns_perm = x_cns.permute(0, 3, 2, 1)  # [B, 96, 4, 32]
        x_cns_dag = torch.matmul(x_cns_perm, adj.t())  # Spatial Mix

        # x = x_cns_dag.reshape(B, 96, -1)
        # x = x.mean(dim=1)
        # return self.mlp(x), torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)

        # --- Step 3: Fusion & SNN ---
        # Flatten channels and features: [B, 96, 32*4]
        flat_cns = x_cns_dag.reshape(B, 96, -1)
        flat_ans = x_ans.permute(0, 3, 1, 2).reshape(B, 96, -1)

        # Concat: [B, 96, 144]
        snn_in = torch.cat([flat_cns, flat_ans], dim=-1)

        # snn_in = snn_in.mean(dim=1)
        # return self.mlp(snn_in), torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)

        # LIF
        snn_drive = self.snn_input_proj(snn_in)  # [B, 96, 64]
        spikes, _, rates = self.lif(snn_drive)  # rates: [B, 64]

        # === 新增：计算发放率 Loss ===
        # 目标：让所有神经元的平均发放率接近 0.05 (5% 的时间在发放)
        # spikes shape: [Batch, Time, Neurons]
        # 我们希望在 Batch 和 Time 维度上取平均，得到每个神经元的平均活跃度
        mean_firing_rate = torch.mean(spikes, dim=(0, 1))  # [Neurons]

        target_rate = 0.05  # 目标发放率，可以设为 0.05 ~ 0.1
        # 使用 MSE 惩罚偏离目标的神经元
        fire_rate_loss = torch.mean((mean_firing_rate - target_rate) ** 2)
        # ==========================

        # --- Step 4: Front-Door Intervention ---
        # rates is our 'Z', use it to query Memory to get 'M'
        m_causal, attn, rec_loss, div_loss = self.memory(rates) # 捕获 attn 用于可视化
        # 这是之前71ACC版本，没有别的损失的
        # m_causal, attn = self.memory(rates)


        # --- Step 5: Prediction ---
        logits = self.classifier(m_causal)

        return logits, torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True), attn, rec_loss, div_loss, fire_rate_loss

        # 这是之前71ACC版本，没有别的损失的
        return logits, torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)

        return logits, adj, spikes

# 第一阶段独立接口；保留旧模型绑定，真实数据入口留待第二阶段对接。
from frontdoor_scm import (
    FrontDoorConfig, CNSMediator, ANSMediator, ReferenceEncoder, OutcomeHead,
    ReferenceBank, FrontDoorEstimator, frontdoor_loss,
)
