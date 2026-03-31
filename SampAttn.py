from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.linear = nn.Linear(in_dim, 2 * out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        gate, value = x.chunk(2, dim=-1)
        return self.out_proj(value * F.silu(gate))


class RoPE(nn.Module):
    """
    旋转位置嵌入 (Rotary Position Embedding)
    对输入的 Q 和 K 张量（形状均为 [B, L, D]）应用 RoPE。
    要求 D 为偶数。
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        参数:
            dim: 特征维度 (必须是偶数)
            max_seq_len: 预计算的最大序列长度
            base: 频率计算的基数 (默认为10000)
        """
        super().__init__()
        assert dim % 2 == 0, "特征维度必须为偶数"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 预计算频率 theta
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # [dim/2]

        # 生成位置索引 [0, 1, ..., max_seq_len-1]
        position = torch.arange(max_seq_len).float().unsqueeze(1)  # [max_seq_len, 1]

        # 计算角度: [max_seq_len, dim/2]
        angles = position * theta.unsqueeze(0)

        # 计算 cos 和 sin，并扩展为 [max_seq_len, dim] (每个角度重复两次)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # [max_seq_len, dim]
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)  # [max_seq_len, dim]

        # 注册为缓冲区 (不参与梯度计算)
        self.register_buffer("cos_cached", cos.unsqueeze(0))  # [1, max_seq_len, dim]
        self.register_buffer("sin_cached", sin.unsqueeze(0))  # [1, max_seq_len, dim]

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        参数:
            q: Query 张量, 形状 [B, L, D]
            k: Key   张量, 形状 [B, L, D]
        返回:
            q_embed, k_embed: 旋转后的张量, 形状与输入相同
        """
        B, L, D = q.shape
        assert D == self.dim, f"输入维度 {D} 与初始化维度 {self.dim} 不符"
        assert (
            L <= self.max_seq_len
        ), f"序列长度 {L} 超过最大缓存长度 {self.max_seq_len}"

        # 获取当前长度对应的 cos 和 sin，并确保设备与输入一致
        cos = self.cos_cached[:, :L, :].to(q.device)  # [1, L, D]
        sin = self.sin_cached[:, :L, :].to(q.device)  # [1, L, D]

        # 应用旋转嵌入
        q_embed = self._apply_rope(q, cos, sin)
        k_embed = self._apply_rope(k, cos, sin)

        return q_embed, k_embed

    def _apply_rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        对单个张量应用 RoPE (使用 rotate_half 技巧)
        x: [B, L, D]
        cos, sin: [1, L, D]
        """
        # rotate_half: 将最后维度分成两半并交换，同时改变符号
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)

        # 旋转公式: x * cos + rotated * sin
        return x * cos + rotated * sin


class BlockAttention(nn.Module):
    def __init__(
        self,
        D: int,
        L: int,
        block_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert L % block_length == 0
        self.L = L
        self.D = D
        self.block_length = block_length
        self.K = L // block_length

        self.Query = nn.Linear(D, D, bias=False)
        self.Key = nn.Linear(D, D, bias=False)
        self.Value = nn.Linear(D, D, bias=False)

        self.transform = SwiGLU(D * block_length, D)
        self.ffn = SwiGLU(D)
        self.norm = nn.RMSNorm(D)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(self.K, self.K))
        self.causal_mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, 0.0
        )

    def forward(self, x, rope: RoPE):
        mask = self.causal_mask.to(x.device, x.dtype)

        Q = self.Query(x)
        K = self.Key(x)
        V = self.Value(x)
        K, Q = rope(K, Q)

        Q = Q.reshape(-1, self.K, self.block_length * self.D)
        K = K.reshape(-1, self.K, self.block_length * self.D)
        V = V.reshape(-1, self.K, self.block_length * self.D)

        Q = self.transform(Q)
        K = self.transform(K)
        V = self.transform(V)

        attn = F.softmax(Q @ K.transpose(-1, -2) / (self.D**0.5) + mask, dim=-1) @ V

        attn = self.norm(attn)
        attn = self.ffn(attn)
        return self.dropout(attn)


class KVQueryLayer(nn.Module):
    def __init__(self, D: int, L: int, block_length: int):
        super().__init__()
        assert L % block_length == 0
        self.Key = nn.Linear(D, L // block_length, bias=False)
        self.Value = nn.Linear(D, D, bias=False)

    def forward(self, x):
        K = self.Key(x)  # [B, L, K]
        V = self.Value(x)
        return K.transpose(-1, -2) @ V, K


class HyperConnection(nn.Module):
    """
    论文中的 Hyper-Connections (HC) 结构，包含三个可学习映射。
    此处为简化版，聚焦于 mHC 的核心约束逻辑。
    """

    def __init__(self, dim, n=4, sinkhorn_iters=20):
        """
        Args:
            dim: 模型维度 C (即单个流的维度)
            n: 残差流扩展倍数
            sinkhorn_iters: Sinkhorn-Knopp 迭代次数
        """
        super().__init__()
        self.n = n
        self.dim = dim
        self.sinkhorn_iters = sinkhorn_iters

        # 定义生成动态映射的线性投影 (对应论文公式7)
        # 注意：输入是 flattened 后的 n*dim 维向量
        self.proj_pre = nn.Linear(n * dim, n, bias=False)
        self.proj_post = nn.Linear(n * dim, n, bias=False)
        # 输出维度是 n*n，用于重塑为 H_res 矩阵
        self.proj_res = nn.Linear(n * dim, n * n, bias=False)

        # 可学习的门控因子和偏置 (对应论文公式7中的 alpha 和 b)
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))

        self.bias_pre = nn.Parameter(torch.zeros(1, n))
        self.bias_post = nn.Parameter(torch.zeros(1, n))
        self.bias_res = nn.Parameter(torch.zeros(1, n, n))  # 重塑为 n*n

        # 用于输入 x 的 RMSNorm，在 flattened 向量上操作
        self.norm = nn.RMSNorm(n * dim)

    def _sinkhorn_knopp(self, mat):
        """Sinkhorn-Knopp 算法将矩阵投影到双随机矩阵流形"""
        # 确保输入为正 (exp操作)
        mat = torch.exp(mat)
        for _ in range(self.sinkhorn_iters):
            # 行归一化
            mat = mat / mat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            # 列归一化
            mat = mat / mat.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        return mat

    def forward(self, x):
        # x 是残差流: [batch, seq_len, n, dim] 或类似形状
        # 为了简化，假设输入 x 的形状为 [..., n, dim]
        original_shape = x.shape
        # 合并 batch 和 seq_len 维度，并 flatten 最后一个维度: [..., n*dim]
        x_flat = x.reshape(-1, self.n * self.dim)

        # 应用 RMSNorm (对应论文公式7)
        x_norm = self.norm(x_flat)

        # 计算动态部分 (对应论文公式7)
        dyn_pre = self.proj_pre(x_norm)  # [..., n]
        dyn_post = self.proj_post(x_norm)  # [..., n]
        dyn_res = self.proj_res(x_norm)  # [..., n*n]

        # 加入门控因子和偏置 (对应论文公式7和16)
        # 注意：这里简化了除以norm的操作，论文中在kernel fusion里处理
        h_pre_tilde = self.alpha_pre * dyn_pre + self.bias_pre
        h_post_tilde = self.alpha_post * dyn_post + self.bias_post
        h_res_tilde = self.alpha_res * dyn_res + self.bias_res.reshape(1, -1)

        # 重塑 H_res 为矩阵形式: [..., n, n]
        h_res_tilde = h_res_tilde.reshape(-1, self.n, self.n)

        # ----- 核心 mHC 约束操作 (对应论文公式8) -----
        # 1. 对 H_pre 和 H_post 应用 Sigmoid (非负约束)
        H_pre = torch.sigmoid(h_pre_tilde)  # [..., n]
        H_post = 2 * torch.sigmoid(h_post_tilde)  # [..., n]

        # 2. 对 H_res 应用 Sinkhorn-Knopp (双随机约束)
        # 处理 H_res 时，对每个位置的矩阵独立进行迭代
        H_res = self._sinkhorn_knopp(h_res_tilde)  # [..., n, n]

        # 将结果恢复为 (batch, seq_len) 的原始形状
        batch_seq_len = original_shape[:-2]
        H_pre = H_pre.reshape(*batch_seq_len, self.n)  # [..., n]
        H_post = H_post.reshape(*batch_seq_len, self.n)  # [..., n]
        H_res = H_res.reshape(*batch_seq_len, self.n, self.n)  # [..., n, n]

        return H_pre, H_post, H_res


class Chainward(nn.Module):
    def __init__(self, D: int, n_HC: int = 4, exp: float = 2):
        super().__init__()
        self.d = D**0.5
        self.Q = nn.Linear(D, D, bias=False)
        self.K = nn.Linear(D, D, bias=False)
        self.V = nn.Linear(D, D, bias=False)

        # self.n_HC = n_HC
        # self.hc = HyperConnection(n_HC, exp)
        self.hcx = HyperConnection(D, n_HC)
        self.hcy = HyperConnection(D, n_HC)
        self.ffn = SwiGLU(D)
        self.norm = nn.RMSNorm(D)

    def forward(self, x, y):
        Hx_pre, Hx_post, Hx_res = self.hcx(x)
        Hy_pre, Hy_post, Hy_res = self.hcy(y)

        x_pre = torch.einsum("...nd,...n->...d", x, Hx_pre)
        y_per = torch.einsum("...nd,...n->...d", y, Hy_pre)

        Q = self.Q(x_pre)
        K = self.K(y_per)
        V = self.V(y_per)

        out = F.softmax(Q @ K.transpose(-1, -2) / self.d, dim=-1) @ V
        out = x_pre + out + y_per
        out = self.ffn(self.norm(out)) + out

        x_post_contrib = Hx_post.unsqueeze(-1) * out.unsqueeze(-2)
        y_post_contrib = Hy_post.unsqueeze(-1) * out.unsqueeze(-2)

        x_res_contrib = torch.einsum("...ij,...jd->...id", Hx_res, x)
        y_res_contrib = torch.einsum("...ij,...jd->...id", Hy_res, y)

        new_out = x_post_contrib + y_post_contrib + x_res_contrib + y_res_contrib

        return new_out


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        D: int,
        L: int,
        block_length: int,
        n_layer: int,
        dropout: float = 0.1,
        n_HC: int = 4,
        exp: float = 2.0,
    ):
        super().__init__()
        self.n_HC = n_HC
        self.n_layer = n_layer
        self.embedding = nn.Embedding(vocab_size, D)
        self.rope = RoPE(D, L)

        self.decoders = nn.ModuleList(
            [BlockAttention(D, L, block_length, dropout) for _ in range(n_layer)]
        )

        self.kvq = KVQueryLayer(D, L, block_length)
        self.chainward = Chainward(D, n_HC, exp)
        self.norm = nn.RMSNorm(D)

    def forward(self, x):
        x = self.embedding(x)
        query, key = self.kvq(x)
        query = query.unsqueeze(2).repeat(1, 1, self.n_HC, 1)
        for decoder in self.decoders:
            att = decoder(x, self.rope)
            att = att.unsqueeze(2).repeat(1, 1, self.n_HC, 1)
            query = query + self.chainward(query, att)
        query = query.sum(dim=2)
        query = self.norm(key @ query)

        query = query @ self.embedding.weight.T
        return query


class MuSGD(Optimizer):
    """
    MuSGD (Momentum SGD) Optimizer

    参数:
        params (iterable): 待优化的参数
        lr (float): 学习率 (默认: 0.01)
        momentum (float): 动量因子 (默认: 0.9)
        weight_decay (float): L2 正则化系数 (默认: 0)
        dampening (float): 动量阻尼系数 (默认: 0)
        nesterov (bool): 是否使用 Nesterov 动量 (默认: False)
    """

    def __init__(
        self, params, lr=0.01, momentum=0.9, weight_decay=0, dampening=0, nesterov=False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
        super(MuSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MuSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行单步优化

        参数:
            closure (callable, optional): 重新评估模型并返回损失的闭包
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                # L2 正则化 (weight decay)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # 动量更新
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.zeros_like(p)
                    buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                # Nesterov 动量
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

                # 参数更新
                p.add_(d_p, alpha=-lr)

        return loss

    def get_lr(self):
        """获取当前学习率"""
        return [group["lr"] for group in self.param_groups]


if __name__ == "__main__":
    import time

    B = 1
    D, L = 128, 300
    block_len = 10
    K = L // block_len
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Transformer(6400, D, L, block_len, 3, n_HC=4, exp=2).to(device)

    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_size}")

    data = torch.randint(0, 6400, (B, L + 1), device=device)
    x = data[:, :-1]
    y = data[:, 1:]

    optimizer = MuSGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-8
    )
    out_flatten = nn.Flatten(0, 1)

    epochs = 1000
    model.train()
    losses = []
    try:
        for i in range(epochs):
            start = time.time()
            out = model(x)
            loss = F.cross_entropy(out_flatten(out), out_flatten(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"Epoch: {i}, Loss: {loss.item()}, lr: {optimizer.get_lr()[0]}")
            losses.append(loss.item())
            if i % 100 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_cpu = param.grad.detach().cpu()
                        print(
                            f"Layer: {name:20} | Grad Norm: {grad_cpu.norm(2).item():.6e}"
                        )
                    else:
                        print(
                            f"Layer: {name:20} | Grad is NONE!"
                        )  # 检查是否有层没参与计算
                input("Press Enter to continue...")
    except KeyboardInterrupt:
        pass

    plt.plot(losses)
    plt.show()
