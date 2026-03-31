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
        self, D: int, L: int, block_length: int, hidden_dim: int, dropout: float = 0.1
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

        self.norm1 = nn.RMSNorm(D)
        self.norm2 = nn.RMSNorm(D)
        self.ffn = SwiGLU(D)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(self.block_length, self.block_length))
        self.causal_mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, 0.0
        )

    def forward(self, x, rope: RoPE):
        mask = self.causal_mask.to(x.device, x.dtype)
        x = self.norm1(x)
        x = x.reshape(-1, self.K, self.block_length, self.D)

        Q = self.Query(x)
        K = self.Key(x)
        V = self.Value(x)
        K, Q = rope(K, Q)

        Q = Q.reshape(-1, self.K, self.block_length, self.D)
        K = K.reshape(-1, self.K, self.block_length, self.D)

        att = F.softmax(Q @ K.transpose(-1, -2) / (self.D**0.5) + mask, dim=-1)
        
        att = 

        return self.dropout(x)


class BlockMoE(nn.Module):
    def __init__(
        self,
        D: int,
        L: int,
        block_length: int,
        hidden_dim: int,
        n_block: int,
        dropout: float = 0.1,
        topk: int = 1,
    ):
        super().__init__()
        self.topk = topk
        self.n_block = n_block
        self.fx = SwiGLU(D, n_block)
        self.blocks = nn.ModuleList(
            [
                BlockAttention(D, L, block_length, hidden_dim, dropout)
                for _ in range(n_block)
            ]
        )
        self.norm = nn.RMSNorm(D)
        self.dropout = nn.Dropout(dropout)
        self.ffn = SwiGLU(D)

    def forward(self, x, rope: RoPE):
        B, L, D = x.shape

        active = torch.sum(self.fx(x), dim=-2)  # [B, n_block]
        probs = F.softmax(active, dim=-1)  # [B, n_block]
        _, idxs = torch.topk(probs, self.topk, dim=-1)  # [B, topk]

        mask = torch.zeros_like(probs).scatter_(1, idxs, 1)  # [B, n_block]
        freq = mask.mean(dim=0)  # [n_block]
        target = torch.full_like(freq, self.topk / self.n_block)
        aux_loss = F.mse_loss(freq, target)

        attn = torch.zeros_like(x)  # [B, L, D]
        unique_blocks = torch.unique(idxs)
        for j in unique_blocks:
            mask_j = (idxs == j).any(dim=1)  # [B] boolean
            if not mask_j.any():
                continue
            sample_indices = torch.where(mask_j)[0]  # [num_j]
            sub_x = x[sample_indices]  # [num_j, L, D]
            sub_attn = self.blocks[j](sub_x, rope)  # [num_j, L, D]
            attn[sample_indices] += sub_attn

        x = x + self.ffn(attn)
        x = self.norm(x)
        x = self.dropout(x)
        return x, aux_loss


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        D: int,
        L: int,
        block_length: int,
        hidden_dim: int,
        n_layer: int,
        n_block: int,
        topk_block: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.embedding = nn.Embedding(vocab_size, D)
        self.decoders = nn.ModuleList(
            [
                BlockMoE(D, L, block_length, hidden_dim, n_block, dropout, topk_block)
                for _ in range(n_layer)
            ]
        )
        self.rope = RoPE(D, L // block_length)
        self.ffn = SwiGLU(D)
        self.norm = nn.RMSNorm(D)

    def forward(self, x):
        x = self.embedding(x)
        aux_loss = 0
        for decoder in self.decoders:
            x, ros_loss = decoder(x, self.rope)
            aux_loss = aux_loss + ros_loss
        aux_loss = aux_loss / self.n_layer
        x = self.ffn(x)
        x = self.norm(x)
        out = x @ self.embedding.weight.t()
        return out, aux_loss


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

    B = 4
    D, L = 128, 1000
    block_len = 10
    hidden_dim = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Transformer(6400, D, L, block_len, hidden_dim, 1, 4, topk_block=1).to(
        device
    )

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
            out, aux_loss = model(x)
            loss = F.cross_entropy(out_flatten(out), out_flatten(y)) + aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(
                f"Epoch: {i}, Loss: {loss.item() - aux_loss.item()}, cost time: {time.time() - start}s"
            )
            losses.append(loss.item())
    except KeyboardInterrupt:
        pass

    plt.title("Block-att Loss")
    plt.plot(losses)
    plt.show()
