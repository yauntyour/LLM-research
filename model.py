import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LightningEncoder(nn.Module):
    def __init__(self, D: int, topk: int, dropout: float = 0.1):
        super().__init__()
        assert topk > 0
        self.topk = topk
        self.d = D**0.5
        self.Q = nn.Linear(D, D, bias=False)
        self.K = nn.Linear(D, D, bias=False)
        self.V = nn.Linear(D, D, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.RMSNorm(D)
        self.norm2 = nn.RMSNorm(D)
        self.ffn = SwiGLU(D)
        self.fc = SwiGLU(D, 1)

    def forward(self, x, rope: RoPE):
        B, L, D = x.shape

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        Q, K = rope(Q, K)

        x = self.norm1(x)
        attn = F.softmax(Q @ K.transpose(-2, -1) / self.d, dim=-1) @ V
        x = self.dropout(self.norm2(x + self.ffn(attn)))

        if L > self.topk:
            score = torch.sum(self.fc(x), dim=-1)
            _, topk_idx = torch.topk(score, self.topk, dim=-1)
            return x[torch.arange(B, device=x.device).unsqueeze(1), topk_idx]
        return x


class AttentionLayer(nn.Module):
    def __init__(self, D: int, dropout: float = 0.1):
        super().__init__()
        self.K = nn.Linear(D, D, bias=False)
        self.Q = nn.Linear(D, D, bias=False)
        self.V = nn.Linear(D, D, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.RMSNorm(D)
        self.norm2 = nn.RMSNorm(D)
        self.ffn = SwiGLU(D)
        self.d = D**0.5

        self.L = None
        self.causal_mask = None

    def forward(self, x, rope: RoPE):
        assert len(x.shape) >= 2
        if self.L is None or self.L != x.shape[-2]:
            self.L = x.shape[-2]
            mask = torch.tril(torch.ones(self.L, self.L)).to(x.device)
            self.causal_mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(
                mask == 1, 0.0
            )
        x = self.norm1(x)
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)

        Q, K = rope(Q, K)

        attn = F.softmax(Q @ K.transpose(-2, -1) / self.d, dim=-1) @ V

        x = x + self.ffn(attn)
        x = self.norm2(x)
        return self.dropout(x)


class BlockMoE(nn.Module):
    def __init__(self, D: int, n_block: int, dropout: float = 0.1, topk: int = 2):
        super().__init__()
        assert topk > 0 and topk <= n_block
        self.topk = topk
        self.n_block = n_block
        self.fx = SwiGLU(D, n_block)
        self.blocks = nn.ModuleList(
            [AttentionLayer(D, dropout) for _ in range(n_block)]
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
        topk_encoder: int,
        n_layer: int,
        n_block: int,
        topk_block: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.embedding = nn.Embedding(vocab_size, D)
        self.rope = RoPE(D)
        self.encoder = LightningEncoder(D, topk_encoder, dropout)
        self.decoders = nn.ModuleList(
            [BlockMoE(D, n_block, dropout, topk_block) for _ in range(n_layer)]
        )
        self.ffn = SwiGLU(D)
        self.norm = nn.RMSNorm(D)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x, self.rope)
        aux_loss = 0
        for decoder in self.decoders:
            x, ros_loss = decoder(x, self.rope)
            aux_loss += ros_loss
        aux_loss /= self.n_layer
        x = self.ffn(x)
        x = self.norm(x)

        out = x @ self.embedding.weight.t()
        return out, aux_loss


if __name__ == "__main__":
    import time

    B, V, D, L = 4, 6400, 128, 32
    K = L // 4
    out_flatten = nn.Flatten(0, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randint(0, V, (4, L), device=device)
    y = torch.randint(0, V, (4, K), device=device)
    print(f"In: {x.shape}")
    model = Transformer(V, D, K, 4, 8).to(device)
    start = time.time()
    out, aux_loss = model(x)
    print(f"Time: {(time.time() - start) * 1000:.4f}ms")
    print(f"Out: {out.shape}")
    loss = aux_loss + F.cross_entropy(out_flatten(out), y.flatten())
    print(f"Loss: {loss.item()}")
    loss.backward()

    num_params = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in model.buffers()
    )
    print(f"Model size: {num_params}")
    print(f"Model memory size: {num_params * 4 / 1024 / 1024}MB")
    input()
