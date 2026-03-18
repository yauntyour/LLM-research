import torch
import torch.nn as nn
import math


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


if __name__ == "__main__":
    batch_size = 2
    seq_len = 1
    dim = 64

    rope = RoPE(dim=dim, max_seq_len=1)

    # 2*D*L = buffer_size
    print(f"Model size: {sum(p.numel() for p in rope.buffers())}")

    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)

    q_rot, k_rot = rope(q, k)

    print(f"输入 Q 形状: {q.shape}")
    print(f"输出 Q_rot 形状: {q_rot.shape}")
    print(f"输入 K 形状: {k.shape}")
    print(f"输出 K_rot 形状: {k_rot.shape}")
