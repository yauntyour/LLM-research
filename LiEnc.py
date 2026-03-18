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

    def forward(self, x):
        B, L, D = x.shape
        assert L >= self.topk
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        x = self.norm1(x)
        attn = F.softmax(Q @ K.transpose(-2, -1) / self.d, dim=-1) @ V
        x = self.dropout(self.norm2(x + self.ffn(attn)))

        score = torch.sum(self.fc(x), dim=-1)
        _, topk_idx = torch.topk(score, self.topk, dim=-1)
        return x[torch.arange(x.shape[0]), topk_idx]


if __name__ == "__main__":
    model = LightningEncoder(64, 4)
    x = torch.randn(4, 16, 64)
    print(model(x).shape)
