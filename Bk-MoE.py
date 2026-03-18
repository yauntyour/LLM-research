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

    def forward(self, x):
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

        attn = F.softmax(Q @ K.transpose(-2, -1) / self.d, dim=-1) @ V

        x = x + self.ffn(attn)
        x = self.norm2(x)
        return self.dropout(x)


class BlockRouter(nn.Module):
    def __init__(self, D: int, n_block: int, dropout: float = 0.1, topk: int = 2):
        super().__init__()
        self.topk = topk
        self.n_block = n_block
        self.fx = SwiGLU(D, n_block)
        self.blocks = nn.ModuleList(
            [AttentionLayer(D, dropout) for _ in range(n_block)]
        )
        self.norm = nn.RMSNorm(D)
        self.dropout = nn.Dropout(dropout)
        self.ffn = SwiGLU(D)

    def forward(self, x):
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
            sub_attn = self.blocks[j](sub_x)  # [num_j, L, D]
            attn[sample_indices] += sub_attn

        x = x + self.ffn(attn)
        x = self.norm(x)
        x = self.dropout(x)
        return x, aux_loss


if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, 1024, 4096, device=device)
    model = BlockRouter(4096, 4, topk=4).to(device)

    start = time.time()
    out, aux_loss = model(x)
    print(f"Time: {(time.time() - start)*1000:.4f}ms")
    print(out.shape)

    loss = aux_loss + F.mse_loss(out, x)
    loss.backward()
