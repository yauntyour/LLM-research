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
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.ffn = SwiGLU(D)
        self.d = D**0.5

    def forward(self, x):
        x = self.norm1(x)
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)

        attn = F.softmax(Q @ K.transpose(-2, -1) / self.d, dim=-1) @ V

        x = x + self.ffn(attn)
        x = self.norm2(x)
        return self.dropout(x)


class MoE(nn.Module):
    def __init__(self, D: int, n_block: int, dropout: float = 0.1, top_k: int = 2):
        super().__init__()
        self.n_block = n_block
        self.top_k = top_k

        self.fx = SwiGLU(D, n_block)
        self.blocks = nn.ModuleList(
            [AttentionLayer(D, dropout) for _ in range(n_block)]
        )
        self.norm = nn.LayerNorm(D)
        self.dropout = nn.Dropout(dropout)
        self.ffn = SwiGLU(D, D)

    def forward(self, x):
        B, L, D = x.shape
        probs = F.softmax(self.fx(x), dim=-1)  # [B, L, n_block]
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        topk_weights = F.softmax(topk_probs, dim=-1)

        final_attn = torch.zeros_like(x)
        expert_counts = torch.zeros(self.n_block, device=x.device)

        for i in range(self.n_block):
            mask = topk_indices == i
            if not mask.any():
                continue
            w = (topk_weights * mask.float()).sum(dim=-1)

            active_mask = w > 0
            if not active_mask.any():
                continue

            expert_counts[i] += active_mask.sum()

            selected_x = x[active_mask]
            selected_w = w[active_mask].unsqueeze(-1)

            out = self.blocks[i](selected_x)
            final_attn[active_mask] += out * selected_w

        load_frac = expert_counts / (B * L * self.top_k)
        aux_loss = ((load_frac - 1.0 / self.n_block) ** 2).sum() * self.n_block

        out = x + self.ffn(final_attn)
        out = self.norm(out)

        return self.dropout(out), aux_loss


if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, 1024, 4096, device=device)
    model = MoE(4096, 4, top_k=4).to(device)

    start = time.time()
    out, aux_loss = model(x)
    print(f"Time: {(time.time() - start) * 1000:.4f}ms")
    print(out.shape, aux_loss)

    loss = aux_loss + F.mse_loss(out, x)
    loss.backward()
