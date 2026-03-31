import torch
import torch.nn as nn
import torch.nn.functional as F

L = 4
D = 8

mask = torch.tril(torch.ones(L, L))
causal = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
ones_norm = torch.ones([L, 1])


def softmax(Q, K, V):
    out = F.softmax(Q @ K.transpose(-1, -2) + causal, dim=-1) @ V
    return out


def linear(Q, K, V):
    alpha = K.transpose(-1, -2) @ V
    beta = K.transpose(-1, -2).sum(dim=-1, keepdim=True)
    attn = (Q @ alpha) / (Q @ beta)
    return attn


class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fc = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x):
        x = self.fc(x)
        gate, value = x.chunk(2, dim=-1)
        return F.silu(gate) * value


class net(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            SwiGLU(D),
            nn.ReLU(),
            SwiGLU(D),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


model = net(D)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.95))

epochs = 1000
for epoch in range(epochs):
    Q = torch.randn([1, L, D])
    K = torch.randn([1, L, D])
    V = torch.randn([1, L, D])
    target = softmax(Q, K, V)
    x = linear(Q, K, V)
    out = model(x)
    loss = F.mse_loss(out, target)
    loss.backward()
    optimizer.step()
    print(f"Loss:{loss.item()}")
