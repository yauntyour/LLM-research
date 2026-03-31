from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fc = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x):
        x = self.fc(x)
        gate, value = x.chunk(2, dim=-1)
        return F.silu(gate) * value


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
        x_flat = x.view(-1, self.n * self.dim)

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
        h_res_tilde = self.alpha_res * dyn_res + self.bias_res.view(1, -1)

        # 重塑 H_res 为矩阵形式: [..., n, n]
        h_res_tilde = h_res_tilde.view(-1, self.n, self.n)

        # ----- 核心 mHC 约束操作 (对应论文公式8) -----
        # 1. 对 H_pre 和 H_post 应用 Sigmoid (非负约束)
        H_pre = torch.sigmoid(h_pre_tilde)  # [..., n]
        H_post = 2 * torch.sigmoid(h_post_tilde)  # [..., n]

        # 2. 对 H_res 应用 Sinkhorn-Knopp (双随机约束)
        # 处理 H_res 时，对每个位置的矩阵独立进行迭代
        H_res = self._sinkhorn_knopp(h_res_tilde)  # [..., n, n]

        # 将结果恢复为 (batch, seq_len) 的原始形状
        batch_seq_len = original_shape[:-2]
        H_pre = H_pre.view(*batch_seq_len, self.n)  # [..., n]
        H_post = H_post.view(*batch_seq_len, self.n)  # [..., n]
        H_res = H_res.view(*batch_seq_len, self.n, self.n)  # [..., n, n]

        return H_pre, H_post, H_res


class LinearAttention(nn.Module):
    def __init__(
        self,
        D: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = D**0.5

        self.Q = nn.Linear(D, D, bias=False)
        self.K = nn.Linear(D, D, bias=False)
        self.V = nn.Linear(D, D, bias=False)

        self.ffn = SwiGLU(D)
        self.norm1 = nn.RMSNorm(D)
        self.norm2 = nn.RMSNorm(D)
        # self.hc = HyperConnection(D, n_HC, sinkhorn_iters)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, ones_norm, mask_norm):
        """H_pre, H_post, H_res = self.hc(x)

        x_pre = torch.einsum("...nd,...n->...d", x, H_pre)
        x_norm = self.norm1(x_pre)

        Q = self.Q(x_norm)
        K = self.K(x_norm)
        V = self.V(x_norm)

        alpha = K.transpose(-1, -2) @ V
        beta = K.transpose(-1, -2) @ ones_norm
        attn = (Q @ alpha + causal_mask @ V) / (Q @ beta + mask_norm)

        attn = x_pre + attn
        out = self.ffn(self.norm(attn)) + attn
        post_contrib = H_post.unsqueeze(-1) * out.unsqueeze(-2)
        res_contrib = torch.einsum("...ij,...jd->...id", H_res, x)
        new_x = res_contrib + post_contrib

        return self.dropout(new_x)"""

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        alpha = K.transpose(-1, -2) @ V
        beta = K.transpose(-1, -2) @ ones_norm
        attn = (Q @ alpha) / (Q @ beta)

        x = self.ffn(self.norm2(attn)) + x
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        D: int,
        L: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D)
        self.rope
        self.blocks = nn.ModuleList(
            [LinearAttention(D, dropout) for _ in range(n_layers)]
        )
        self.ffn = SwiGLU(D)
        self.norm = nn.RMSNorm(D)
        self.register_buffer("ones_norm", torch.ones([L, 1]))

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, self.ones_norm)
        x = self.ffn(self.norm(x)) + x
        out = x @ self.embedding.weight.T
        return out


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
    vocab_size = 6400
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Decoder(vocab_size, D, L, 1).to(device)

    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_size}")

    data = torch.randint(0, vocab_size, (B, L + 1), device=device)
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
            print(f"Epoch: {i}, Loss: {loss.item()}, cost time: {time.time() - start}s")
            losses.append(loss.item())
    except KeyboardInterrupt:
        pass

    plt.plot(losses)
    plt.show()
