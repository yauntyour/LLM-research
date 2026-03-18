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


class AttentionLayer(nn.Module):
    def __init__(self, D: int, dropout: float = 0.1, n: int = 4, use_mhc: bool = True):
        """
        Args:
            D: 模型维度
            n: 残差流扩展倍数
            use_mhc: 是否使用 mHC (True 则对 H_res 施加双随机约束，否则为原始 HC)
        """
        super().__init__()
        self.n = n
        self.use_mhc = use_mhc

        # 核心的 QKV 投影，输入是经过 H_pre 聚合后的单流 (维度 D)
        self.K = nn.Linear(D, D, bias=False)
        self.Q = nn.Linear(D, D, bias=False)
        self.V = nn.Linear(D, D, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(D)  # Attention 后的 norm
        self.norm2 = nn.LayerNorm(D)  # FFN 后的 norm
        self.ffn = SwiGLU(D)
        self.d = D**0.5

        # mHC/HC 连接模块
        # 注意：这里假设残差流已经存在，即输入 x 的形状是 [..., n, D]
        # HyperConnection 模块负责计算 H_pre, H_post, H_res
        self.hc = HyperConnection(D, n=n)

    def forward(self, x):
        # 假设输入 x 已经是 n 流残差形式: [batch, seq_len, n, D]
        # 1. 应用 H_pre 聚合 n 个流到单流输入给 Attention
        H_pre, H_post, H_res = self.hc(x)

        # H_pre 形状: [batch, seq_len, 1, n] 或 [batch, seq_len, n]
        # 计算聚合输入: 对 n 维度加权求和
        # x 形状: [batch, seq_len, n, D]
        # 通过 H_pre 加权聚合得到 layer_input: [batch, seq_len, D]
        # 简化：使用 einsum 进行批量加权求和
        layer_input = torch.einsum("...nd,...n->...d", x, H_pre)

        # 2. 标准的 Attention 和 FFN 处理 (在聚合后的单流上进行)
        normed_input = self.norm1(layer_input)
        K = self.K(normed_input)
        Q = self.Q(normed_input)
        V = self.V(normed_input)
        attn_output = F.softmax(Q @ K.transpose(-2, -1) / self.d, dim=-1) @ V
        # 残差连接 (在单流上)
        attn_output = layer_input + attn_output

        ffn_output = self.ffn(self.norm2(attn_output))
        # 最终的层输出 (单流)
        layer_output_single = attn_output + ffn_output

        # 3. 应用 H_post 和 H_res 将单流输出映射回 n 流残差
        # H_post: [batch, seq_len, n], 用于将单流输出 broadcast 并加权到每个流
        # H_res: [batch, seq_len, n, n], 用于混合旧的 n 流状态 x

        # 计算 H_post 的贡献: 将 layer_output_single 通过 H_post 广播到 n 个流
        post_contrib = H_post.unsqueeze(-1) * layer_output_single.unsqueeze(
            -2
        )  # [..., n, D]

        # 计算 H_res 的贡献: 用 H_res 矩阵混合旧的流 x
        # x: [..., n, D], H_res: [..., n, n]
        res_contrib = torch.einsum("...ij,...jd->...id", H_res, x)  # [..., n, D]

        # 新的残差流 = H_res @ 旧流 + H_post 广播后的层输出
        new_x = res_contrib + post_contrib

        return self.dropout(new_x)


if __name__ == "__main__":
    # Quick test
    B, T, D = 2, 10, 64
    n = 4
    layer1 = AttentionLayer(D, n=n, use_mhc=True)
    layer2 = AttentionLayer(D, n=n, use_mhc=False)
    x = torch.randn(B, T, n, D)
    out1 = layer1(x)
    print("Output:", out1.sum())
    out2 = layer2(x)
    print("Output:", out2.sum())
