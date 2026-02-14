"""朴素 attention 和 PyTorch SDPA 封装"""

import torch
import torch.nn.functional as F
import math
from typing import Optional


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    mask: Optional[torch.Tensor] = None,
                    scale: Optional[float] = None) -> torch.Tensor:
    """
    朴素 attention 实现（会生成 N*N 中间矩阵）。
    Q, K, V: [B, H, N, d]
    mask: 可广播到 [B, H, N, N] 的 mask，有效位 0，无效位 -inf
    返回: [B, H, N, d]
    """
    d = Q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # S: [B, H, N, N]
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if mask is not None:
        S = S + mask.to(S.dtype)

    # 数值稳定：全 -inf 行 softmax 输出为 0
    P = torch.softmax(S, dim=-1)
    P = torch.nan_to_num(P, nan=0.0)

    O = torch.matmul(P, V)
    return O


def sdpa_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                   is_causal: bool = False,
                   attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    PyTorch SDPA 封装。
    Q, K, V: [B, H, N, d]
    """
    return F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=attn_mask,
        is_causal=is_causal,
    )
