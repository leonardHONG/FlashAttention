"""上层调用接口：封装 Triton kernel 为易用 API"""

import torch
from typing import Optional
from src.kernel.flash_attn_triton import flash_attn_forward, flash_attn_forward_debug


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    seqlens_k: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FlashAttention 前向传播统一接口。

    参数:
        Q, K, V: [B, H, N, d]，float16
        causal: 是否启用 causal mask
        seqlens_k: [B] int32，每个 batch 的有效 K 长度

    返回:
        O: [B, H, N, d]
    """
    return flash_attn_forward(Q, K, V, causal=causal, seqlens_k=seqlens_k)


def flash_attention_debug(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    seqlens_k: Optional[torch.Tensor] = None,
):
    """
    带 early-exit 统计的 debug 接口

    返回:
        O: [B, H, N, d]
        skip_counts: [B*H, num_q_blocks]，每个 Q block 跳过的 K block 数
    """
    return flash_attn_forward_debug(Q, K, V, causal=causal, seqlens_k=seqlens_k)
