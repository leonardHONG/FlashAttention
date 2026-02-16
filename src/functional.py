"""上层调用接口: 封装 Triton kernel 为易用 API, 支持训练级梯度传播"""

import torch
from typing import Optional
from src.kernel.flash_attn_triton import flash_attn_forward, flash_attn_forward_debug
from src.kernel.flash_attn_bwd_triton import flash_attn_backward


# ============================================================
# Autograd Function
# ============================================================
class FlashAttentionFunc(torch.autograd.Function):
    """支持完整梯度传播的 FlashAttention autograd 封装"""

    @staticmethod
    def forward(ctx, Q, K, V, causal, seqlens_k):
        Out, L = flash_attn_forward(Q, K, V, causal=causal, seqlens_k=seqlens_k)
        ctx.save_for_backward(Q, K, V, Out, L)
        ctx.causal = causal
        ctx.seqlens_k = seqlens_k
        return Out

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, Out, L = ctx.saved_tensors
        dO = dO.contiguous()
        dQ, dK, dV = flash_attn_backward(
            dO, Q, K, V, Out, L,
            causal=ctx.causal,
            seqlens_k=ctx.seqlens_k,
        )
        return dQ, dK, dV, None, None


# ============================================================
# 公开接口
# ============================================================
def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    seqlens_k: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FlashAttention 统一接口, 支持前向和反向传播

    参数:
        Q, K, V: [B, H, N, d] float16/bf16
        causal: 是否启用 causal mask
        seqlens_k: [B] int32 每个 batch 的有效 K 长度

    返回:
        O: [B, H, N, d]
    """
    return FlashAttentionFunc.apply(Q, K, V, causal, seqlens_k)


def flash_attention_debug(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    seqlens_k: Optional[torch.Tensor] = None,
):
    """
    带 early-exit 统计的 debug 接口, 不支持梯度

    返回:
        O: [B, H, N, d]
        skip_counts: [B*H, num_q_blocks] 每个 Q block 跳过的 K block 数
    """
    Out, _L, skip_counts = flash_attn_forward_debug(
        Q, K, V, causal=causal, seqlens_k=seqlens_k
    )
    return Out, skip_counts
