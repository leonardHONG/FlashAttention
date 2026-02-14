"""工具函数：生成数据、构造 mask、带宽计算"""

import torch
from typing import Optional, List


def gen_qkv(B: int, H: int, N: int, d: int,
            dtype: torch.dtype = torch.float16,
            device: str = "cuda"):
    """生成随机 Q, K, V 张量，形状 [B, H, N, d]。"""
    shape = (B, H, N, d)
    Q = torch.randn(shape, dtype=dtype, device=device)
    K = torch.randn(shape, dtype=dtype, device=device)
    V = torch.randn(shape, dtype=dtype, device=device)
    return Q, K, V


def make_causal_mask(N: int, device: str = "cuda"):
    """生成 causal mask: 上三角为 -inf，其余为 0。形状 [N, N]"""
    mask = torch.zeros(N, N, device=device, dtype=torch.float32)
    mask.masked_fill_(torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1), float("-inf"))
    return mask


def make_padding_mask(seqlens: List[int], max_N: int, device: str = "cuda"):
    """
    根据每个样本的实际长度生成 padding mask。
    seqlens: 长度列表，len == B
    返回形状 [B, 1, 1, max_N]，有效位为 0，padding 位为 -inf
    """
    B = len(seqlens)
    mask = torch.zeros(B, 1, 1, max_N, device=device, dtype=torch.float32)
    for i, sl in enumerate(seqlens):
        if sl < max_N:
            mask[i, :, :, sl:] = float("-inf")
    return mask


def compute_bandwidth(bytes_rw: float, time_s: float) -> float:
    """计算 HBM 带宽 (GB/s)"""
    if time_s <= 0:
        return 0.0
    return bytes_rw / time_s / 1e9
