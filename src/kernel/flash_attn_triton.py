"""
FlashAttention 前向传播 Triton Kernel

核心思想：
  - Q block 常驻寄存器，K/V block 流式加载
  - Online Softmax：维护 m_i, l_i, acc_i
  - 不保存 N×N 中间矩阵 S/P
  - 支持 Causal Mask + Padding Mask + Block Early Exit
  - 可选 debug 模式：统计被跳过的 block 数
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================
# Autotune 配置
# ============================================================
def _get_configs():
    """返回 autotune 候选配置列表。"""
    configs = [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32},  num_warps=4, num_stages=3),
    ]
    return configs


def _get_configs_debug():
    """debug 模式用固定配置（避免 autotune 干扰统计）。"""
    return [triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3)]


# ============================================================
# 主 Kernel
# ============================================================
@triton.autotune(configs=_get_configs(), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    L_ptr,
    sm_scale,
    Seqlens_k_ptr,
    stride_bh, stride_n, stride_d,
    stride_lbh,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FlashAttention 前向 kernel：一个 program 处理一个 Q block"""
    # 索引
    pid_m = tl.program_id(0)
    bh_id = tl.program_id(1)

    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # 基地址偏移
    base_off = bh_id * stride_bh
    q_base = Q_ptr + base_off
    k_base = K_ptr + base_off
    v_base = V_ptr + base_off
    o_base = Out_ptr + base_off

    # 加载 Q block
    q_ptrs = q_base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    mask_q = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)

    # 初始化 online softmax 状态
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 有效 K 长度
    if HAS_PADDING:
        seqlen_k = tl.load(Seqlens_k_ptr + bh_id)
    else:
        seqlen_k = N_CTX

    # K block 遍历上界
    if IS_CAUSAL:
        kv_len = tl.minimum(q_start + BLOCK_M, seqlen_k)
    else:
        kv_len = seqlen_k

    num_k_blocks = tl.cdiv(kv_len, BLOCK_N)

    # 遍历 K/V blocks
    for j in range(0, num_k_blocks):
        k_start = j * BLOCK_N
        offs_n = k_start + tl.arange(0, BLOCK_N)

        # 加载 K block: [BLOCK_N, HEAD_DIM]
        k_ptrs = k_base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
        mask_k = offs_n[:, None] < seqlen_k
        k = tl.load(k_ptrs, mask=mask_k, other=0.0)

       
        s = tl.dot(q, tl.trans(k))
        s = s * sm_scale

        # 边界 Mask：K 列超出有效范围的设为 -inf
        boundary_mask = offs_n[None, :] < seqlen_k
        s = tl.where(boundary_mask, s, float("-inf"))

        # Causal Mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # Online Softmax 更新
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # 加载 V block: [BLOCK_N, HEAD_DIM]
        v_ptrs = v_base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
        mask_v = offs_n[:, None] < seqlen_k
        v = tl.load(v_ptrs, mask=mask_v, other=0.0)

        acc = acc + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # 归一化
    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l[:, None]
    acc = tl.where(l_i[:, None] == 0.0, 0.0, acc)

    # 存储 logsumexp: L = m + log(l)
    l_log = tl.where(l_i == 0.0, 0.0, tl.log(l_i))
    L_val = tl.where(m_i == float("-inf"), float("-inf"), m_i + l_log)
    l_base = L_ptr + bh_id * stride_lbh
    l_ptrs = l_base + offs_m
    mask_l = offs_m < N_CTX
    tl.store(l_ptrs, L_val, mask=mask_l)

    # 写出
    o_ptrs = o_base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    mask_o = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_o)


# ============================================================
# Debug Kernel
# ============================================================
@triton.autotune(configs=_get_configs_debug(), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _flash_attn_fwd_kernel_debug(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    L_ptr,
    sm_scale,
    Seqlens_k_ptr,
    # 统计输出：每个 program 写一个跳过计数
    Skip_count_ptr,
    stride_bh, stride_n, stride_d,
    stride_lbh,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """带 early-exit 统计的 debug kernel。"""
    pid_m = tl.program_id(0)
    bh_id = tl.program_id(1)

    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    base_off = bh_id * stride_bh
    q_base = Q_ptr + base_off
    k_base = K_ptr + base_off
    v_base = V_ptr + base_off
    o_base = Out_ptr + base_off

    q_ptrs = q_base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    mask_q = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)

    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if HAS_PADDING:
        seqlen_k = tl.load(Seqlens_k_ptr + bh_id)
    else:
        seqlen_k = N_CTX

    # 总 block 数和跳过计数
    total_k_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    skipped: tl.int32 = 0

    if IS_CAUSAL:
        kv_len = tl.minimum(q_start + BLOCK_M, seqlen_k)
    else:
        kv_len = seqlen_k

    num_k_blocks = tl.cdiv(kv_len, BLOCK_N)

    # 被 early exit 跳过的 = 总块数 - 实际遍历块数
    skipped = total_k_blocks - num_k_blocks

    for j in range(0, num_k_blocks):
        k_start = j * BLOCK_N
        offs_n = k_start + tl.arange(0, BLOCK_N)

        k_ptrs = k_base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
        mask_k = offs_n[:, None] < seqlen_k
        k = tl.load(k_ptrs, mask=mask_k, other=0.0)

        s = tl.dot(q, tl.trans(k))
        s = s * sm_scale

        # 边界 Mask
        boundary_mask = offs_n[None, :] < seqlen_k
        s = tl.where(boundary_mask, s, float("-inf"))

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
        mask_v = offs_n[:, None] < seqlen_k
        v = tl.load(v_ptrs, mask=mask_v, other=0.0)

        acc = acc + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l[:, None]
    acc = tl.where(l_i[:, None] == 0.0, 0.0, acc)

    # 存储 logsumexp
    l_log = tl.where(l_i == 0.0, 0.0, tl.log(l_i))
    L_val = tl.where(m_i == float("-inf"), float("-inf"), m_i + l_log)
    l_base = L_ptr + bh_id * stride_lbh
    l_ptrs = l_base + offs_m
    mask_l = offs_m < N_CTX
    tl.store(l_ptrs, L_val, mask=mask_l)

    o_ptrs = o_base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    mask_o = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_o)

    # 写出跳过计数
    num_q_blocks = tl.cdiv(N_CTX, BLOCK_M)
    skip_idx = bh_id * num_q_blocks + pid_m
    tl.store(Skip_count_ptr + skip_idx, skipped)


# ============================================================
# Python 调用入口
# ============================================================
def flash_attn_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    seqlens_k: torch.Tensor = None,
) -> torch.Tensor:
    """
    FlashAttention 前向传播。

    参数:
        Q, K, V: [B, H, N, d]，float16
        causal: 是否启用 causal mask
        seqlens_k: [B]，int32，每个 batch 的有效 K 长度（None 表示无 padding）

    返回:
        O: [B, H, N, d]
    """
    B, H, N, d = Q.shape
    assert K.shape == V.shape == Q.shape, "Q/K/V 形状必须一致"
    assert Q.is_cuda, "需要 CUDA 张量"

    Out = torch.empty_like(Q)
    # logsumexp 存储: [B*H, N], fp32
    L = torch.empty(B * H, N, dtype=torch.float32, device=Q.device)
    sm_scale = 1.0 / math.sqrt(d)

    # reshape 为 [B*H, N, d]
    q_flat = Q.reshape(B * H, N, d)
    k_flat = K.reshape(B * H, N, d)
    v_flat = V.reshape(B * H, N, d)
    o_flat = Out.reshape(B * H, N, d)

    has_padding = seqlens_k is not None
    if has_padding:
        seqlens_k = seqlens_k.to(torch.int32).to(Q.device)
        seqlens_k_expanded = seqlens_k.unsqueeze(1).expand(B, H).reshape(B * H).contiguous()
    else:
        seqlens_k_expanded = torch.empty(0, dtype=torch.int32, device=Q.device)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)

    _flash_attn_fwd_kernel[grid](
        q_flat, k_flat, v_flat, o_flat,
        L,
        sm_scale,
        seqlens_k_expanded,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        L.stride(0),
        N_CTX=N,
        HEAD_DIM=d,
        IS_CAUSAL=causal,
        HAS_PADDING=has_padding,
    )

    return Out, L


def flash_attn_forward_debug(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    seqlens_k: torch.Tensor = None,
):
    """
    带 early-exit 统计的 debug 版本。

    返回:
        O: [B, H, N, d]
        skip_counts: [B*H, num_q_blocks]，每个 Q block 跳过的 K block 数
    """
    B, H, N, d = Q.shape
    assert K.shape == V.shape == Q.shape
    assert Q.is_cuda

    Out = torch.empty_like(Q)
    L = torch.empty(B * H, N, dtype=torch.float32, device=Q.device)
    sm_scale = 1.0 / math.sqrt(d)

    q_flat = Q.reshape(B * H, N, d)
    k_flat = K.reshape(B * H, N, d)
    v_flat = V.reshape(B * H, N, d)
    o_flat = Out.reshape(B * H, N, d)

    has_padding = seqlens_k is not None
    if has_padding:
        seqlens_k = seqlens_k.to(torch.int32).to(Q.device)
        seqlens_k_expanded = seqlens_k.unsqueeze(1).expand(B, H).reshape(B * H).contiguous()
    else:
        seqlens_k_expanded = torch.empty(0, dtype=torch.int32, device=Q.device)

    BLOCK_M = 64  # debug 模式固定
    num_q_blocks = math.ceil(N / BLOCK_M)
    skip_counts = torch.zeros(B * H * num_q_blocks, dtype=torch.int32, device=Q.device)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)

    _flash_attn_fwd_kernel_debug[grid](
        q_flat, k_flat, v_flat, o_flat,
        L,
        sm_scale,
        seqlens_k_expanded,
        skip_counts,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        L.stride(0),
        N_CTX=N,
        HEAD_DIM=d,
        IS_CAUSAL=causal,
        HAS_PADDING=has_padding,
    )

    skip_counts = skip_counts.reshape(B * H, num_q_blocks)
    return Out, L, skip_counts
