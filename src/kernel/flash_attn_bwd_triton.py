"""
FlashAttention 反向传播 Triton Kernel

2-kernel 架构避免 atomic 冲突:
  - Kernel A: 以 k_block 为并行轴, 累加 dK 和 dV
  - Kernel B: 以 q_block 为并行轴, 累加 dQ
  - Forward 保存 logsumexp L, Backward 用 L 重建 P
  - 不显式构建 N*N attention matrix
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================
# Autotune 配置
# ============================================================
def _get_bwd_configs():
    """反向 kernel autotune 候选配置"""
    configs = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    ]
    return configs


# ============================================================
# Kernel A: dK + dV
# ============================================================
@triton.autotune(configs=_get_bwd_configs(), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _flash_attn_bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, dO_ptr,
    L_ptr,
    dK_ptr, dV_ptr,
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
    """以 k_block 为并行轴, 遍历 q_block 累加 dK 和 dV"""
    pid_n = tl.program_id(0)
    bh_id = tl.program_id(1)

    k_start = pid_n * BLOCK_N
    offs_n = k_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    base_off = bh_id * stride_bh

    # 有效 K 长度
    if HAS_PADDING:
        seqlen_k = tl.load(Seqlens_k_ptr + bh_id)
    else:
        seqlen_k = N_CTX

    # 加载 K block
    k_ptrs = K_ptr + base_off + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
    mask_k = offs_n[:, None] < seqlen_k
    k = tl.load(k_ptrs, mask=mask_k, other=0.0)

    # 加载 V block
    v_ptrs = V_ptr + base_off + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
    v = tl.load(v_ptrs, mask=mask_k, other=0.0)

    # fp32 累加器
    dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # 遍历所有 q_block
    num_q_blocks = tl.cdiv(N_CTX, BLOCK_M)

    # causal 时只需遍历 q_start >= k_start 的 q_block
    if IS_CAUSAL:
        q_block_start = k_start // BLOCK_M
    else:
        q_block_start = 0

    for i in range(q_block_start, num_q_blocks):
        q_start = i * BLOCK_M
        offs_m = q_start + tl.arange(0, BLOCK_M)

        # 加载 Q block
        q_ptrs = Q_ptr + base_off + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
        mask_q = offs_m[:, None] < N_CTX
        q = tl.load(q_ptrs, mask=mask_q, other=0.0)

        # 加载 dO block
        do_ptrs = dO_ptr + base_off + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
        do = tl.load(do_ptrs, mask=mask_q, other=0.0)

        # 加载 L block
        l_ptrs = L_ptr + bh_id * stride_lbh + offs_m
        mask_l = offs_m < N_CTX
        l_i = tl.load(l_ptrs, mask=mask_l, other=float("-inf"))

        # 加载 Out block
        o_ptrs = Out_ptr + base_off + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
        o = tl.load(o_ptrs, mask=mask_q, other=0.0)

        # 重建 logits: S = Q @ K^T * scale
        s = tl.dot(q, tl.trans(k))
        s = s * sm_scale

        # 边界 mask
        boundary_mask = offs_n[None, :] < seqlen_k
        s = tl.where(boundary_mask, s, float("-inf"))

        # causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # 恢复 P = exp(S - L)
        p = tl.exp(s - l_i[:, None])
        p = tl.where(s == float("-inf"), 0.0, p)

        dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)
        dp = tl.dot(do, tl.trans(v))       
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        ds = p * (dp - delta[:, None])
        dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale

    # 写出 dK
    dk_ptrs = dK_ptr + base_off + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
    mask_out = offs_n[:, None] < N_CTX
    tl.store(dk_ptrs, dk_acc.to(dK_ptr.dtype.element_ty), mask=mask_out)

    # 写出 dV
    dv_ptrs = dV_ptr + base_off + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
    tl.store(dv_ptrs, dv_acc.to(dV_ptr.dtype.element_ty), mask=mask_out)


# ============================================================
# Kernel B: dQ
# ============================================================
@triton.autotune(configs=_get_bwd_configs(), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _flash_attn_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, dO_ptr,
    L_ptr,
    dQ_ptr,
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
    """以 q_block 为并行轴, 遍历 k_block 累加 dQ"""
    pid_m = tl.program_id(0)
    bh_id = tl.program_id(1)

    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    base_off = bh_id * stride_bh

    # 有效 K 长度
    if HAS_PADDING:
        seqlen_k = tl.load(Seqlens_k_ptr + bh_id)
    else:
        seqlen_k = N_CTX

    # 加载 Q block
    q_ptrs = Q_ptr + base_off + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    mask_q = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)

    # 加载 dO block
    do_ptrs = dO_ptr + base_off + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    do = tl.load(do_ptrs, mask=mask_q, other=0.0)

    # 加载 L block
    l_ptrs = L_ptr + bh_id * stride_lbh + offs_m
    mask_l = offs_m < N_CTX
    l_i = tl.load(l_ptrs, mask=mask_l, other=float("-inf"))

    # 加载 Out block
    o_ptrs = Out_ptr + base_off + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    o = tl.load(o_ptrs, mask=mask_q, other=0.0)

    # delta = rowsum(dO * O)
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)

    # fp32 累加器
    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # K block 遍历上界
    if IS_CAUSAL:
        kv_len = tl.minimum(q_start + BLOCK_M, seqlen_k)
    else:
        kv_len = seqlen_k

    num_k_blocks = tl.cdiv(kv_len, BLOCK_N)

    for j in range(0, num_k_blocks):
        k_start = j * BLOCK_N
        offs_n = k_start + tl.arange(0, BLOCK_N)

        # 加载 K block
        k_ptrs = K_ptr + base_off + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
        mask_k = offs_n[:, None] < seqlen_k
        k = tl.load(k_ptrs, mask=mask_k, other=0.0)

        # 加载 V block
        v_ptrs = V_ptr + base_off + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
        v = tl.load(v_ptrs, mask=mask_k, other=0.0)

        # 重建 logits
        s = tl.dot(q, tl.trans(k))
        s = s * sm_scale

        # 边界 mask
        boundary_mask = offs_n[None, :] < seqlen_k
        s = tl.where(boundary_mask, s, float("-inf"))

        # causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # 恢复 P, -inf 修正
        p = tl.exp(s - l_i[:, None])
        p = tl.where(s == float("-inf"), 0.0, p)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq_acc += tl.dot(ds.to(k.dtype), k) * sm_scale

    # 写出 dQ
    dq_ptrs = dQ_ptr + base_off + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    mask_out = offs_m[:, None] < N_CTX
    tl.store(dq_ptrs, dq_acc.to(dQ_ptr.dtype.element_ty), mask=mask_out)


# ============================================================
# Python 调用入口
# ============================================================
def flash_attn_backward(
    dO: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Out: torch.Tensor,
    L: torch.Tensor,
    causal: bool = False,
    seqlens_k: torch.Tensor = None,
):
    """
    FlashAttention 反向传播

    参数:
        dO: [B, H, N, d] 上游梯度
        Q, K, V: [B, H, N, d] 前向输入
        Out: [B, H, N, d] 前向输出
        L: [B*H, N] logsumexp
        causal: 是否启用 causal mask
        seqlens_k: [B] int32 每个 batch 的有效 K 长度

    返回:
        dQ, dK, dV: [B, H, N, d]
    """
    B, H, N, d = Q.shape
    assert dO.shape == Q.shape
    assert K.shape == V.shape == Q.shape
    assert Out.shape == Q.shape
    assert L.shape == (B * H, N)

    sm_scale = 1.0 / math.sqrt(d)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # reshape 为 [B*H, N, d]
    q_flat = Q.reshape(B * H, N, d)
    k_flat = K.reshape(B * H, N, d)
    v_flat = V.reshape(B * H, N, d)
    o_flat = Out.reshape(B * H, N, d)
    do_flat = dO.reshape(B * H, N, d)
    dq_flat = dQ.reshape(B * H, N, d)
    dk_flat = dK.reshape(B * H, N, d)
    dv_flat = dV.reshape(B * H, N, d)

    has_padding = seqlens_k is not None
    if has_padding:
        seqlens_k = seqlens_k.to(torch.int32).to(Q.device)
        seqlens_k_expanded = seqlens_k.unsqueeze(1).expand(B, H).reshape(B * H).contiguous()
    else:
        seqlens_k_expanded = torch.empty(0, dtype=torch.int32, device=Q.device)

    # Kernel A: dK + dV, 以 k_block 为并行轴
    grid_a = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), B * H)

    _flash_attn_bwd_dk_dv_kernel[grid_a](
        q_flat, k_flat, v_flat, o_flat, do_flat,
        L,
        dk_flat, dv_flat,
        sm_scale,
        seqlens_k_expanded,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        L.stride(0),
        N_CTX=N,
        HEAD_DIM=d,
        IS_CAUSAL=causal,
        HAS_PADDING=has_padding,
    )

    # Kernel B: dQ, 以 q_block 为并行轴
    grid_b = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)

    _flash_attn_bwd_dq_kernel[grid_b](
        q_flat, k_flat, v_flat, o_flat, do_flat,
        L,
        dq_flat,
        sm_scale,
        seqlens_k_expanded,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        L.stride(0),
        N_CTX=N,
        HEAD_DIM=d,
        IS_CAUSAL=causal,
        HAS_PADDING=has_padding,
    )

    return dQ, dK, dV
