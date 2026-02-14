"""正确性验证：FlashAttention vs 参考实现"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
import math
from src.kernel.flash_attn_triton import flash_attn_forward, flash_attn_forward_debug
from src.reference import naive_attention, sdpa_attention
from src.utils import gen_qkv, make_causal_mask, make_padding_mask


RTOL = 1e-2
ATOL = 1e-2
DEVICE = "cuda"


def _check_close(out, ref, msg=""):
    """检查两个张量是否足够接近"""
    out_f, ref_f = out.float(), ref.float()
    if not torch.allclose(out_f, ref_f, rtol=RTOL, atol=ATOL):
        diff = (out_f - ref_f).abs()
        pytest.fail(f"{msg} max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")


# ============================================================
# 基础正确性
# ============================================================
class TestBasicCorrectness:
    """随机输入 vs SDPA / naive。"""

    @pytest.mark.parametrize("N", [128, 256, 512])
    @pytest.mark.parametrize("d", [64, 128])
    def test_no_mask(self, N, d):
        B, H = 2, 4
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        out = flash_attn_forward(Q, K, V, causal=False)
        ref = sdpa_attention(Q, K, V, is_causal=False)
        _check_close(out, ref, f"no_mask N={N} d={d}")

    @pytest.mark.parametrize("N", [128, 256, 512])
    def test_vs_naive(self, N):
        B, H, d = 2, 4, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        out = flash_attn_forward(Q, K, V, causal=False)
        ref = naive_attention(Q, K, V)
        _check_close(out, ref, f"vs_naive N={N}")


# ============================================================
# Causal Mask
# ============================================================
class TestCausalMask:

    @pytest.mark.parametrize("N", [128, 256, 512])
    @pytest.mark.parametrize("d", [64, 128])
    def test_causal(self, N, d):
        B, H = 2, 4
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        out = flash_attn_forward(Q, K, V, causal=True)
        ref = sdpa_attention(Q, K, V, is_causal=True)
        _check_close(out, ref, f"causal N={N} d={d}")

    @pytest.mark.parametrize("N", [128, 256])
    def test_causal_vs_naive(self, N):
        B, H, d = 2, 4, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        mask = make_causal_mask(N, device=DEVICE)
        out = flash_attn_forward(Q, K, V, causal=True)
        ref = naive_attention(Q, K, V, mask=mask)
        _check_close(out, ref, f"causal_vs_naive N={N}")


# ============================================================
# Padding Mask
# ============================================================
class TestPaddingMask:

    def test_padding_basic(self):
        B, H, N, d = 2, 4, 256, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        seqlens = [200, 150]
        seqlens_k = torch.tensor(seqlens, dtype=torch.int32, device=DEVICE)
        out = flash_attn_forward(Q, K, V, causal=False, seqlens_k=seqlens_k)
        pad_mask = make_padding_mask(seqlens, N, device=DEVICE)
        ref = naive_attention(Q, K, V, mask=pad_mask)
        _check_close(out, ref, "padding_basic")

    def test_padding_full_length(self):
        B, H, N, d = 2, 4, 128, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        seqlens_k = torch.tensor([N, N], dtype=torch.int32, device=DEVICE)
        out = flash_attn_forward(Q, K, V, causal=False, seqlens_k=seqlens_k)
        ref = naive_attention(Q, K, V)
        _check_close(out, ref, "padding_full_length")


# ============================================================
# 极端情况
# ============================================================
class TestNumericalEdgeCases:

    def test_all_masked_row(self):
        B, H, N, d = 1, 1, 64, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        out = flash_attn_forward(Q, K, V, causal=True)
        ref = sdpa_attention(Q, K, V, is_causal=True)
        _check_close(out, ref, "all_masked_row")

    def test_large_values(self):
        """输入含较大值 — Q*10 会放大 FP16 舍入误差，容差适度放宽。"""
        B, H, N, d = 1, 2, 128, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        Q = Q * 10.0
        out = flash_attn_forward(Q, K, V, causal=False)
        ref = naive_attention(Q, K, V)
        # 放大输入导致 FP16 累积误差增大，放宽至 atol=0.02
        out_f, ref_f = out.float(), ref.float()
        diff = (out_f - ref_f).abs()
        assert diff.max().item() < 0.02, \
            f"large_values max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}"

    def test_single_token(self):
        B, H, N, d = 2, 4, 1, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        out = flash_attn_forward(Q, K, V, causal=False)
        ref = naive_attention(Q, K, V)
        _check_close(out, ref, "single_token")


# ============================================================
# FP32 精度对比
# ============================================================
class TestFP32Precision:
    """用 FP32 参考计算，区分 FP16 量化误差与算法错误"""

    @pytest.mark.parametrize("N", [128, 256])
    def test_fp32_reference(self, N):
        """FP16 kernel 输出 vs FP32 naive，误差应在 FP16 精度范围内"""
        B, H, d = 2, 4, 64
        # FP32 输入
        Q32 = torch.randn(B, H, N, d, dtype=torch.float32, device=DEVICE)
        K32 = torch.randn(B, H, N, d, dtype=torch.float32, device=DEVICE)
        V32 = torch.randn(B, H, N, d, dtype=torch.float32, device=DEVICE)

        # FP32 参考
        ref_f32 = naive_attention(Q32, K32, V32)

        # FP16 kernel
        Q16, K16, V16 = Q32.half(), K32.half(), V32.half()
        out_f16 = flash_attn_forward(Q16, K16, V16, causal=False)

        # 误差分析
        diff = (out_f16.float() - ref_f32).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\n[FP32对比] N={N}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        # FP16 精度上限约 1e-3 级别，这里容许略大（累积误差）
        assert max_diff < 0.1, f"FP32 对比误差过大: max_diff={max_diff}"

    @pytest.mark.parametrize("N", [128, 256])
    def test_fp32_causal(self, N):
        """Causal 场景下的 FP32 对比。"""
        B, H, d = 2, 4, 64
        Q32 = torch.randn(B, H, N, d, dtype=torch.float32, device=DEVICE)
        K32 = torch.randn(B, H, N, d, dtype=torch.float32, device=DEVICE)
        V32 = torch.randn(B, H, N, d, dtype=torch.float32, device=DEVICE)

        mask = make_causal_mask(N, device=DEVICE)
        ref_f32 = naive_attention(Q32, K32, V32, mask=mask)

        Q16, K16, V16 = Q32.half(), K32.half(), V32.half()
        out_f16 = flash_attn_forward(Q16, K16, V16, causal=True)

        diff = (out_f16.float() - ref_f32).abs()
        max_diff = diff.max().item()
        print(f"\n[FP32 Causal] N={N}: max_diff={max_diff:.6f}")
        assert max_diff < 0.1, f"FP32 causal 误差过大: max_diff={max_diff}"


# ============================================================
# Early Exit 统计验证
# ============================================================
class TestEarlyExitStatistics:
    """验证 causal mask 下 early exit 跳过比例接近理论值。"""

    def test_causal_skip_ratio(self):
        """N=2048, causal=True 时跳过比例应接近 50%。"""
        B, H, N, d = 1, 1, 2048, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)

        out, skip_counts = flash_attn_forward_debug(Q, K, V, causal=True)

        BLOCK_M = 64
        BLOCK_N = 64
        num_q_blocks = math.ceil(N / BLOCK_M)
        total_k_blocks = math.ceil(N / BLOCK_N)

        total_possible = num_q_blocks * total_k_blocks  # 所有 program 计算的总 block 数
        total_skipped = skip_counts.sum().item()

        skip_ratio = total_skipped / total_possible
        print(f"\n[Early Exit] N={N}: 跳过 {total_skipped}/{total_possible} blocks, "
              f"比例={skip_ratio:.2%}")

        # 理论上 causal 跳过约 50%（上三角）
        assert skip_ratio > 0.3, f"跳过比例过低: {skip_ratio:.2%}，预期 >30%"
        assert skip_ratio < 0.7, f"跳过比例过高: {skip_ratio:.2%}，预期 <70%"

        # 验证输出正确性不受影响
        ref = sdpa_attention(Q, K, V, is_causal=True)
        _check_close(out, ref, "early_exit_correctness")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
