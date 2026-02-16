"""Mask 专项测试：Causal / Padding / 混合"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.kernel.flash_attn_triton import flash_attn_forward
from src.reference import naive_attention, sdpa_attention
from src.utils import gen_qkv, make_causal_mask, make_padding_mask


RTOL = 1e-2
ATOL = 1e-2
DEVICE = "cuda"


def _check_close(out, ref, msg=""):
    out_f, ref_f = out.float(), ref.float()
    if not torch.allclose(out_f, ref_f, rtol=RTOL, atol=ATOL):
        diff = (out_f - ref_f).abs()
        pytest.fail(f"{msg} max_diff={diff.max().item():.6f}")


class TestCausalBlockEarlyExit:
    """验证 Causal Mask 下 block-level early exit 不影响结果"""

    @pytest.mark.parametrize("N", [64, 128, 256, 512])
    def test_causal_various_sizes(self, N):
        B, H, d = 2, 4, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)

        out, _ = flash_attn_forward(Q, K, V, causal=True)
        ref = sdpa_attention(Q, K, V, is_causal=True)
        _check_close(out, ref, f"causal_early_exit N={N}")


class TestPaddingVariousLengths:
    """不同长度组合的 Padding Mask"""

    @pytest.mark.parametrize("seqlens", [
        [100, 200],
        [50, 256],
        [256, 256],
        [1, 128],
    ])
    def test_padding_lengths(self, seqlens):
        B = len(seqlens)
        H, N, d = 4, 256, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        seqlens_k = torch.tensor(seqlens, dtype=torch.int32, device=DEVICE)

        out, _ = flash_attn_forward(Q, K, V, causal=False, seqlens_k=seqlens_k)

        pad_mask = make_padding_mask(seqlens, N, device=DEVICE)
        ref = naive_attention(Q, K, V, mask=pad_mask)
        _check_close(out, ref, f"padding seqlens={seqlens}")


class TestMixedCausalPadding:
    """混合 Causal + Padding（手动组合 mask 作为参考）"""

    def test_causal_with_padding(self):
        B, H, N, d = 2, 4, 256, 64
        seqlens = [200, 150]
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        seqlens_k = torch.tensor(seqlens, dtype=torch.int32, device=DEVICE)

        out, _ = flash_attn_forward(Q, K, V, causal=True, seqlens_k=seqlens_k)

        # 参考：Causal + Padding 合并 mask
        causal_m = make_causal_mask(N, device=DEVICE)          # [N, N]
        pad_m = make_padding_mask(seqlens, N, device=DEVICE)   # [B, 1, 1, N]
        combined = causal_m.unsqueeze(0).unsqueeze(0) + pad_m  # 广播到 [B, 1, N, N]
        ref = naive_attention(Q, K, V, mask=combined)

        _check_close(out, ref, "causal+padding")


class TestAllMaskedRows:
    """全 mask 行输出应为 0"""

    def test_padding_zero_length(self):
        """seqlen_k=0 → 所有行全 mask → 输出全零"""
        B, H, N, d = 1, 2, 64, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        seqlens_k = torch.tensor([0], dtype=torch.int32, device=DEVICE)

        out, _ = flash_attn_forward(Q, K, V, causal=False, seqlens_k=seqlens_k)

        # 所有行应为 0（无有效 K）
        assert torch.allclose(out.float(), torch.zeros_like(out).float(), atol=1e-6), \
            "seqlen_k=0 时输出应为全零"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
