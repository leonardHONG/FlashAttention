"""反向传播正确性验证: FlashAttention backward vs PyTorch SDPA"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
import math
from src.functional import flash_attention
from src.reference import sdpa_attention
from src.utils import gen_qkv


RTOL = 1e-2
ATOL = 1e-2
DEVICE = "cuda"


def _check_close(out, ref, msg=""):
    """检查两个张量是否足够接近"""
    out_f, ref_f = out.float(), ref.float()
    if not torch.allclose(out_f, ref_f, rtol=RTOL, atol=ATOL):
        diff = (out_f - ref_f).abs()
        pytest.fail(f"{msg} max_diff={diff.max().item():.6f}")


def _get_sdpa_grads(Q, K, V, causal=False):
    """用 PyTorch SDPA 计算参考梯度"""
    Q = Q.clone().detach().requires_grad_(True)
    K = K.clone().detach().requires_grad_(True)
    V = V.clone().detach().requires_grad_(True)
    O = sdpa_attention(Q, K, V, is_causal=causal)
    dO = torch.randn_like(O)
    O.backward(dO)
    return dO, Q.grad, K.grad, V.grad


def _get_flash_grads(Q, K, V, dO, causal=False, seqlens_k=None):
    """用 FlashAttention 计算梯度"""
    Q = Q.clone().detach().requires_grad_(True)
    K = K.clone().detach().requires_grad_(True)
    V = V.clone().detach().requires_grad_(True)
    O = flash_attention(Q, K, V, causal=causal, seqlens_k=seqlens_k)
    O.backward(dO)
    return Q.grad, K.grad, V.grad


# ============================================================
# 基础反向传播
# ============================================================
class TestBackwardBasic:
    """随机输入 vs SDPA backward"""

    @pytest.mark.parametrize("N", [64, 128, 256])
    @pytest.mark.parametrize("d", [64])
    def test_no_mask(self, N, d):
        B, H = 2, 4
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        dO, ref_dQ, ref_dK, ref_dV = _get_sdpa_grads(Q, K, V, causal=False)
        flash_dQ, flash_dK, flash_dV = _get_flash_grads(Q, K, V, dO, causal=False)
        _check_close(flash_dQ, ref_dQ, f"dQ no_mask N={N} d={d}")
        _check_close(flash_dK, ref_dK, f"dK no_mask N={N} d={d}")
        _check_close(flash_dV, ref_dV, f"dV no_mask N={N} d={d}")


# ============================================================
# Causal Mask 反向传播
# ============================================================
class TestBackwardCausal:
    """causal mask 下的梯度验证"""

    @pytest.mark.parametrize("N", [64, 128, 256])
    def test_causal(self, N):
        B, H, d = 2, 4, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        dO, ref_dQ, ref_dK, ref_dV = _get_sdpa_grads(Q, K, V, causal=True)
        flash_dQ, flash_dK, flash_dV = _get_flash_grads(Q, K, V, dO, causal=True)
        _check_close(flash_dQ, ref_dQ, f"dQ causal N={N}")
        _check_close(flash_dK, ref_dK, f"dK causal N={N}")
        _check_close(flash_dV, ref_dV, f"dV causal N={N}")


# ============================================================
# Padding Mask 反向传播
# ============================================================
class TestBackwardPadding:
    """padding mask 下的梯度验证, 用 naive attention 做参考"""

    @pytest.mark.parametrize("seqlens", [[200, 150], [128, 256]])
    def test_padding(self, seqlens):
        B = len(seqlens)
        H, N, d = 4, 256, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        seqlens_k = torch.tensor(seqlens, dtype=torch.int32, device=DEVICE)

        # flash backward
        Q_f = Q.clone().detach().requires_grad_(True)
        K_f = K.clone().detach().requires_grad_(True)
        V_f = V.clone().detach().requires_grad_(True)
        O_f = flash_attention(Q_f, K_f, V_f, causal=False, seqlens_k=seqlens_k)
        dO = torch.randn_like(O_f)
        O_f.backward(dO)

        # naive 参考: 用 padding mask 构造
        from src.utils import make_padding_mask
        from src.reference import naive_attention
        pad_mask = make_padding_mask(seqlens, N, device=DEVICE)
        Q_n = Q.clone().detach().float().requires_grad_(True)
        K_n = K.clone().detach().float().requires_grad_(True)
        V_n = V.clone().detach().float().requires_grad_(True)
        O_n = naive_attention(Q_n, K_n, V_n, mask=pad_mask)
        O_n.backward(dO.float())

        _check_close(Q_f.grad, Q_n.grad.half(), f"dQ padding seqlens={seqlens}")
        _check_close(K_f.grad, K_n.grad.half(), f"dK padding seqlens={seqlens}")
        _check_close(V_f.grad, V_n.grad.half(), f"dV padding seqlens={seqlens}")


# ============================================================
# 极端情况
# ============================================================
class TestBackwardEdgeCases:
    """极端情况下的梯度验证"""

    def test_single_token(self):
        B, H, N, d = 2, 4, 1, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        dO, ref_dQ, ref_dK, ref_dV = _get_sdpa_grads(Q, K, V, causal=False)
        flash_dQ, flash_dK, flash_dV = _get_flash_grads(Q, K, V, dO, causal=False)
        _check_close(flash_dQ, ref_dQ, "dQ single_token")
        _check_close(flash_dK, ref_dK, "dK single_token")
        _check_close(flash_dV, ref_dV, "dV single_token")

    def test_all_masked_padding(self):
        """seqlen_k=0 时所有梯度应为零"""
        B, H, N, d = 1, 2, 64, 64
        Q, K, V = gen_qkv(B, H, N, d, device=DEVICE)
        seqlens_k = torch.tensor([0], dtype=torch.int32, device=DEVICE)

        Q_f = Q.clone().detach().requires_grad_(True)
        K_f = K.clone().detach().requires_grad_(True)
        V_f = V.clone().detach().requires_grad_(True)
        O_f = flash_attention(Q_f, K_f, V_f, causal=False, seqlens_k=seqlens_k)
        dO = torch.randn_like(O_f)
        O_f.backward(dO)

        zero = torch.zeros_like(Q)
        _check_close(Q_f.grad, zero, "dQ all_masked")
        _check_close(K_f.grad, zero, "dK all_masked")
        _check_close(V_f.grad, zero, "dV all_masked")


# ============================================================
# Autograd gradcheck
# ============================================================
class TestGradcheck:
    """用 torch.autograd.gradcheck 做数值梯度检查"""

    def test_gradcheck_small(self):
        """小张量 float64 gradcheck"""
        B, H, N, d = 1, 1, 16, 16
        Q = torch.randn(B, H, N, d, dtype=torch.float64, device=DEVICE, requires_grad=True)
        K = torch.randn(B, H, N, d, dtype=torch.float64, device=DEVICE, requires_grad=True)
        V = torch.randn(B, H, N, d, dtype=torch.float64, device=DEVICE, requires_grad=True)

        from src.kernel.flash_attn_triton import flash_attn_forward
        from src.kernel.flash_attn_bwd_triton import flash_attn_backward

        def func(q, k, v):
            out, L = flash_attn_forward(q, k, v, causal=False)
            return out

        assert torch.autograd.gradcheck(
            func, (Q, K, V),
            eps=1e-6, atol=1e-3, rtol=1e-3,
            nondet_tol=1e-3,
        ), "gradcheck failed"

    def test_gradcheck_causal(self):
        """causal 模式 gradcheck"""
        B, H, N, d = 1, 1, 16, 16
        Q = torch.randn(B, H, N, d, dtype=torch.float64, device=DEVICE, requires_grad=True)
        K = torch.randn(B, H, N, d, dtype=torch.float64, device=DEVICE, requires_grad=True)
        V = torch.randn(B, H, N, d, dtype=torch.float64, device=DEVICE, requires_grad=True)

        def func(q, k, v):
            return flash_attention(q, k, v, causal=True)

        assert torch.autograd.gradcheck(
            func, (Q, K, V),
            eps=1e-6, atol=1e-3, rtol=1e-3,
            nondet_tol=1e-3,
        ), "causal gradcheck failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
