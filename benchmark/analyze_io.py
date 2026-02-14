"""
IO 复杂度验证与带宽分析

计算理论 IO 量，并提供 ncu 命令模板来对比实际 DRAM 访问量
验证 FlashAttention 的 IO 降低效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import math
import time
from src.kernel.flash_attn_triton import flash_attn_forward
from src.reference import naive_attention
from src.utils import gen_qkv


def theoretical_io_naive(B: int, H: int, N: int, d: int, dtype_bytes: int = 2):
    """
    朴素 attention 的理论 IO 量

   
    读写 S 和 P 各一次
    """
    qkv_io = 3 * B * H * N * d * dtype_bytes  # 读 Q, K, V
    s_io = 2 * B * H * N * N * dtype_bytes     # 写+读 S
    p_io = 2 * B * H * N * N * dtype_bytes     # 写+读 P
    o_io = B * H * N * d * dtype_bytes          # 写 O
    total = qkv_io + s_io + p_io + o_io
    return {
        "Q/K/V读": qkv_io,
        "S读写": s_io,
        "P读写": p_io,
        "O写": o_io,
        "总计": total,
        "N²项": s_io + p_io,
    }


def theoretical_io_flash(B: int, H: int, N: int, d: int, dtype_bytes: int = 2):
    """
    FlashAttention 的理论 IO 量

    读 Q + 读 K + 读 V + 写 O 
    不落盘 S/P 中间矩阵
    """
    total = 4 * B * H * N * d * dtype_bytes
    return {
        "读Q": B * H * N * d * dtype_bytes,
        "读K": B * H * N * d * dtype_bytes,
        "读V": B * H * N * d * dtype_bytes,
        "写O": B * H * N * d * dtype_bytes,
        "总计": total,
        "N²项": 0,
    }


def analyze_io_reduction(B=2, H=4, d=64, seq_lengths=None):
    """分析不同序列长度下的 IO 降低比例"""
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192]

    print("=" * 80)
    print("FlashAttention IO 复杂度理论分析")
    print("=" * 80)
    print(f"配置: B={B}, H={H}, d={d}, dtype=FP16 (2 bytes)")
    print()

    print(f"{'N':>6} | {'朴素IO(MB)':>12} | {'Flash IO(MB)':>12} | {'IO降低':>8} | {'N²项占比':>10}")
    print("-" * 65)

    for N in seq_lengths:
        naive_io = theoretical_io_naive(B, H, N, d)
        flash_io = theoretical_io_flash(B, H, N, d)

        naive_mb = naive_io["总计"] / 1024 / 1024
        flash_mb = flash_io["总计"] / 1024 / 1024
        reduction = 1 - flash_io["总计"] / naive_io["总计"]
        n2_ratio = naive_io["N²项"] / naive_io["总计"]

        print(f"{N:>6} | {naive_mb:>10.1f}MB | {flash_mb:>10.1f}MB | "
              f"{reduction:>7.1%} | {n2_ratio:>9.1%}")

    print()


def benchmark_actual_io(B=2, H=4, d=64, seq_lengths=None, device="cuda"):
    """
    实际运行并测量时间，估算等效带宽

    注：精确 DRAM 字节数需要用 ncu，这里用 cudaEvent 计时 + 理论 IO 估算带宽
    """
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096]

    print("=" * 80)
    print("FlashAttention 实际运行带宽估算")
    print("=" * 80)
    print(f"配置: B={B}, H={H}, d={d}")
    print()

    print(f"{'N':>6} | {'方法':>10} | {'时间(ms)':>10} | {'理论IO(MB)':>12} | {'等效BW(GB/s)':>14}")
    print("-" * 70)

    for N in seq_lengths:
        Q, K, V = gen_qkv(B, H, N, d, device=device)

        # --- 朴素 ---
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        
        for _ in range(3):
            _ = naive_attention(Q, K, V)
        torch.cuda.synchronize()

        start.record()
        for _ in range(10):
            _ = naive_attention(Q, K, V)
        end.record()
        torch.cuda.synchronize()
        naive_time_ms = start.elapsed_time(end) / 10
        naive_io = theoretical_io_naive(B, H, N, d)
        naive_bw = naive_io["总计"] / (naive_time_ms / 1000) / 1e9

        # FlashAttention 
        for _ in range(3):
            _ = flash_attn_forward(Q, K, V, causal=False)
        torch.cuda.synchronize()

        start.record()
        for _ in range(10):
            _ = flash_attn_forward(Q, K, V, causal=False)
        end.record()
        torch.cuda.synchronize()
        flash_time_ms = start.elapsed_time(end) / 10
        flash_io = theoretical_io_flash(B, H, N, d)
        flash_bw = flash_io["总计"] / (flash_time_ms / 1000) / 1e9

        print(f"{N:>6} | {'朴素':>10} | {naive_time_ms:>8.2f}ms | "
              f"{naive_io['总计']/1024/1024:>10.1f}MB | {naive_bw:>12.1f}")
        print(f"{'':>6} | {'Flash':>10} | {flash_time_ms:>8.2f}ms | "
              f"{flash_io['总计']/1024/1024:>10.1f}MB | {flash_bw:>12.1f}")
        print("-" * 70)

    print()


def print_ncu_commands():
    """打印 ncu profiling 命令模板"""
    print("=" * 80)
    print("NCU Profiling 命令（获取实际 DRAM 字节数）")
    print("=" * 80)
    print()
    print("# 运行 ncu 收集 DRAM 指标：")
    print("ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \\")
    print("    --kernel-name _flash_attn_fwd_kernel \\")
    print("    python -c \"")
    print("import torch; import sys; sys.path.insert(0, '.')  ")
    print("from src.kernel.flash_attn_triton import flash_attn_forward")
    print("from src.utils import gen_qkv")
    print("Q,K,V = gen_qkv(2,4,1024,64)")
    print("flash_attn_forward(Q,K,V)")
    print("\"")
    print()
    print("# 对比理论值：")
    print("# Flash IO = 4 * B * H * N * d * 2 bytes")
    print("# 若 ncu 实测 理论值  IO-aware 验证通过")
    print("# 若 ncu 实测 >> 理论值  存在多余 IO")
    print()
    print("# 关键指标解读：")
    print("# dram__throughput     — 是否接近带宽上限")
    print("# dram__bytes_write    — Flash 应大幅低于朴素")
    print("# sm__warp_issue_stalled_long_scoreboard — 内存等待是否减少")
    print("# sm__issue_active     — 计算活跃度是否更高")
    print()


if __name__ == "__main__":
    analyze_io_reduction()
    print()
    benchmark_actual_io()
    print()
    print_ncu_commands()
