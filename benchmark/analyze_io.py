"""
IO 复杂度验证与带宽分析

功能:
  - 理论 IO 计算并输出 CSV
  - 实测带宽估算并输出 CSV
  - 生成 IO vs N 和 BW vs N 图表
  - ncu 命令模板
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import math
import csv
from src.kernel.flash_attn_triton import flash_attn_forward
from src.reference import naive_attention
from src.utils import gen_qkv

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================
# 理论 IO 计算
# ============================================================
def theoretical_io_naive(B, H, N, d, dtype_bytes=2):
    """朴素 attention 理论 IO 量"""
    qkv_io = 3 * B * H * N * d * dtype_bytes   # 读 Q K V
    s_io = 2 * B * H * N * N * dtype_bytes      # 写+读 S
    p_io = 2 * B * H * N * N * dtype_bytes      # 写+读 P
    o_io = B * H * N * d * dtype_bytes           # 写 O
    total = qkv_io + s_io + p_io + o_io
    return {
        "qkv_read": qkv_io,
        "s_rw": s_io,
        "p_rw": p_io,
        "o_write": o_io,
        "total": total,
        "n2_term": s_io + p_io,
    }


def theoretical_io_flash(B, H, N, d, dtype_bytes=2):
    """FlashAttention 理论 IO 量"""
    total = 4 * B * H * N * d * dtype_bytes
    return {
        "q_read": B * H * N * d * dtype_bytes,
        "k_read": B * H * N * d * dtype_bytes,
        "v_read": B * H * N * d * dtype_bytes,
        "o_write": B * H * N * d * dtype_bytes,
        "total": total,
        "n2_term": 0,
    }


# ============================================================
# 理论分析 + CSV 输出
# ============================================================
def analyze_io_reduction(B=2, H=4, d=64, seq_lengths=None,
                        output_csv="benchmark/results/theoretical_io.csv"):
    """分析不同 N 下的 IO 降低比例, 输出 CSV"""
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192]

    print("=" * 80)
    print("FlashAttention IO Complexity Analysis")
    print("=" * 80)
    print(f"Config: B={B} H={H} d={d} dtype=FP16")
    print()

    header = ["B", "H", "N", "D", "naive_io_bytes", "flash_io_bytes",
              "naive_io_mb", "flash_io_mb", "io_reduction", "n2_ratio"]
    rows = []

    print(f"{'N':>6} | {'Naive IO MB':>12} | {'Flash IO MB':>12} | "
          f"{'Reduction':>10} | {'N2 Ratio':>10}")
    print("-" * 65)

    for N in seq_lengths:
        naive_io = theoretical_io_naive(B, H, N, d)
        flash_io = theoretical_io_flash(B, H, N, d)

        naive_mb = naive_io["total"] / 1024 / 1024
        flash_mb = flash_io["total"] / 1024 / 1024
        reduction = 1 - flash_io["total"] / naive_io["total"]
        n2_ratio = naive_io["n2_term"] / naive_io["total"]

        print(f"{N:>6} | {naive_mb:>10.1f}MB | {flash_mb:>10.1f}MB | "
              f"{reduction:>9.1%} | {n2_ratio:>9.1%}")

        rows.append([
            B, H, N, d,
            naive_io["total"], flash_io["total"],
            f"{naive_mb:.2f}", f"{flash_mb:.2f}",
            f"{reduction:.4f}", f"{n2_ratio:.4f}",
        ])

    print()

    # 写 CSV
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved: {output_csv}")

    return rows


# ============================================================
# 实测带宽估算
# ============================================================
WARMUP = 3
REPEAT = 10


def benchmark_actual_io(B=2, H=4, d=64, seq_lengths=None, device="cuda",
                        output_csv="benchmark/results/measured_bw.csv"):
    """实测运行时间 + 理论 IO 估算等效带宽, 输出 CSV"""
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096]

    print("=" * 80)
    print("FlashAttention Bandwidth Estimation")
    print("=" * 80)
    print(f"Config: B={B} H={H} d={d}")
    print()

    header = ["B", "H", "N", "D", "impl", "time_ms",
              "theoretical_io_mb", "bandwidth_gb_s"]
    rows = []

    print(f"{'N':>6} | {'Impl':>10} | {'Time ms':>10} | "
          f"{'IO MB':>10} | {'BW GB/s':>12}")
    print("-" * 65)

    for N in seq_lengths:
        Q, K, V = gen_qkv(B, H, N, d, device=device)

        # --- Naive ---
        for _ in range(WARMUP):
            _ = naive_attention(Q, K, V)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPEAT):
            _ = naive_attention(Q, K, V)
        end.record()
        torch.cuda.synchronize()
        naive_time_ms = start.elapsed_time(end) / REPEAT
        naive_io = theoretical_io_naive(B, H, N, d)
        naive_bw = naive_io["total"] / (naive_time_ms / 1000) / 1e9
        naive_io_mb = naive_io["total"] / 1024 / 1024

        rows.append([B, H, N, d, "naive", f"{naive_time_ms:.3f}",
                     f"{naive_io_mb:.2f}", f"{naive_bw:.1f}"])

        # --- Flash ---
        for _ in range(WARMUP):
            _ = flash_attn_forward(Q, K, V, causal=False)
        torch.cuda.synchronize()

        start.record()
        for _ in range(REPEAT):
            _ = flash_attn_forward(Q, K, V, causal=False)
        end.record()
        torch.cuda.synchronize()
        flash_time_ms = start.elapsed_time(end) / REPEAT
        flash_io = theoretical_io_flash(B, H, N, d)
        flash_bw = flash_io["total"] / (flash_time_ms / 1000) / 1e9
        flash_io_mb = flash_io["total"] / 1024 / 1024

        rows.append([B, H, N, d, "triton_flash", f"{flash_time_ms:.3f}",
                     f"{flash_io_mb:.2f}", f"{flash_bw:.1f}"])

        print(f"{N:>6} | {'naive':>10} | {naive_time_ms:>8.2f}ms | "
              f"{naive_io_mb:>8.1f}MB | {naive_bw:>10.1f}")
        print(f"{N:>6} | {'flash':>10} | {flash_time_ms:>8.2f}ms | "
              f"{flash_io_mb:>8.1f}MB | {flash_bw:>10.1f}")
        print("-" * 65)

        del Q, K, V
        torch.cuda.empty_cache()

    print()

    # 写 CSV
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved: {output_csv}")

    return rows


# ============================================================
# 图表生成
# ============================================================
def plot_io_scaling(output_dir="benchmark/results", B=2, H=4, d=64,
                    seq_lengths=None):
    """生成 IO vs N 理论对比图"""
    if not HAS_MPL:
        return
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192]

    naive_ios = []
    flash_ios = []
    for N in seq_lengths:
        naive_ios.append(theoretical_io_naive(B, H, N, d)["total"] / 1024 / 1024)
        flash_ios.append(theoretical_io_flash(B, H, N, d)["total"] / 1024 / 1024)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(seq_lengths, naive_ios, "^-", color="#e74c3c",
            label="Naive O(N^2)", linewidth=2, markersize=6)
    ax.plot(seq_lengths, flash_ios, "o-", color="#2ecc71",
            label="Flash O(Nd)", linewidth=2, markersize=6)

    ax.set_xlabel("Sequence Length N", fontsize=12)
    ax.set_ylabel("Theoretical IO (MB)", fontsize=12)
    ax.set_title("IO Complexity: Naive vs FlashAttention", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, "io_vs_N.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_bw_scaling(bw_csv="benchmark/results/measured_bw.csv",
                    output_dir="benchmark/results"):
    """从 measured_bw.csv 生成 BW vs N 图"""
    if not HAS_MPL:
        return
    if not os.path.exists(bw_csv):
        print(f"BW CSV not found: {bw_csv}")
        return

    with open(bw_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fig, ax = plt.subplots(figsize=(8, 5))
    for impl, color, marker in [("naive", "#e74c3c", "^"),
                                 ("triton_flash", "#2ecc71", "o")]:
        data = [r for r in rows if r["impl"] == impl]
        ns = [int(r["N"]) for r in data]
        bws = [float(r["bandwidth_gb_s"]) for r in data]
        if ns:
            ax.plot(ns, bws, f"{marker}-", color=color,
                    label=impl, linewidth=2, markersize=6)

    ax.set_xlabel("Sequence Length N", fontsize=12)
    ax.set_ylabel("Effective Bandwidth (GB/s)", fontsize=12)
    ax.set_title("Bandwidth vs Sequence Length", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, "bw_vs_N.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ============================================================
# NCU 命令模板
# ============================================================
def print_ncu_commands():
    """打印 ncu profiling 命令模板"""
    print("=" * 80)
    print("NCU Profiling Commands")
    print("=" * 80)
    print()
    print("# Collect DRAM metrics:")
    print("ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \\")
    print("    --kernel-name _flash_attn_fwd_kernel \\")
    print('    python -c "')
    print("import torch; import sys; sys.path.insert(0, '.')")
    print("from src.kernel.flash_attn_triton import flash_attn_forward")
    print("from src.utils import gen_qkv")
    print("Q,K,V = gen_qkv(2,4,1024,64)")
    print("flash_attn_forward(Q,K,V)")
    print('"')
    print()
    print("# Compare with theory:")
    print("# Flash IO = 4 * B * H * N * d * 2 bytes")
    print("# ncu measured ~ theory  -> IO-aware verified")
    print("# ncu measured >> theory  -> excess IO detected")
    print()
    print("# Key metrics:")
    print("# dram__throughput     - near HBM peak -> memory bound")
    print("# dram__bytes_write    - Flash << Naive")
    print("# sm__issue_active     - compute utilization")
    print()


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    analyze_io_reduction()
    print()
    plot_io_scaling()
    print()
    benchmark_actual_io()
    print()
    plot_bw_scaling()
    print()
    print_ncu_commands()
