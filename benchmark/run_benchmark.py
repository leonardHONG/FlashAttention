"""
性能基准测试：对比 naive / SDPA / Triton FlashAttention。

测试维度：
  N {512, 1024, 2048, 4096, 8192}
  d {64, 128}
  causal{True, False}
指标：Latency (ms)、显存峰值 (MB)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import csv
import argparse
from src.kernel.flash_attn_triton import flash_attn_forward
from src.reference import naive_attention, sdpa_attention
from src.utils import gen_qkv


WARMUP = 5
REPEAT = 20


def print_sdpa_backend_info():
    """打印 SDPA 后端配置，确保对比公平"""
    print("=" * 60)
    print("PyTorch SDPA 后端配置")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本:    {torch.version.cuda}")
    try:
        print(f"Flash SDP:        {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"Math SDP:         {torch.backends.cuda.math_sdp_enabled()}")
        print(f"Mem-efficient SDP:{torch.backends.cuda.mem_efficient_sdp_enabled()}")
    except AttributeError:
        print("（当前 PyTorch 版本不支持查询 SDP 后端状态）")
    print("=" * 60)
    print()


def measure_latency(fn, *args, **kwargs):
    """用 cudaEvent 测量函数延迟 (ms)"""
    # warmup
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEAT):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPEAT


def measure_memory(fn, *args, **kwargs):
    """测量峰值显存 (MB)"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    return peak


def run_benchmark(output_csv="benchmark/results.csv",
                  seq_lengths=None, head_dims=None,
                  B=2, H=4, device="cuda"):
    """运行完整 benchmark，结果写入 CSV"""
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192]
    if head_dims is None:
        head_dims = [64, 128]

    print_sdpa_backend_info()

    results = []
    header = ["N", "d", "causal", "方法", "延迟(ms)", "显存峰值(MB)"]

    print(f"{'N':>6} | {'d':>4} | {'causal':>6} | {'方法':>10} | {'延迟(ms)':>10} | {'显存(MB)':>10}")
    print("-" * 65)

    for N in seq_lengths:
        for d in head_dims:
            for causal in [False, True]:
                try:
                    Q, K, V = gen_qkv(B, H, N, d, device=device)
                except torch.cuda.OutOfMemoryError:
                    print(f"{N:>6} | {d:>4} | {str(causal):>6} | {'OOM':>10}")
                    continue

                configs = []

                # --- Naive ---
                if N <= 4096:  # 大 N 时 naive 会 OOM
                    try:
                        t = measure_latency(naive_attention, Q, K, V,
                                            mask=None if not causal else None)
                        m = measure_memory(naive_attention, Q, K, V)
                        configs.append(("naive", t, m))
                    except torch.cuda.OutOfMemoryError:
                        configs.append(("naive", -1, -1))

                # --- SDPA ---
                try:
                    t = measure_latency(sdpa_attention, Q, K, V,
                                        is_causal=causal)
                    m = measure_memory(sdpa_attention, Q, K, V,
                                       is_causal=causal)
                    configs.append(("sdpa", t, m))
                except Exception as e:
                    configs.append(("sdpa", -1, -1))

                # --- Triton FlashAttention ---
                try:
                    t = measure_latency(flash_attn_forward, Q, K, V,
                                        causal=causal)
                    m = measure_memory(flash_attn_forward, Q, K, V,
                                       causal=causal)
                    configs.append(("triton_flash", t, m))
                except Exception as e:
                    configs.append(("triton_flash", -1, -1))

                for name, t, m in configs:
                    print(f"{N:>6} | {d:>4} | {str(causal):>6} | {name:>10} | "
                          f"{t:>8.2f}ms | {m:>8.1f}MB")
                    results.append([N, d, causal, name, f"{t:.2f}", f"{m:.1f}"])

                # 释放显存
                del Q, K, V
                torch.cuda.empty_cache()

    # 写 CSV
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

    print(f"\n结果已保存至: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttention Benchmark")
    parser.add_argument("--output", default="benchmark/results.csv")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--H", type=int, default=4)
    args = parser.parse_args()
    run_benchmark(output_csv=args.output, B=args.B, H=args.H)
