"""
性能基准测试

支持:
  - forward-only 模式
  - train 模式: 分别计时 fwd / bwd, 用 CUDA events + sync
  - tokens/s / speedup_vs_sdpa / skip_ratio / peak_mem
  - GPU SM / CUDA driver / torch / triton 版本记录
  - CSV schema
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import csv
import math
import argparse
from src.functional import flash_attention
from src.kernel.flash_attn_triton import flash_attn_forward, flash_attn_forward_debug
from src.reference import naive_attention, sdpa_attention
from src.utils import gen_qkv


# ============================================================
# 环境信息
# ============================================================
def get_env_info():
    """采集 GPU / CUDA / PyTorch / Triton 版本"""
    gpu_name = "unknown"
    gpu_sm = "unknown"
    cuda_driver = "unknown"
    try:
        gpu_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        gpu_sm = f"sm_{cap[0]}{cap[1]}"
        cuda_driver = torch.version.cuda or "unknown"
    except Exception:
        pass

    torch_ver = torch.__version__

    triton_ver = "unknown"
    try:
        import triton
        triton_ver = triton.__version__
    except Exception:
        pass

    return {
        "gpu_name": gpu_name,
        "gpu_sm": gpu_sm,
        "cuda_driver": cuda_driver,
        "torch_version": torch_ver,
        "triton_version": triton_ver,
    }


# ============================================================
# 核心计时
# ============================================================
def measure_fwd(attn_fn, Q, K, V, warmup, repeat, **kwargs):
    """
    Forward-only 计时 + 峰值显存

    返回: fwd_ms, peak_mem_mb
    """
    # warmup
    for _ in range(warmup):
        attn_fn(Q, K, V, **kwargs)
    torch.cuda.synchronize()

    # peak memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        attn_fn(Q, K, V, **kwargs)
    end.record()
    torch.cuda.synchronize()
    fwd_ms = start.elapsed_time(end) / repeat

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    return fwd_ms, peak_mem_mb


def measure_train(attn_fn, Q_base, K_base, V_base, warmup, repeat, **kwargs):
    """
    Train 计时: 分别测 fwd 和 bwd, 用 CUDA events + sync

    返回: fwd_ms, bwd_ms, total_ms, peak_mem_mb
    """
    # warmup
    for _ in range(warmup):
        q = Q_base.clone().detach().requires_grad_(True)
        k = K_base.clone().detach().requires_grad_(True)
        v = V_base.clone().detach().requires_grad_(True)
        o = attn_fn(q, k, v, **kwargs)
        loss = o.sum()
        loss.backward()
        q.grad = None; k.grad = None; v.grad = None
    torch.cuda.synchronize()

    # 正式计时
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    fwd_total = 0.0
    bwd_total = 0.0

    for _ in range(repeat):
        q = Q_base.clone().detach().requires_grad_(True)
        k = K_base.clone().detach().requires_grad_(True)
        v = V_base.clone().detach().requires_grad_(True)

        # fwd 计时
        s_fwd = torch.cuda.Event(enable_timing=True)
        e_fwd = torch.cuda.Event(enable_timing=True)
        s_fwd.record()
        o = attn_fn(q, k, v, **kwargs)
        loss = o.sum()
        e_fwd.record()
        torch.cuda.synchronize()
        fwd_total += s_fwd.elapsed_time(e_fwd)

        # bwd 计时
        s_bwd = torch.cuda.Event(enable_timing=True)
        e_bwd = torch.cuda.Event(enable_timing=True)
        s_bwd.record()
        loss.backward()
        e_bwd.record()
        torch.cuda.synchronize()
        bwd_total += s_bwd.elapsed_time(e_bwd)

        # 清梯度防累积
        q.grad = None; k.grad = None; v.grad = None

    fwd_ms = fwd_total / repeat
    bwd_ms = bwd_total / repeat
    total_ms = fwd_ms + bwd_ms
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    return fwd_ms, bwd_ms, total_ms, peak_mem_mb


def get_skip_ratio(Q, K, V, causal=True):
    """获取 early-exit 跳过比例"""
    try:
        _, _, skip_counts = flash_attn_forward_debug(Q, K, V, causal=causal)
        B, H, N, d = Q.shape
        BLOCK_M = 64
        BLOCK_N = 64
        num_q_blocks = math.ceil(N / BLOCK_M)
        total_k_blocks = math.ceil(N / BLOCK_N)
        max_blocks = B * H * num_q_blocks * total_k_blocks
        skipped = skip_counts.sum().item()
        return skipped / max_blocks if max_blocks > 0 else 0.0
    except Exception:
        return -1.0


# ============================================================
# impl 分发
# ============================================================
def _fwd_naive(Q, K, V, **kw):
    return naive_attention(Q, K, V)

def _fwd_sdpa(Q, K, V, causal=False, **kw):
    return sdpa_attention(Q, K, V, is_causal=causal)

def _fwd_triton(Q, K, V, causal=False, seqlens_k=None, **kw):
    return flash_attn_forward(Q, K, V, causal=causal, seqlens_k=seqlens_k)

def _train_naive(Q, K, V, **kw):
    return naive_attention(Q, K, V)

def _train_sdpa(Q, K, V, causal=False, **kw):
    return sdpa_attention(Q, K, V, is_causal=causal)

def _train_triton(Q, K, V, causal=False, seqlens_k=None, **kw):
    return flash_attention(Q, K, V, causal=causal, seqlens_k=seqlens_k)

FWD_IMPLS = {
    "naive": _fwd_naive,
    "sdpa": _fwd_sdpa,
    "triton_flash": _fwd_triton,
}
TRAIN_IMPLS = {
    "naive": _train_naive,
    "sdpa": _train_sdpa,
    "triton_flash": _train_triton,
}


# ============================================================
# 主 benchmark
# ============================================================
CSV_HEADER = [
    "gpu_name", "gpu_sm", "cuda_driver",
    "torch_version", "triton_version",
    "dtype", "mode",
    "impl", "B", "H", "N", "D",
    "causal", "seqlen_k",
    "fwd_ms", "bwd_ms", "total_ms",
    "tokens_per_s", "peak_mem_mb",
    "speedup_vs_sdpa", "skip_ratio",
]


def run_benchmark(args):
    """运行完整 benchmark"""
    env = get_env_info()

    seq_lengths = [int(x) for x in args.N.split(",")]
    head_dims = [int(x) for x in args.d.split(",")]
    impls = [x.strip() for x in args.impl.split(",")]
    causal_list = {"0": [False], "1": [True], "both": [False, True]}[args.causal]
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    print("=" * 110)
    print(f"FlashAttention Benchmark  mode={args.mode}  dtype={args.dtype}")
    print("=" * 110)
    print(f"GPU:     {env['gpu_name']}  {env['gpu_sm']}")
    print(f"CUDA:    {env['cuda_driver']}")
    print(f"PyTorch: {env['torch_version']}")
    print(f"Triton:  {env['triton_version']}")
    print(f"warmup={args.warmup}  repeat={args.repeat}")
    print()

    print(f"{'N':>6} | {'D':>4} | {'caus':>5} | {'impl':>12} | "
          f"{'fwd_ms':>8} | {'bwd_ms':>8} | {'total':>8} | "
          f"{'tok/s':>12} | {'mem_MB':>8} | {'speedup':>8} | {'skip':>6}")
    print("-" * 115)

    results = []

    for N in seq_lengths:
        for d in head_dims:
            for causal in causal_list:
                try:
                    Q, K, V = gen_qkv(args.B, args.H, N, d,
                                      device="cuda", dtype=dtype)
                except torch.cuda.OutOfMemoryError:
                    print(f"{N:>6} | {d:>4} | {str(causal):>5} | OOM")
                    continue

                # seqlen_k: -1 表示无 padding
                seqlen_k_val = -1

                # 收集 sdpa total_ms 用于 speedup
                sdpa_total_ms = -1.0
                config_rows = {}

                for impl in impls:
                    # naive 大 N 跳过
                    if impl == "naive" and N > 4096:
                        continue

                    fwd_ms = 0.0
                    bwd_ms = 0.0
                    total_ms = 0.0
                    peak_mem_mb = 0.0
                    skip_ratio = -1.0

                    fn_kwargs = {"causal": causal}

                    try:
                        if args.mode == "fwd":
                            fn = FWD_IMPLS[impl]
                            fwd_ms, peak_mem_mb = measure_fwd(
                                fn, Q, K, V,
                                warmup=args.warmup, repeat=args.repeat,
                                **fn_kwargs)
                            total_ms = fwd_ms

                        elif args.mode == "train":
                            fn = TRAIN_IMPLS[impl]
                            fwd_ms, bwd_ms, total_ms, peak_mem_mb = measure_train(
                                fn, Q, K, V,
                                warmup=args.warmup, repeat=args.repeat,
                                **fn_kwargs)

                        # skip_ratio
                        if impl == "triton_flash" and causal:
                            skip_ratio = get_skip_ratio(Q, K, V, causal=True)

                    except (torch.cuda.OutOfMemoryError, Exception) as e:
                        print(f"  {impl} error: {e}")
                        torch.cuda.empty_cache()
                        continue

                    # tokens/s
                    tokens = args.B * args.H * N
                    total_s = total_ms / 1000.0
                    tokens_per_s = tokens / total_s if total_s > 0 else -1.0

                    if impl == "sdpa":
                        sdpa_total_ms = total_ms

                    config_rows[impl] = {
                        "fwd_ms": fwd_ms,
                        "bwd_ms": bwd_ms,
                        "total_ms": total_ms,
                        "tokens_per_s": tokens_per_s,
                        "peak_mem_mb": peak_mem_mb,
                        "skip_ratio": skip_ratio,
                    }

                # 写出并打印
                for impl, data in config_rows.items():
                    speedup = sdpa_total_ms / data["total_ms"] \
                        if sdpa_total_ms > 0 and data["total_ms"] > 0 \
                        else -1.0

                    row = [
                        env["gpu_name"], env["gpu_sm"], env["cuda_driver"],
                        env["torch_version"], env["triton_version"],
                        args.dtype, args.mode,
                        impl, args.B, args.H, N, d,
                        causal, seqlen_k_val,
                        f"{data['fwd_ms']:.3f}",
                        f"{data['bwd_ms']:.3f}",
                        f"{data['total_ms']:.3f}",
                        f"{data['tokens_per_s']:.0f}",
                        f"{data['peak_mem_mb']:.1f}",
                        f"{speedup:.3f}",
                        f"{data['skip_ratio']:.4f}",
                    ]
                    results.append(row)

                    print(
                        f"{N:>6} | {d:>4} | {str(causal):>5} | {impl:>12} | "
                        f"{data['fwd_ms']:>8.2f} | {data['bwd_ms']:>8.2f} | "
                        f"{data['total_ms']:>8.2f} | "
                        f"{data['tokens_per_s']:>12.0f} | "
                        f"{data['peak_mem_mb']:>8.1f} | "
                        f"{speedup:>8.3f} | "
                        f"{data['skip_ratio']:>6.3f}"
                    )

                del Q, K, V
                torch.cuda.empty_cache()

    # 写 CSV
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(results)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttention Benchmark")
    parser.add_argument("--mode", choices=["fwd", "train"], default="fwd",
                        help="fwd: forward-only / train: forward+backward")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--causal", choices=["0", "1", "both"], default="both",
                        help="0: no causal / 1: causal / both")
    parser.add_argument("--d", default="64,128", help="head dims, comma separated")
    parser.add_argument("--N", default="512,1024,2048,4096,8192",
                        help="sequence lengths, comma separated")
    parser.add_argument("--impl", default="naive,sdpa,triton_flash",
                        help="implementations, comma separated")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", default="benchmark/results/results.csv")
    args = parser.parse_args()
    run_benchmark(args)
