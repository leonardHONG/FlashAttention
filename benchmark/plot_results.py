"""
读取 benchmark CSV 绘制性能图

6 种图表:
  1 Forward latency vs N
  2 Train total latency vs N
  3 Train breakdown: fwd + bwd 分段
  4 tokens/s vs N
  5 Speedup vs SDPA vs N
  6 Peak memory vs N
"""

import sys
import os
import csv
import argparse

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed")


# ============================================================
# 数据加载
# ============================================================
def load_csv(path):
    """加载 CSV 返回行列表"""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def filter_rows(rows, mode=None, causal=None, d=None):
    """按条件过滤"""
    out = []
    for r in rows:
        if mode and r.get("mode", "") != mode:
            continue
        if causal is not None and r.get("causal", "") != str(causal):
            continue
        if d is not None and r.get("D", "") != str(d):
            continue
        out.append(r)
    return out


def group_by_impl(rows):
    """按 impl 分组"""
    groups = {}
    for r in rows:
        impl = r.get("impl", "unknown")
        groups.setdefault(impl, []).append(r)
    return groups


# ============================================================
# 配色
# ============================================================
COLORS = {
    "naive": "#e74c3c",
    "sdpa": "#3498db",
    "triton_flash": "#2ecc71",
}
MARKERS = {
    "naive": "^",
    "sdpa": "s",
    "triton_flash": "o",
}
FWD_COLOR = "#3498db"
BWD_COLOR = "#e67e22"


# ============================================================
# 辅助
# ============================================================
def _get_group_keys(rows):
    """获取 d, causal 分组键"""
    keys = set()
    for r in rows:
        keys.add((r.get("D", "64"), r.get("causal", "False")))
    return sorted(keys)


def _causal_str(causal):
    return "Causal" if causal == "True" else "No Mask"


def _safe_float(val, default=-1.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ============================================================
# 1. Forward latency vs N
# ============================================================
def plot_forward_latency(rows, output_dir):
    if not HAS_MPL:
        return
    fwd_rows = filter_rows(rows, mode="fwd")
    if not fwd_rows:
        return

    for d, causal in _get_group_keys(fwd_rows):
        subset = filter_rows(fwd_rows, d=int(d), causal=(causal == "True"))
        impls = group_by_impl(subset)

        fig, ax = plt.subplots(figsize=(8, 5))
        for impl, data in impls.items():
            pairs = [(int(r["N"]), _safe_float(r["fwd_ms"])) for r in data
                     if _safe_float(r["fwd_ms"]) > 0]
            if pairs:
                pairs.sort()
                ns, ts = zip(*pairs)
                ax.plot(ns, ts, marker=MARKERS.get(impl, "o"),
                        color=COLORS.get(impl), label=impl,
                        linewidth=2, markersize=6)

        ax.set_xlabel("Sequence Length N", fontsize=12)
        ax.set_ylabel("Forward Latency (ms)", fontsize=12)
        ax.set_title(f"Forward Latency (D={d}, {_causal_str(causal)})", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fname = os.path.join(output_dir, f"forward_latency_d{d}_causal{causal}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")


# ============================================================
# 2. Train total latency vs N
# ============================================================
def plot_train_latency(rows, output_dir):
    if not HAS_MPL:
        return
    train_rows = filter_rows(rows, mode="train")
    if not train_rows:
        return

    for d, causal in _get_group_keys(train_rows):
        subset = filter_rows(train_rows, d=int(d), causal=(causal == "True"))
        impls = group_by_impl(subset)

        fig, ax = plt.subplots(figsize=(8, 5))
        for impl, data in impls.items():
            pairs = [(int(r["N"]), _safe_float(r["total_ms"])) for r in data
                     if _safe_float(r["total_ms"]) > 0]
            if pairs:
                pairs.sort()
                ns, ts = zip(*pairs)
                ax.plot(ns, ts, marker=MARKERS.get(impl, "o"),
                        color=COLORS.get(impl), label=impl,
                        linewidth=2, markersize=6)

        ax.set_xlabel("Sequence Length N", fontsize=12)
        ax.set_ylabel("Train Total Latency (ms)", fontsize=12)
        ax.set_title(f"Train Latency (D={d}, {_causal_str(causal)})", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fname = os.path.join(output_dir, f"train_latency_d{d}_causal{causal}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")


# ============================================================
# 3. Train breakdown: fwd + bwd 堆叠柱状图
# ============================================================
def plot_train_breakdown(rows, output_dir):
    if not HAS_MPL:
        return
    train_rows = filter_rows(rows, mode="train")
    if not train_rows:
        return

    for d, causal in _get_group_keys(train_rows):
        subset = filter_rows(train_rows, d=int(d), causal=(causal == "True"))
        # 只画有 fwd_ms 和 bwd_ms 的 impl
        impls_data = {}
        for r in subset:
            fwd = _safe_float(r["fwd_ms"])
            bwd = _safe_float(r["bwd_ms"])
            if fwd > 0 and bwd > 0:
                impl = r["impl"]
                impls_data.setdefault(impl, []).append(r)

        if not impls_data:
            # 回退: 用 total_ms 和曲线图
            fig, ax = plt.subplots(figsize=(8, 5))
            impls = group_by_impl(subset)
            for impl, data in impls.items():
                total = _safe_float(data[0].get("total_ms")) if data else 0
                if total > 0:
                    pairs = [(int(r["N"]), _safe_float(r["total_ms"])) for r in data
                             if _safe_float(r["total_ms"]) > 0]
                    if pairs:
                        pairs.sort()
                        ns, ts = zip(*pairs)
                        ax.plot(ns, ts, marker=MARKERS.get(impl, "o"),
                                color=COLORS.get(impl), label=f"{impl} total",
                                linewidth=2, markersize=6)

            ax.set_xlabel("Sequence Length N", fontsize=12)
            ax.set_ylabel("Latency (ms)", fontsize=12)
            ax.set_title(f"Train Breakdown (D={d}, {_causal_str(causal)})", fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            fname = os.path.join(output_dir,
                                 f"train_breakdown_d{d}_causal{causal}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {fname}")
            continue

        # 堆叠柱状图: 每个 N 位置有多个 impl 的 fwd+bwd 柱子
        all_ns = sorted(set(int(r["N"]) for data in impls_data.values()
                            for r in data))
        impl_names = sorted(impls_data.keys())
        num_impls = len(impl_names)
        x_indices = list(range(len(all_ns)))
        bar_width = 0.8 / max(num_impls, 1)

        fig, ax = plt.subplots(figsize=(10, 5))

        for i, impl in enumerate(impl_names):
            data = impls_data[impl]
            n_map = {}
            for r in data:
                n_map[int(r["N"])] = (
                    _safe_float(r["fwd_ms"]),
                    _safe_float(r["bwd_ms"]),
                )
            fwd_vals = [n_map.get(n, (0, 0))[0] for n in all_ns]
            bwd_vals = [n_map.get(n, (0, 0))[1] for n in all_ns]
            offsets = [x + i * bar_width for x in x_indices]

            ax.bar(offsets, fwd_vals, bar_width, label=f"{impl} fwd",
                   color=FWD_COLOR, alpha=0.6 + 0.2 * i, edgecolor="white")
            ax.bar(offsets, bwd_vals, bar_width, bottom=fwd_vals,
                   label=f"{impl} bwd",
                   color=BWD_COLOR, alpha=0.6 + 0.2 * i, edgecolor="white")

        center_offsets = [x + bar_width * (num_impls - 1) / 2 for x in x_indices]
        ax.set_xticks(center_offsets)
        ax.set_xticklabels([str(n) for n in all_ns])
        ax.set_xlabel("Sequence Length N", fontsize=12)
        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.set_title(f"Train Breakdown: Fwd + Bwd (D={d}, {_causal_str(causal)})",
                     fontsize=14)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")

        fname = os.path.join(output_dir,
                             f"train_breakdown_d{d}_causal{causal}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")


# ============================================================
# 4. tokens/s vs N
# ============================================================
def plot_tokens_per_s(rows, output_dir):
    if not HAS_MPL:
        return

    for mode in ["fwd", "train"]:
        mode_rows = filter_rows(rows, mode=mode)
        if not mode_rows:
            continue

        for d, causal in _get_group_keys(mode_rows):
            subset = filter_rows(mode_rows, d=int(d), causal=(causal == "True"))
            impls = group_by_impl(subset)

            fig, ax = plt.subplots(figsize=(8, 5))
            for impl, data in impls.items():
                pairs = [(int(r["N"]), _safe_float(r["tokens_per_s"])) for r in data
                         if _safe_float(r["tokens_per_s"]) > 0]
                if pairs:
                    pairs.sort()
                    ns, tps = zip(*pairs)
                    ax.plot(ns, tps, marker=MARKERS.get(impl, "o"),
                            color=COLORS.get(impl), label=impl,
                            linewidth=2, markersize=6)

            mode_str = "Forward" if mode == "fwd" else "Train"
            ax.set_xlabel("Sequence Length N", fontsize=12)
            ax.set_ylabel("Tokens/s", fontsize=12)
            ax.set_title(f"{mode_str} Throughput (D={d}, {_causal_str(causal)})",
                         fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

            fname = os.path.join(output_dir,
                                 f"tokens_per_s_{mode}_d{d}_causal{causal}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {fname}")


# ============================================================
# 5. Speedup vs SDPA vs N
# ============================================================
def plot_speedup(rows, output_dir):
    if not HAS_MPL:
        return

    for mode in ["fwd", "train"]:
        mode_rows = filter_rows(rows, mode=mode)
        if not mode_rows:
            continue

        for d, causal in _get_group_keys(mode_rows):
            subset = filter_rows(mode_rows, d=int(d), causal=(causal == "True"))
            # 只画非 sdpa 的 impl
            for impl_name in ["triton_flash", "naive"]:
                data = [r for r in subset if r["impl"] == impl_name]
                pairs = [(int(r["N"]), _safe_float(r["speedup_vs_sdpa"]))
                         for r in data if _safe_float(r["speedup_vs_sdpa"]) > 0]
                if not pairs:
                    continue

                fig, ax = plt.subplots(figsize=(8, 5))
                pairs.sort()
                ns, sps = zip(*pairs)
                ax.plot(ns, sps, "o-", color=COLORS.get(impl_name, "#333"),
                        label=f"{impl_name} vs SDPA",
                        linewidth=2, markersize=6)
                ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5,
                           label="1x baseline")

                mode_str = "Forward" if mode == "fwd" else "Train"
                ax.set_xlabel("Sequence Length N", fontsize=12)
                ax.set_ylabel("Speedup vs SDPA", fontsize=12)
                ax.set_title(
                    f"{mode_str} Speedup: {impl_name} (D={d}, {_causal_str(causal)})",
                    fontsize=14)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)

                fname = os.path.join(
                    output_dir,
                    f"speedup_{mode}_{impl_name}_d{d}_causal{causal}.png")
                fig.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {fname}")


# ============================================================
# 6. Peak memory vs N
# ============================================================
def plot_peak_memory(rows, output_dir):
    if not HAS_MPL:
        return

    for mode in ["fwd", "train"]:
        mode_rows = filter_rows(rows, mode=mode)
        if not mode_rows:
            continue

        for d, causal in _get_group_keys(mode_rows):
            subset = filter_rows(mode_rows, d=int(d), causal=(causal == "True"))
            impls = group_by_impl(subset)

            fig, ax = plt.subplots(figsize=(8, 5))
            for impl, data in impls.items():
                pairs = [(int(r["N"]), _safe_float(r["peak_mem_mb"])) for r in data
                         if _safe_float(r["peak_mem_mb"]) > 0]
                if pairs:
                    pairs.sort()
                    ns, mems = zip(*pairs)
                    ax.plot(ns, mems, marker=MARKERS.get(impl, "s"),
                            color=COLORS.get(impl), label=impl,
                            linewidth=2, markersize=6)

            mode_str = "Forward" if mode == "fwd" else "Train"
            ax.set_xlabel("Sequence Length N", fontsize=12)
            ax.set_ylabel("Peak Memory (MB)", fontsize=12)
            ax.set_title(f"{mode_str} Memory (D={d}, {_causal_str(causal)})",
                         fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            fname = os.path.join(output_dir,
                                 f"memory_{mode}_d{d}_causal{causal}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {fname}")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--input", default="benchmark/results/results.csv")
    parser.add_argument("--output-dir", default="benchmark/results/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_csv(args.input)

    plot_forward_latency(rows, args.output_dir)
    plot_train_latency(rows, args.output_dir)
    plot_train_breakdown(rows, args.output_dir)
    plot_tokens_per_s(rows, args.output_dir)
    plot_speedup(rows, args.output_dir)
    plot_peak_memory(rows, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")
