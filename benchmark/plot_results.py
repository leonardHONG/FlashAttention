"""读取 benchmark CSV 结果，绘制 latency / memory 对比图"""

import sys
import os
import csv
import argparse

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 尝试使用系统中文字体
    import matplotlib.font_manager as fm
    _zh_font = None
    for name in ["Microsoft YaHei", "SimHei", "SimSun", "WenQuanYi Micro Hei"]:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            _zh_font = name
            break
    if _zh_font:
        plt.rcParams["font.sans-serif"] = [_zh_font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("警告: 未安装 matplotlib，无法绘图")


def load_csv(path):
    """加载 CSV，返回行列表"""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_latency(rows, output_dir="benchmark"):
    """按 (d, causal) 分组，绘制 latency vs N 对比图"""
    if not HAS_MPL:
        return

    groups = {}
    for r in rows:
        key = (r["d"], r["causal"])
        groups.setdefault(key, []).append(r)

    for (d, causal), data in groups.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        methods = {}
        for r in data:
            name = r["方法"]
            methods.setdefault(name, {"N": [], "t": []})
            t = float(r["延迟(ms)"])
            if t > 0:
                methods[name]["N"].append(int(r["N"]))
                methods[name]["t"].append(t)

        colors = {"naive": "#e74c3c", "sdpa": "#3498db", "triton_flash": "#2ecc71"}
        for name, vals in methods.items():
            ax.plot(vals["N"], vals["t"], "o-", label=name,
                    color=colors.get(name, None), linewidth=2, markersize=6)

        ax.set_xlabel("序列长度 N", fontsize=12)
        ax.set_ylabel("延迟 (ms)", fontsize=12)
        causal_str = "Causal" if causal == "True" else "No Mask"
        ax.set_title(f"Latency (d={d}, {causal_str})", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fname = f"{output_dir}/latency_d{d}_causal{causal}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存: {fname}")


def plot_memory(rows, output_dir="benchmark"):
    """按 (d, causal) 分组，绘制 memory vs N 对比图。"""
    if not HAS_MPL:
        return

    groups = {}
    for r in rows:
        key = (r["d"], r["causal"])
        groups.setdefault(key, []).append(r)

    for (d, causal), data in groups.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        methods = {}
        for r in data:
            name = r["方法"]
            methods.setdefault(name, {"N": [], "m": []})
            m = float(r["显存峰值(MB)"])
            if m > 0:
                methods[name]["N"].append(int(r["N"]))
                methods[name]["m"].append(m)

        colors = {"naive": "#e74c3c", "sdpa": "#3498db", "triton_flash": "#2ecc71"}
        for name, vals in methods.items():
            ax.plot(vals["N"], vals["m"], "s-", label=name,
                    color=colors.get(name, None), linewidth=2, markersize=6)

        ax.set_xlabel("序列长度 N", fontsize=12)
        ax.set_ylabel("显存峰值 (MB)", fontsize=12)
        causal_str = "Causal" if causal == "True" else "No Mask"
        ax.set_title(f"Peak Memory (d={d}, {causal_str})", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        fname = f"{output_dir}/memory_d{d}_causal{causal}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制 benchmark 结果")
    parser.add_argument("--input", default="benchmark/results.csv")
    parser.add_argument("--output-dir", default="benchmark")
    args = parser.parse_args()

    rows = load_csv(args.input)
    plot_latency(rows, args.output_dir)
    plot_memory(rows, args.output_dir)
