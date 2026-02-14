# FlashAttention 前向传播 — 从零实现

> **当前实现: Forward only**。Backward 作为后续扩展方向。

从零复现 FlashAttention 前向传播的 Triton 实现，涵盖完整的工程验证体系。

## 特性

- **IO-aware 计算重排** — 不落盘 N×N 中间矩阵，IO 从 O(N²) 降至 O(Nd)
- **Online Softmax** — 流式更新 m/l/acc，数值稳定
- **Causal Mask** — 支持 block-level early exit
- **Padding Mask** — 变长序列，越界安全读取
- **Block Early Exit** — causal/padding 场景下跳过完全无效 tile
- **Autotune** — 自动搜索最优 BLOCK_M/N、num_warps、num_stages
- **Debug 模式** — 统计被跳过的 block 数量，验证 early exit 效果
- **IO 复杂度验证** — 理论分析 + ncu 实测对比

## 项目结构

```
flashattn-from-scratch/
├── src/
│   ├── kernel/
│   │   └── flash_attn_triton.py  # Triton kernel（核心）
│   ├── functional.py             # 上层 API
│   ├── reference.py              # 朴素/SDPA 参考实现
│   └── utils.py                  # 工具函数
├── tests/
│   ├── test_correctness.py       # 正确性 + FP32 精度 + Early Exit 统计
│   └── test_masking.py           # Mask 专项测试
├── benchmark/
│   ├── run_benchmark.py          # 性能基准（含 SDPA 后端信息）
│   ├── plot_results.py           # 结果可视化
│   └── analyze_io.py             # IO 复杂度验证
├── profile/
│   └── ncu_notes.md              # Profiling 笔记
└── README.md
```

## 快速开始

### 环境要求

```
Python >= 3.8
PyTorch >= 2.0 (CUDA)
Triton >= 2.1
pytest
matplotlib
```

### 运行测试

```bash
cd flashattn-from-scratch
python -m pytest tests/ -v -s
```

### 运行 Benchmark

```bash
python benchmark/run_benchmark.py
python benchmark/plot_results.py
```

### IO 复杂度分析

```bash
python benchmark/analyze_io.py
```

## 使用示例

```python
import torch
from src.functional import flash_attention

B, H, N, d = 2, 8, 2048, 64
Q = torch.randn(B, H, N, d, dtype=torch.float16, device="cuda")
K = torch.randn(B, H, N, d, dtype=torch.float16, device="cuda")
V = torch.randn(B, H, N, d, dtype=torch.float16, device="cuda")

# 基础
O = flash_attention(Q, K, V)

# Causal
O = flash_attention(Q, K, V, causal=True)

# Padding（变长序列）
seqlens = torch.tensor([1500, 2048], dtype=torch.int32, device="cuda")
O = flash_attention(Q, K, V, seqlens_k=seqlens)

# Debug: 查看 early exit 统计
from src.kernel.flash_attn_triton import flash_attn_forward_debug
O, skip_counts = flash_attn_forward_debug(Q, K, V, causal=True)
print(f"跳过 block 数: {skip_counts.sum().item()}")
```

## 数值稳定性策略

| 风险 | 处理 |
|------|------|
| `-inf - (-inf) → NaN` | `alpha = where(m == -inf, 0, exp(m - m_new))` |
| 全 mask 行 | `l == 0` 时输出置零 |
| exp 溢出 | 始终减去当前最大值 `m_new` |

## 迁移指南（4070 → A100）

| 项目 | 4070 | A100 |
|------|------|------|
| BLOCK_M | 64 | 128 |
| BLOCK_N | 64 | 128 |
| num_warps | 4 | 8 |
| num_stages | 3 | 4 |

使用 autotune 自动选择，无需手动调参。

## 后续扩展

- [ ] Backward 实现
- [ ] Mixed precision 优化
- [ ] KV-cache streaming attention
- [ ] 多卡 Attention pipeline
