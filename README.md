# FlashAttention — 从零实现 Forward + Backward

> **当前实现: Forward + Backward**, 支持训练级梯度传播

从零复现 FlashAttention 的 Triton 实现, 涵盖完整的工程验证体系

## 特性

- **IO-aware 计算重排** — 不落盘 N*N 中间矩阵, IO 从 O(N^2) 降至 O(Nd)
- **Online Softmax** — 流式更新 m/l/acc, 数值稳定
- **Causal Mask** — 支持 block-level early exit
- **Padding Mask** — 变长序列, 越界安全读取
- **Block Early Exit** — causal/padding 场景下跳过完全无效 tile
- **Autotune** — 自动搜索最优 BLOCK_M/N, num_warps, num_stages
- **Debug 模式** — 统计被跳过的 block 数量, 验证 early exit 效果
- **IO 复杂度验证** — 理论分析 + ncu 实测对比
- **完整 Backward** — 2-kernel 架构, 不依赖 atomic, IO-aware 反向传播
- **训练级 API** — torch.autograd.Function 封装, drop-in 替换 SDPA


## 项目结构

```
flashattn-from-scratch/
├── src/
│   ├── kernel/
│   │   ├── flash_attn_triton.py      # 前向 Triton kernel
│   │   └── flash_attn_bwd_triton.py  # 反向 Triton kernel
│   ├── functional.py                  # 上层 API + Autograd
│   ├── reference.py                   # 朴素/SDPA 参考实现
│   └── utils.py                       # 工具函数
├── tests/
│   ├── test_correctness.py            # 前向正确性 + FP32 精度 + Early Exit
│   ├── test_masking.py                # Mask 专项测试
│   └── test_backward.py              # 反向传播正确性 + gradcheck
├── benchmark/
│   ├── run_benchmark.py               # 性能基准 fwd/train 模式
│   ├── plot_results.py                # 5 种图表生成
│   ├── analyze_io.py                  # IO 复杂度验证 + 带宽分析
│   └── results/                       # CSV 数据 + 图表输出
├── profile/
│   └── ncu_notes.md                   # Profiling 笔记
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

# 前向测试
python -m pytest tests/test_correctness.py tests/test_masking.py -v -s

# 反向测试
python -m pytest tests/test_backward.py -v -s

# 全部
python -m pytest tests/ -v -s
```

### 运行 Benchmark

```bash
# Forward-only
python benchmark/run_benchmark.py --mode fwd

# Forward + Backward 训练模式
python benchmark/run_benchmark.py --mode train

# 常用参数
python benchmark/run_benchmark.py --mode train \
    --dtype fp16 --causal both \
    --d 64,128 --N 512,1024,2048,4096,8192 \
    --impl sdpa,triton_flash \
    --warmup 10 --repeat 20

# 生成 6 种图表
python benchmark/plot_results.py
```

输出:
- `benchmark/results/results.csv`
- `benchmark/results/plots/forward_latency_*.png`
- `benchmark/results/plots/train_latency_*.png`
- `benchmark/results/plots/train_breakdown_*.png`
- `benchmark/results/plots/tokens_per_s_*.png`
- `benchmark/results/plots/speedup_*.png`
- `benchmark/results/plots/memory_*.png`

### IO 复杂度分析

```bash
python benchmark/analyze_io.py
```

输出:
- `benchmark/results/theoretical_io.csv` — 理论 IO 数据
- `benchmark/results/measured_bw.csv` — 实测带宽数据
- `benchmark/results/io_vs_N.png` — IO scaling 曲线
- `benchmark/results/bw_vs_N.png` — 带宽 scaling 曲线

## 使用示例

### 推理

```python
import torch
from src.functional import flash_attention

B, H, N, d = 2, 8, 2048, 64
Q = torch.randn(B, H, N, d, dtype=torch.float16, device="cuda")
K = torch.randn(B, H, N, d, dtype=torch.float16, device="cuda")
V = torch.randn(B, H, N, d, dtype=torch.float16, device="cuda")

O = flash_attention(Q, K, V)
O = flash_attention(Q, K, V, causal=True)

seqlens = torch.tensor([1500, 2048], dtype=torch.int32, device="cuda")
O = flash_attention(Q, K, V, seqlens_k=seqlens)
```

### 训练

```python
import torch
from src.functional import flash_attention

B, H, N, d = 2, 8, 512, 64
Q = torch.randn(B, H, N, d, dtype=torch.float16, device="cuda", requires_grad=True)
K = torch.randn(B, H, N, d, dtype=torch.float16, device="cuda", requires_grad=True)
V = torch.randn(B, H, N, d, dtype=torch.float16, device="cuda", requires_grad=True)

O = flash_attention(Q, K, V, causal=True)
loss = O.sum()
loss.backward()
# Q.grad, K.grad, V.grad 已计算
```

### Debug: Early Exit 统计

```python
from src.functional import flash_attention_debug
O, skip_counts = flash_attention_debug(Q, K, V, causal=True)
```

## Benchmark CSV Schema

```
gpu_name, gpu_sm, cuda_driver,
torch_version, triton_version,
dtype, mode,
impl, B, H, N, D,
causal, seqlen_k,
fwd_ms, bwd_ms, total_ms,
tokens_per_s, peak_mem_mb,
speedup_vs_sdpa, skip_ratio
```

计时规范:
- fwd_ms: forward kernel GPU 时间
- bwd_ms: backward GPU 时间
- total_ms = fwd_ms + bwd_ms
- 全程 CUDA events + synchronize
- speedup_vs_sdpa 基于 total_ms 计算

6 种图表:

| 图表 | 文件 |
|------|------|
| Forward latency vs N | forward_latency_*.png |
| Train total latency vs N | train_latency_*.png |
| Train breakdown fwd+bwd | train_breakdown_*.png |
| tokens/s vs N | tokens_per_s_*.png |
| Speedup vs SDPA | speedup_*.png |
| Peak memory vs N | memory_*.png |

## Backward 设计

采用 2-kernel 架构避免 atomic 冲突:

| Kernel | 并行轴 | 计算内容 | 遍历方向 |
|--------|--------|---------|---------|
| Kernel A | k_block | dK + dV | 遍历所有 q_block |
| Kernel B | q_block | dQ | 遍历所有 k_block |

核心: Forward 保存 logsumexp L, Backward 用 P = exp(S - L) 重建注意力权重

## 数值稳定性策略

| 风险 | 处理 |
|------|------|
| -inf - (-inf) -> NaN | alpha = where(m == -inf, 0, exp(m - m_new)) |
| 全 mask 行 | l == 0 时输出/梯度置零 |
| exp 溢出 | 始终减去 L 或当前最大值 |
| fp16 累加误差 | 所有累加使用 fp32 |

## 性能预期

```
Forward speedup: 1.6x - 2.3x vs SDPA
Train speedup:   1.4x - 1.9x vs SDPA
Peak memory:     40% - 70% reduction
IO reduction:    O(N^2) -> O(Nd)
```

实测带宽接近 GPU HBM 峰值 -> 计算 IO-bound 验证通过
实测 DRAM bytes 接近理论值 -> IO-aware 策略验证通过

## 后续扩展

- [x] Backward 实现
- [x] 训练级 API
- [ ] Mixed precision 优化
- [ ] KV-cache streaming attention
- [ ] 多卡 Attention pipeline
