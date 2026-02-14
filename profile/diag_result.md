============================================================
1) Causal Skip Ratio
============================================================
 B  H     N   d |  causal_skip no_mask_skip | max_diff
----------------------------------------------------------------------
 2  4   256  64 |       37.5%         0.0% | 0.000244
 2  4   512  64 |       43.8%         0.0% | 0.000488
 2  4  1024  64 |       46.9%         0.0% | 0.000244
 2  4  1024 128 |       46.9%         0.0% | 0.000488
 1  8  2048  64 |       48.4%         0.0% | 0.000244

============================================================
2) Autotune 选中的 Config
============================================================
  [cache] (4 entries):
    (1024, 128, 'torch.float16', 'torch.float16', 'torch.float16', 'torch.float16', 'torch.int32') -> BLOCK_M: 128, BLOCK_N: 128, num_warps: 8, num_ctas: 1, num_stages: 2, maxnreg: None
    (2048, 64, 'torch.float16', 'torch.float16', 'torch.float16', 'torch.float16', 'torch.int32') -> BLOCK_M: 128, BLOCK_N: 64, num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None
    (4096, 64, 'torch.float16', 'torch.float16', 'torch.float16', 'torch.float16', 'torch.int32') -> BLOCK_M: 128, BLOCK_N: 64, num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None
    (512, 64, 'torch.float16', 'torch.float16', 'torch.float16', 'torch.float16', 'torch.int32') -> BLOCK_M: 64, BLOCK_N: 64, num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None

============================================================
3) NCU Profiling 命令
============================================================

=== causal=False ===
ncu --metrics dram__bytes_read.sum,smsp__warp_issue_stalled_long_scoreboard_pct,sm__warps_active.avg.pct_of_peak_sustained_active ^
    --kernel-name flash --launch-count 1 ^
    conda run -n a python profile/ncu_run.py --N 1024 --d 128 --causal False

=== causal=True ===
ncu --metrics dram__bytes_read.sum,smsp__warp_issue_stalled_long_scoreboard_pct,sm__warps_active.avg.pct_of_peak_sustained_active ^
    --kernel-name flash --launch-count 1 ^
    conda run -n a python profile/ncu_run.py --N 1024 --d 128 --causal True