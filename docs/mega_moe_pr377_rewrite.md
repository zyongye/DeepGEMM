# PR #377 MXFP4 MegaMoE rewrite

This branch layers native MXFP4 activations and optional FP8 combine transport
onto the dynamic MegaMoE scheduler released in DeepGEMM PR #377.

## What changed

- Routed L1 and L2 use packed E2M1 activations and weights with block-32
  UE8M0 scales.  The default path issues native `kind::mxf4` SM100 MMA; set
  `DG_MEGA_MXF4_KIND=0` to retain the unpacked `kind::mxf8f6f4` fallback.
- The SwiGLU epilogue quantizes directly into the packed L2-input ring.
- Dispatch copies the physical MXFP4 token size (`hidden / 2` bytes); using
  the logical hidden width would overrun adjacent ring slots.
- `combine_dtype=torch.float8_e4m3fn` quantizes each remote partial output to
  E4M3 with one UE8M0 scale per 128 columns.  Reduction remains FP32 and the
  final output remains BF16.
- PR #377's scheduler constructs tasks from the finalized token count of every
  expert.  It does not use the fixed expert waves or the assumed 2x imbalance
  factor present in `e1e5123`.

Native MXFP4 routed experts currently require `num_shared_experts=0`.  The
FP8-activation path continues to support fused shared experts, including FP8
combine transport.

## Low-concurrency configurations

For EP8, 384 routed experts, top-6, H=7168 and I=3072:

| tokens/rank | `e1e5123` | PR #377 rewrite |
|---:|---|---|
| 32 | BM16, BK256, 8 experts/wave | BM32, BK128, dynamic tasks |
| 64 | BM16, BK256, 8 experts/wave | BM32, BK128, dynamic tasks |
| 128 | BM32, BK128, 8 experts/wave | BM32, BK128, dynamic tasks |
| 256 | BM64, BK128, 8 experts/wave | BM96, BK128, dynamic tasks |

CUDA 13.1 `ptxas` static resources for the native-MXFP4/FP8-combine kernels:

| tokens/rank | `e1e5123` | PR #377 rewrite |
|---:|---|---|
| 32/64 | 128 registers, 32-byte stack, no spill | 128 registers, 64-byte stack, 8-byte spill load/store |
| 128 | 128 registers, 32-byte stack, no spill | 128 registers, 64-byte stack, 8-byte spill load/store |
| 256 | 168 registers, 32-byte stack, no spill | 128 registers, 56-byte stack, no spill |

With a requested 8192-token capacity (aligned by the public API to 8448), 148
SMs/rank, native MXFP4 activations, and FP8 combine, the symmetric-buffer
layout falls from 4.430 GiB in `e1e5123` to 1.059 GiB in the rewrite.  PR
#377's reusable rings reduce the allocation by 4.18x (3.371 GiB).

These EP8 numbers are static configuration and allocation comparisons.  The
EP4 runtime measurements below use the actual allocation reported by each
revision.

## GB300 runtime measurements

Measured on 2026-07-15 on one node with 4 NVIDIA GB300 GPUs (152 SMs/GPU),
driver 580.105.08, CUDA 13.1, and PyTorch 2.13.0+cu130.  Both revisions used
EP4, 384 experts, top-6, H=7168, I=3072, an 8192-token requested capacity
(aligned to 8448), native MXFP4 activations, and FP8 combine.  Results are the
median of 7 Kineto samples with 50 kernel launches per sample, after 5
warmups; latency is the maximum observed rank latency.

Uniform routing:

| tokens/rank | `e1e5123` (us) | PR #377 rewrite (us) | speedup | expert max/mean |
|---:|---:|---:|---:|---:|
| 32  | 469.37 | 465.16 | 1.009x | 3.67x |
| 64  | 520.31 | 517.25 | 1.006x | 3.31x |
| 128 | 526.33 | 523.41 | 1.006x | 2.50x |
| 256 | 548.87 | 530.54 | 1.035x | 1.88x |

Rank-balanced hot-expert routing (`--routing-skew 0.5`):

| tokens/rank | `e1e5123` (us) | PR #377 rewrite (us) | speedup | expert max/mean |
|---:|---:|---:|---:|---:|
| 32  | 337.55 | 324.97 | 1.039x | 20.00x |
| 64  | 417.31 | 400.33 | 1.042x | 19.73x |
| 128 | 532.42 | 510.40 | 1.043x | 17.08x |
| 256 | 644.59 | 588.42 | 1.095x | 17.88x |

No low-concurrency regression was observed, including under the 17--20x
max/mean expert imbalance.  The EP4 symmetric buffer fell from 2.533 GiB to
0.745 GiB per rank, a 70.6% reduction (3.40x smaller).

## Correctness

Run the self-contained single-rank reference test first:

```bash
PYTHONPATH=$PWD python tests/test_mega_moe_ref.py
```

The test covers FP8/BF16, FP8/FP8-combine, MXFP4/BF16, and
MXFP4/FP8-combine.  It can also validate the unpacked fallback:

```bash
DG_MEGA_MXF4_KIND=0 PYTHONPATH=$PWD python tests/test_mega_moe_ref.py
```

On the GB300 node, both native and fallback MXFP4 passed with a 0.0435
reference diff for BF16 combine and 0.0439 for FP8 combine (limits: 0.10).

## Apples-to-apples `e1e5123` benchmark

Build `e1e5123` and this branch in separate worktrees.  Run the same benchmark
file sequentially against both packages and use separate JIT caches:

```bash
REWRITE=/path/to/DeepGEMM
E1=/path/to/DeepGEMM-e1e5123

cd /tmp
DG_JIT_CACHE_DIR=/tmp/dg-jit-e1 \
PYTHONPATH="$E1" python "$REWRITE/tests/bench_mega_moe_ab.py" \
  --label e1e5123 --output /tmp/e1e5123-uniform.json

DG_JIT_CACHE_DIR=/tmp/dg-jit-pr377 \
PYTHONPATH="$REWRITE" python "$REWRITE/tests/bench_mega_moe_ab.py" \
  --label pr377-rewrite --output /tmp/pr377-uniform.json

python "$REWRITE/scripts/compare_mega_moe_results.py" \
  /tmp/e1e5123-uniform.json /tmp/pr377-uniform.json
```

The default sweep is 32, 64, 128, and 256 tokens/rank with EP8, 384 experts,
top-6, H=7168, I=3072, native MXFP4, and FP8 combine.  To exercise a
rank-balanced hot-expert distribution, repeat both commands with
`--routing-skew 0.5` and different output names.  The JSON records max/mean
expert load as well as the maximum kernel latency across ranks.
