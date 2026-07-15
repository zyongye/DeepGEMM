# Benchmarking the fused Mega MoE kernel (FP8 / MXFP4)

How to reproduce the Mega MoE performance numbers on another SM100 (Blackwell / GB200)
machine. Covers the perf benchmark, the correctness check, and an optional
apples-to-apples comparison against the upstream SGL DeepGEMM PR.

## 1. Requirements

- **GPU:** SM100 (Blackwell, e.g. GB200 / B200). The fused kernel and FP4 MMA paths are
  SM100-only — it will not build/run on Hopper.
- **Toolchain:** CUDA 13.x (NVCC with `sm_100a`), a recent PyTorch with NVSHMEM/symmetric
  memory, Python 3.10+.
- This repo checked out on branch `megamoe-fp8combine-mxfp4`.

## 2. Build

```bash
cd DeepGEMM
git submodule update --init --recursive      # third_party/{cutlass,fmt,...}
bash develop.sh                               # builds the host _C extension (pybind)
```

`develop.sh` compiles the host side (`csrc/`) into `deep_gemm/_C*.so`. The **device**
kernels (`*.cuh`) are JIT-compiled on first use and cached under `~/.deep_gemm`.

> **First run is slow.** The fused mega kernel takes **~15–25 min** to JIT-compile the
> first time (per distinct compute config). Subsequent runs hit the cache and start in
> seconds. Keep `~/.deep_gemm` warm between runs.

Always run with the repo on `PYTHONPATH` so you don't accidentally import a stale
site-packages `deep_gemm`:

```bash
export PYTHONPATH=$PWD
python -c "import deep_gemm, torch; print(deep_gemm.__file__); print(torch.cuda.get_device_name())"
```

The printed path must point inside this repo, and the device must report an SM100 part.

## 3. Perf benchmark (single GPU)

```bash
PYTHONPATH=$PWD python tests/bench_mega_moe.py
```

Benchmarks the full fused kernel (dispatch a2a → L1 GEMM + SwiGLU → L2 GEMM → combine)
with `bench_kineto` (isolates GPU kernel time), across three compute paths:

| label | activations | weights | MMA |
|---|---|---|---|
| `fp8 acts x fp4 wt`  | FP8 E4M3 (UE8M0) | packed FP4 | `kind::mxf8f6f4`, K=32 |
| `mxfp4 acts (mxf8f6f4)` | packed MXFP4 | packed FP4 | `kind::mxf8f6f4`, K=32 |
| `mxfp4 acts (mxf4-kind)` | packed MXFP4 | packed FP4 | `kind::mxf4` dense, K=64 |

The third path is toggled internally by `DG_MEGA_MXF4_KIND` (the script sets it per run;
default is `1` = mxf4-kind on).

**Sweep shapes** with CLI flags (try a prefill-scale config where the GEMM dominates and
mxf4-kind can pull ahead):

```bash
# small / decode-ish (default) -- overhead-bound, all paths ~equal
PYTHONPATH=$PWD python tests/bench_mega_moe.py \
    --num-tokens 512 --hidden 2048 --intermediate 1024 --num-experts 8 --num-topk 2

# prefill / GEMM-bound -- where mxf4-kind should win
PYTHONPATH=$PWD python tests/bench_mega_moe.py \
    --num-tokens 4096 --hidden 7168 --intermediate 2048 --num-experts 8 --num-topk 8
```

Flags: `--num-tokens --hidden --intermediate --num-experts --num-topk --num-tests --clamp`
(`hidden` and `intermediate` must be multiples of 128; `num-experts` a multiple of world size).

### Reference numbers (GB200, single rank, T=512 H=2048 I=1024 E=8 K=2)

```
fp8 acts x fp4 wt  (mxf8f6f4 K=32)          ~39.1 us     ~330 TFLOPS
mxfp4 acts         (mxf8f6f4 K=32)          ~39.9 us     ~323 TFLOPS
mxfp4 acts         (mxf4-kind K=64)         ~40.0 us     ~322 TFLOPS
```

At this small shape the kernel is **overhead-bound** (dispatch+combine dominates), so all
three paths land within ~3% and mxf4-kind shows no GEMM advantage. Use a prefill-scale
config to see the FP4-compute benefit.

## 4. Correctness check

Self-contained single-rank harness, validates kernel output against a PyTorch reference
for all three formats (fp8/bf16-combine, mxfp4/bf16-combine, mxfp4/fp8-combine):

```bash
PYTHONPATH=$PWD python tests/test_mega_moe_ref.py
```

Pass criteria are per-format relative-error thresholds (printed per config). Run this on a
new machine before trusting perf numbers.

## 5. Multi-rank

`bench_mega_moe.py` / `test_mega_moe_ref.py` are single-rank (`init_dist(0, 1)`). For the
real multi-GPU a2a path, use the multi-process harness:

```bash
PYTHONPATH=$PWD python tests/test_mega_moe.py --num-processes 8
```

(Set `MASTER_ADDR`/`MASTER_PORT` if the default `127.0.0.1:8361` is taken.)

### Low-concurrency routing-imbalance sweep

The default router logits are statistically uniform. Use `--routing-skew` to repeat a
hot-to-cold local-expert bias on every EP rank. This keeps traffic balanced across ranks
while exercising the within-rank imbalance seen with real router outputs. The harness
prints the active-expert count and max/mean expert load before timing.

```bash
for tokens in 32 64 128 256; do
    PYTHONPATH=$PWD python tests/test_mega_moe.py \
        --num-processes 8 \
        --num-max-tokens-per-rank 384 \
        --num-tokens "$tokens" \
        --hidden 7168 \
        --intermediate-hidden 3072 \
        --num-experts 384 \
        --num-topk 6 \
        --mma-type mxfp4xmxfp4 \
        --combine-dtype bf16 \
        --routing-skew 0.5
done
```

Run the same sweep with `--routing-skew 0` for the uniform control. Values around
`0.25` are moderately skewed; `0.5` is intentionally stressful.

## 6. Optional: compare against the SGL DeepGEMM PR

To reproduce the head-to-head against the upstream PR
(sgl-project/DeepGEMM#33 / #27 port), check that branch out into a **separate worktree**
(so `~/.deep_gemm` JIT caches and `_C.so` don't collide), build it, and run an equivalent
env-gated benchmark. The PR gates its features with env vars instead of dtype params:

```bash
DG_USE_FP4_ACTS=1   # MXFP4 activations
DG_USE_MXF4_KIND=1  # dense mxf4 MMA
```

Build the PR worktree (`bash develop.sh`) and run a script analogous to
`tests/bench_mega_moe.py` that sets those env vars per config and calls
`get_symm_buffer_for_mega_moe(...)` without `act_format`. Run the two benchmarks
**sequentially, never concurrently** — both bind the same dist port and share the GPU, so a
concurrent launch fails with `EADDRINUSE` / NCCL init errors and skews timing.

### Reference comparison (GB200, T=512 H=2048 I=1024 E=8 K=2)

| path | this impl | SGL PR-port |
|---|---|---|
| fp8 acts × fp4 wt (mxf8f6f4) | ~39.1 µs | ~39.0 µs |
| mxfp4 acts (mxf8f6f4) | ~39.9 µs | ~39.3 µs |
| mxfp4 acts (mxf4-kind) | ~40.0 µs | ~38.8 µs |

Within ~3% across the board at this overhead-bound shape (the PR carries some extra mxf4
register-partition tuning). Re-run at prefill scale for a GEMM-bound comparison.

## 7. Troubleshooting

| symptom | fix |
|---|---|
| `import deep_gemm` resolves to site-packages | set `PYTHONPATH=$PWD`; uninstall stale wheel |
| `TypeError` on `fp8_fp4_mega_moe` arg count | stale `_C.so`; rerun `bash develop.sh` |
| `DistNetworkError ... EADDRINUSE` port 8361 | another rank/bench is using the port; run sequentially or set `MASTER_PORT` |
| first run hangs for minutes | expected — JIT compiling the mega kernel; watch `~/.deep_gemm` |
| NCCL `unhandled cuda error` at startup | two processes initializing dist on the same GPU/port; run one at a time |
