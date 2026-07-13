# Mega MoE — MXFP4 Activations + FP8 Combine (engine integration)

The fused SM100 (Blackwell/GB200) **Mega MoE** kernel does an entire EP MoE layer in a
single launch over a PyTorch symmetric-memory buffer:

```
dispatch a2a  →  L1 GEMM (gate/up) + SwiGLU  →  L2 GEMM (down)  →  combine a2a + reduce  →  y
```

This doc covers two opt-in capabilities and how to drive the kernel end-to-end from an
inference engine:

- **MXFP4 activations** (`act_format='mxfp4'`) — activations carried as packed E2M1 (FP4)
  + block-32 UE8M0 scales, end-to-end through both GEMMs. ~½ the activation HBM/NVLink
  bytes vs FP8.
- **FP8 combine** (`combine_dtype=torch.float8_e4m3fn`) — the combine all-to-all ships
  E4M3 + block-128 UE8M0 instead of BF16 (½ the combine NVLink bytes). `y` stays BF16.

Both are **off by default** (FP8 acts, BF16 combine) — existing callers are unaffected.

---

## 1. Data layout

All scale factors are **UE8M0** (power-of-two, 1 byte per block) packed 4-per-`int32`.

### Activations (what the engine fills)

| tensor | dtype | shape (per rank) | notes |
|---|---|---|---|
| `buffer.x` (FP8) | `float8_e4m3fn` | `[num_max_tokens, hidden]` | default |
| `buffer.x` (**MXFP4**) | `int8` (`kPackedFP4`) | `[num_max_tokens, hidden/2]` | 2 E2M1/byte: low nibble = even col, high nibble = odd col |
| `buffer.x_sf` | `int32` | `[num_max_tokens, hidden/128]` | block-32 UE8M0, 4 bytes/`int32`, **K-major** |
| `buffer.topk_idx` | `int64` | `[num_max_tokens, num_topk]` | **global** expert ids; `-1` = no expert |
| `buffer.topk_weights` | `float32` | `[num_max_tokens, num_topk]` | combine weights; `0` for masked slots |

`hidden` and `intermediate_hidden` must be multiples of 128. The SF block granularity is
32 for both FP8 and MXFP4 activations (so `x_sf` is identical in shape for both).

### Weights (prepared offline, passed each call)

FP4 E2M1, block-32 UE8M0, after `transform_weights_for_mega_moe` (gate/up interleave for
L1, UTCCP-transposed SF):

- `l1_weights = (w, w_sf)` for the fused gate+up: logical `(E_local, 2*intermediate, hidden)`
- `l2_weights = (w, w_sf)` for down: logical `(E_local, hidden, intermediate)`

SwiGLU convention: the first `intermediate` rows of L1 are the **gate**, the next
`intermediate` are the **up**; output = `silu(gate) * up`.

### Internal (the kernel manages these; the engine does not touch them)

- L1 output / L2 input activations: packed E2M1 (`intermediate/2` bytes/token) when MXFP4.
- Combine buffer: BF16 `[num_topk, num_max_tokens, hidden]`, or E4M3 `[..., hidden]` +
  a parallel per-128 UE8M0 SF slot when `combine_dtype=float8_e4m3fn`.

### `mxf4`-kind (perf only, transparent)

When MXFP4 acts are on, the L1/L2 GEMMs run the dense `tcgen05.mma.kind::mxf4.block_scale`
(K=64, 2 nibbles/byte smem) by default; set `DG_MEGA_MXF4_KIND=0` to force the
`kind::mxf8f6f4` path (K=32, unpacked smem). **Numerics are identical** (both block-32
E2M1) — this only affects speed, and is shape-dependent (helps GEMM-bound/prefill shapes).

---

## 2. Public API

```python
import torch, deep_gemm

buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group,                       # torch.distributed ProcessGroup (size 1 works for single-GPU)
    num_experts,                 # global expert count (must be divisible by group.size())
    num_max_tokens_per_rank,     # capacity; aligned up internally
    num_topk, hidden, intermediate_hidden,
    act_format='mxfp4',          # 'fp8' (default) or 'mxfp4'
    combine_dtype=torch.bfloat16 # or torch.float8_e4m3fn for FP8 combine
)

# Offline, once: FP4-quantize + transform weights (see helper below)
tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1_weights_fp4, l2_weights_fp4)

# Per forward
deep_gemm.fp8_fp4_mega_moe(
    y=y,                         # output, (num_tokens, hidden) bf16
    l1_weights=tl1, l2_weights=tl2,
    sym_buffer=buffer,
    cumulative_local_expert_recv_stats=None,  # optional int32[E_local] counter
    recipe=(1, 1, 32), activation='swiglu',
    activation_clamp=None, fast_math=True,
)
```

`act_format` / `combine_dtype` are declared **once** on the `SymmBuffer` (they set the
buffer layout). The launch infers `use_fp4_acts` from the buffer dtype and reads the
combine format from the buffer — there are no redundant flags to keep in sync.

---

## 3. How to test

Single-rank correctness vs a float32 PyTorch reference (no `deep_ep`/`tilelang` needed):

```bash
PYTHONPATH=$PWD python tests/test_mega_moe_ref.py
```
Expected (`calc_diff` vs reference; lower is better):
```
[floor] fp8   / bf16      0.0215
[feat]  fp8   / fp8e4m3   0.0219     # FP8 combine
[feat]  mxfp4 / bf16      0.0435     # MXFP4 acts
[feat]  mxfp4 / fp8e4m3   0.0439     # both
PASS
```
- Force the `mxf8f6f4` fallback for A/B: `DG_MEGA_MXF4_KIND=0 PYTHONPATH=$PWD python tests/test_mega_moe_ref.py` (same diffs).
- Multi-rank perf/correctness harness: `python tests/test_mega_moe.py --num-processes 8 ...`
  (compares bitwise vs the legacy `deep_ep`+`tilelang` baseline when installed).

Build note: device kernels are JIT-compiled at runtime (the first mega compile is slow,
~15–25 min; subsequent runs hit `~/.deep_gemm`). C++/pybind changes need a rebuild
(`bash develop.sh`); `.cuh` changes need `rm -rf ~/.deep_gemm`.

---

## 4. End-to-end engine integration

### One-time setup (per EP rank)

```python
import torch, torch.distributed as dist
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4

# (a) symmetric buffer for the layer
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk,
    hidden, intermediate_hidden,
    act_format='mxfp4', combine_dtype=torch.float8_e4m3fn)

# (b) FP4-quantize + transform the LOCAL experts' weights, once
def cast_weights_to_fp4(bf16_w):            # bf16_w: (E_local, N, K)
    E, N, K = bf16_w.shape
    w   = torch.empty((E, N, K // 2),  device='cuda', dtype=torch.int8)
    wsf = torch.empty((E, N, K // 32), device='cuda', dtype=torch.float)
    for e in range(E):
        w[e], wsf[e] = per_token_cast_to_fp4(bf16_w[e], use_ue8m0=True, gran_k=32)
    wsf = deep_gemm.transform_sf_into_required_layout(wsf, N, K, (1, 32), E)
    return w, wsf

l1_fp4 = cast_weights_to_fp4(l1_bf16)        # (E_local, 2*intermediate, hidden)
l2_fp4 = cast_weights_to_fp4(l2_bf16)        # (E_local, hidden, intermediate)
tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1_fp4, l2_fp4)
```

### Per forward pass

```python
# x: (T, hidden) bf16 activations for this rank's tokens; T <= num_max_tokens_per_rank
# topk_idx: (T, num_topk) int64 GLOBAL expert ids (-1 for masked); topk_weights: (T, num_topk) f32
xq, xsf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
buffer.x[:T].copy_(xq)
buffer.x_sf[:T].copy_(xsf)
buffer.topk_idx[:T].copy_(topk_idx)
buffer.topk_weights[:T].copy_(topk_weights)

y = torch.empty((T, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(y=y, l1_weights=tl1, l2_weights=tl2, sym_buffer=buffer,
                           activation_clamp=clamp_or_None, fast_math=True)
# y is the combined MoE output for this rank's T tokens.
```

The kernel internally: pulls each token to the rank(s) owning its top-k experts, runs L1
+ SwiGLU + L2 per expert, then scatters/​reduces the weighted partial outputs back to each
token's home rank. The engine only provides `x`, `x_sf`, `topk_idx`, `topk_weights` and
reads `y`.

### Multi-rank (EP) notes

- The buffer is allocated via `torch.distributed._symmetric_memory` when `group.size() > 1`
  (and plain `torch` for size 1). All ranks must allocate identically and rendezvous.
- `topk_idx` holds **global** expert ids; experts `[r*E_local, (r+1)*E_local)` live on rank
  `r` (`num_experts % num_ranks == 0`).
- `num_max_tokens_per_rank` is the per-rank capacity; pad/mask unused slots with
  `topk_idx=-1`, `topk_weights=0`.
- `cumulative_local_expert_recv_stats` (optional `int32[E_local]`) accumulates per-expert
  received-token counts for diagnostics/balancing.

### Constraints checklist

- SM100 (Blackwell). `hidden`, `intermediate_hidden` multiples of 128.
- `recipe == (1, 1, 32)`, `activation == 'swiglu'`.
- Weights FP4 E2M1 + block-32 UE8M0, passed through `transform_weights_for_mega_moe`.
- `combine_dtype ∈ {torch.bfloat16, torch.float8_e4m3fn}`; `act_format ∈ {'fp8','mxfp4'}`.

---

## 5. Quick A/B for your engine

| | act bytes | combine bytes | MMA |
|---|---|---|---|
| `act_format='fp8'`, `combine_dtype=bf16` | `hidden` | `2*hidden` | mxf8f6f4 |
| `act_format='mxfp4'`, `combine_dtype=bf16` | `hidden/2` | `2*hidden` | mxf4 (or mxf8f6f4 via `DG_MEGA_MXF4_KIND=0`) |
| `act_format='mxfp4'`, `combine_dtype=fp8e4m3` | `hidden/2` | `hidden + hidden/128` | mxf4 |

Start with `act_format='mxfp4', combine_dtype=torch.bfloat16`, confirm output parity
against your reference MoE (expect a small relative diff, ~0.04 on random data), then turn
on FP8 combine. Numerics are unchanged by `DG_MEGA_MXF4_KIND`, so flip it purely for perf.
