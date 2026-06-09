# Design: FP8 Combine + MXFP4 Activations for Mega MoE

- **Date:** 2026-06-09
- **Status:** Approved (pending written-spec review)
- **Target:** SM100 / Blackwell (GB200), kernel `sm100_fp8_fp4_mega_moe`
- **Reference:** sgl-project/DeepGEMM#33 (reimplemented, not copied)

## 1. Context & goals

The parent repo (`deepseek-ai/DeepGEMM`) ships a fully-fused SM100 Mega MoE megakernel that does, in one launch over a symmetric NVLink buffer: dispatch a2a → L1 GEMM + SwiGLU → L2 GEMM → combine a2a → reduce. Today:

- **Activations** are E4M3 FP8 (`torch.float8_e4m3fn`), block-32 UE8M0 scaled. Weights are already packed E2M1 FP4 (`kPackedFP4 == torch::kInt8`), block-32 UE8M0 scaled. MMA is `kind::mxf8f6f4` (K=32).
- **Combine** transports BF16 partial expert outputs over NVLink and reduces them in FP32 to a BF16 `y`.

This work adds two opt-in capabilities:

1. **MXFP4 activations** — carry activations end-to-end as packed E2M1 + block-32 UE8M0 scales (dispatch a2a, both GEMM mainloops, L1 epilogue), and run them through the dense `tcgen05.mma.kind::mxf4` (K=64) instruction for higher throughput and halved A/B smem footprint.
2. **FP8 combine** — transport the combine a2a payload as E4M3 + block-128 UE8M0 scales instead of BF16, halving combine NVLink bytes per token. Final `y` remains BF16.

### Non-goals (explicitly out of scope)

- The standalone `mega_moe_pre_dispatch` GPU preprocessor kernel and its pybind/tvm-ffi wrappers (PR #33 stream "pre_dispatch"). Tests continue to fill the input buffer using the existing Python `per_token_cast_to_fp8` / `per_token_cast_to_fp4` quantizers.
- Any non-SM100 architecture.
- Changing the `y` output dtype (stays BF16).

## 2. Design principles ("elegant, not a copy")

PR #33 gated all four of its features behind `DG_USE_*` environment variables with template bools defaulting off. That was a cherry-pick shortcut to avoid touching a release branch's API. The parent repo's idiom is different and is what we follow:

- **Dtype-driven, single source of truth.** No env vars. The activation format is chosen by a readable `act_format` knob at `SymmBuffer` construction that sets the activation buffer dtype (`float8_e4m3fn` → FP8, `int8`/`kPackedFP4` → MXFP4, mirroring how weights are already passed). The kernel launch **infers** `kUseFp4Acts` from `l1_acts.scalar_type()` rather than receiving a redundant boolean — the allocated buffer can never disagree with what the kernel runs.
- **Idiomatic parameter style.** New options sit beside the existing `use_fp8_dispatch` / `activation` / `activation_clamp` / `fast_math` parameters on `SymmBuffer` / `get_symm_buffer_size_for_mega_moe` / `fp8_fp4_mega_moe`.
- **Zero-cost opt-in.** Each feature is an `if constexpr` template bool defaulting `false`. With defaults, generated code is byte-identical to today's baseline (the branches compile away). This is both a correctness guarantee and a backward-compat guarantee.
- **Reuse existing primitives.** The FP4 TMA path already exists in the parent repo and is wired in, not reinvented:
  - `kPackedFP4 = torch::kInt8` (`csrc/utils/math.hpp:11`).
  - `make_tma_2d_desc(..., fp4_unpacked_smem)` maps `kPackedFP4` → `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` (padded) or `_ALIGN8B` (dense) at `csrc/jit_kernels/impls/runtime_utils.hpp:86`.
  - `make_instr_desc_block_scaled<cutlass::float_e2m1_t, ...>` is already used (e.g. `sm100_fp4_mqa_logits.cuh:247`).
  - The 1-CTA `SM100_MMA_MXF4_SS` PTX already exists in `ptx/tcgen05.cuh`.

## 3. Public API contract

Both new options change the symmetric-buffer layout, so both are declared **once** at buffer construction and carried on the `SymmBuffer` object. The launch entry point takes no redundant flags.

### Python (`deep_gemm/mega/__init__.py`)

```python
class SymmBuffer:
    def __init__(self, group, num_experts,
                 num_max_tokens_per_rank, num_topk,
                 hidden, intermediate_hidden,
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu',
                 act_format: str = 'fp8',            # NEW: 'fp8' | 'mxfp4'
                 combine_dtype: torch.dtype = torch.bfloat16):  # NEW: bfloat16 | float8_e4m3fn
        ...

def get_symm_buffer_for_mega_moe(..., act_format='fp8', combine_dtype=torch.bfloat16) -> SymmBuffer: ...

def fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, ...):
    # No new flags. Reads act_format/combine from sym_buffer; passes them through.
    ...
```

- `act_format='mxfp4'` ⇒ `SymmBuffer.x` / `.l1_acts` / `.l2_acts` are allocated as `torch.int8` (`kPackedFP4`) with halved inner dims (`hidden/2`, `intermediate_hidden/2`). SF views are unchanged (still block-32). For `'fp8'`, behavior is exactly as today.
- `combine_dtype=torch.float8_e4m3fn` ⇒ combine token slot is `hidden` bytes (was `hidden*2`) plus a new parallel combine-SF slot of `hidden/128` bytes per token (block-128, non-TMA-aligned). For `torch.bfloat16`, behavior is exactly as today.

Rationale for the asymmetry (`act_format` string vs `combine_dtype` dtype): activations carry a *format* (packed E2M1 layout + block-32 SF + a different MMA path), so a named format reads better; combine is a pure transport-dtype swap (BF16↔FP8) with the SF mechanics implied, so a dtype reads naturally. (Open to unifying to strings during review if symmetry is preferred.)

### C++ (`csrc/apis/mega.hpp`)

- `get_symm_buffer_size_for_mega_moe(...)` takes `act_format` (`std::string`, matching the existing `activation` param) and `use_fp8_combine` (`bool`, derived in Python as `combine_dtype == torch.float8_e4m3fn`), and sizes the token / combine / combine-SF slots accordingly. The returned slice closure produces `int8` (`kPackedFP4`) views for `x`/`l1_acts`/`l2_acts` when `act_format == 'mxfp4'`.
- `fp8_fp4_mega_moe(...)` infers `use_fp4_acts` from `l1_acts.scalar_type() == kPackedFP4`, receives `use_fp8_combine` threaded from the `SymmBuffer` by the Python wrapper, derives `use_mxf4_kind` via heuristics, and asserts `!use_mxf4_kind || use_fp4_acts`.

**Single-source-of-truth summary:** the activation format is observable in the buffer dtype, so the launch *infers* `use_fp4_acts` from the passed tensor — the buffer cannot disagree with the kernel. The combine format is *not* observable (the combine buffer is internal, not a returned slice), so it is declared once via `combine_dtype` on the `SymmBuffer` and threaded consistently to both the sizer and the launch. Both knobs have exactly one authoring site: `SymmBuffer` construction.

## 4. Kernel template parameters

Append three bools after `kFastMath` in `sm100_fp8_fp4_mega_moe_impl<...>` (`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`), all defaulting `false`:

```
..., bool kFastMath = false,
bool kUseFp4Acts = false,
bool kUseMxf4Kind = false,
bool kUseFp8Combine = false
```

The JIT codegen `SM100FP8FP4MegaMoERuntime::generate_impl` (`csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:51`) currently substitutes 23 template args ending at `fast_math`; add three `{}` slots and corresponding `Args` fields fed from the inferred/derived values.

## 5. Feature 1 — FP8 combine (Phase 1; orthogonal, lower risk)

Touches only the combine a2a + reduce. Independent of FP4 acts.

### 5.1 Buffer layout

In both the host sizer (`csrc/apis/mega.hpp`, around lines 84–87) and the device-side layout chain inside the kernel (which mirrors it), the combine region becomes two slots:

```
constexpr uint32_t kCombineGranK = 128;   // assert kHidden % 128 == 0
combine_token_layout = layout::Data(kUseFp8Combine ? hidden : hidden*2);
combine_sf_layout    = layout::Data(kUseFp8Combine ? hidden/kCombineGranK : 0,
                                    /*require_tma_alignment=*/false);  // 7168/128 = 56, not %16
combine_token_buffer = layout::Buffer(combine_token_layout, num_topk, num_max_tokens_per_rank, l2_sf_buffer.get_end_ptr());
combine_sf_buffer    = layout::Buffer(combine_sf_layout,    num_topk, num_max_tokens_per_rank, combine_token_buffer.get_end_ptr());
```

The total-bytes return value ends at `combine_sf_buffer.get_end_ptr()`. `layout::Data`'s constructor already allows non-16B sizes when `require_tma_alignment=false` (`layout/mega_moe.cuh:186`).

### 5.2 L2 epilogue write (quantize-on-write)

Region: the L2 epilogue write-back in `sm100_fp8_fp4_mega_moe.cuh` (~lines 1100–1200). Baseline writes one BF16 `float4` per lane to the combine slot.

FP8 path (`if constexpr (kUseFp8Combine)`):
- Compute per-row amax across the row's `BLOCK_N=128` elements using a **16-lane** shuffle-reduce with mask `0x0000FFFFu << (16u * (lane_idx / 16))`. Full-warp reduction is **wrong** here because the other half-warp may have `break`'d on a padding row (landmine).
- UE8M0 SF from amax via existing `math::fast_log2_ceil(amax * (1/448))` → `sf_inv = fast_pow2(-log2_ceil)`, `sf_byte = log2_ceil + 127`.
- Scale 4 BF16 pairs by `sf_inv`, cast to 8 E4M3 (`__nv_fp8x4_e4m3` ×2 → `uint64`), write via `sym_buffer.map`.
- Write 1 SF byte per (row, N-block) by lane 0 of each 16-lane group into `combine_sf_buffer`.

### 5.3 Combine reduce (dequant-on-read)

Region: the combine reduce in `sm100_fp8_fp4_mega_moe.cuh` (~lines 1223–1352).
- Halve the per-chunk TMA load byte/uint4 counts under `kUseFp8Combine`.
- Change the `move_mask_and_load` lambda's return type from `bool` to `int` (returns the active slot index, or `-1`), so the reduce can index the correct per-slot SF pointer.
- For FP8: read the per-slot SF byte (`sf_idx = (chunk*elems + (j*32+lane)*16) / kCombineGranK`; 16 < 128, so all 16 elements of a uint4 share one SF), dequant E4M3→FP16→FP32 (`cvt.rn.f16x2.e4m3x2` then `cvt.f32.f16`), FMA into the FP32 accumulator scaled by `fast_pow2(sf_byte - 127)`.
- Store-to-smem and the TMA store to `y` stay BF16; only the number of store-uint4s changes. The final `y` byte offset is unchanged (`chunk * kNumChunkBytes`).

## 6. Feature 2 — MXFP4 activations + mxf4-kind (Phase 2; the larger change)

### 6.1 Buffers & TMA descriptors

- Host sizing: token slots become `kPackedFP4` of `hidden/2` (x, l1_acts) and `intermediate_hidden/2` (l2_acts) bytes. SF slots unchanged.
- `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`: build the A/B/L1-out/L2-acts descriptors for FP4. With `use_mxf4_kind`, set `fp4_unpacked_smem = false` (dense `_ALIGN8B`) and halve the swizzle modes; otherwise `_ALIGN16B` (padded). The L1-output descriptor uses **swizzle=0** (canonical dense packed) and N-width `block_n/4` (vs `block_n/2` for FP8). The L2-acts descriptor re-views the buffer as packed FP4.

### 6.2 Mainloop (TMA copy + MMA)

In `sm100_fp8_fp4_mega_moe.cuh`:
- `UMMA_K = kUseMxf4Kind ? 64 : 32`. Per-stage A/B smem and swizzle modes halve for dense FP4.
- A-load / B-load each become a 3-way `if constexpr`:
  - `kUseMxf4Kind`: raw `cute::SM100_TMA_2SM_LOAD_2D::copy` (the generic `tma::copy` computes `BLOCK_INNER_ATOM = swizzle/sizeof(dtype)` assuming ≥1-byte elements and mis-strides sub-byte FP4).
  - `kUseFp4Acts` (non-mxf4): `tma::copy<..., l1_a_dtype_t>` with the FP4 padded layout.
  - else: baseline FP8.
- `expect_tx` byte counts have **three regimes** (the diff comments are precise): FP8-dense or FP4-dense(mxf4) ⇒ `SMEM_*_SIZE_PER_STAGE * 2` (footprint == source bytes/peer × 2 peers); FP4-padded `_ALIGN16B` ⇒ `SMEM_*_SIZE_PER_STAGE` (smem is 2× source due to padding, so source-summed == footprint).
- Build a second instruction descriptor for FP4 with the **correct E2M1 enum**: pass `cutlass::float_e2m1_t` (→ `MXF4Format::E2M1 = 1`) for mxf4-kind, vs `cutlass::detail::float_e2m1_unpacksmem_t` (→ `MXF8F6F4Format::E2M1 = 5`) for the mxf8f6f4 path. Wrong enum ⇒ `cudaErrorIllegalInstruction` on first MMA (**landmine #1**).
- MMA issue branches: `kUseMxf4Kind` → new `SM100_MMA_MXF4_2x1SM_SS::fma` with SF TMEM address using **half-word offset `k*2`** for `scale_vec::2X` (**landmine #2**) and `uint8_t`/`BLOCK_K/2` descriptor advance; `kUseFp4Acts` → `SM100_MMA_MXF8F6F4_2x1SM_SS::fma` with the FP4 idesc; else baseline.

### 6.3 L1 epilogue (SwiGLU → E2M1 store)

Region: L1 epilogue in `sm100_fp8_fp4_mega_moe.cuh` (~lines 929–1048).
- Quantize SwiGLU output to E2M1 with new `math::get_e2m1_sf_and_sf_inv` (divisor `1/6` for E2M1's finfo max, vs `1/448` for E4M3) and `math::cvt_pack_f32_to_e2m1x2`.
- Under SwapAB, `tcgen05.ld` places adjacent N-columns on lanes `T` and `T XOR 4`; packing two values into one FP4 byte requires a `__shfl_xor_sync(..., 4)` to fetch the buddy, plus a half-warp donor gate (`group = lane/4`, active when `group % 2 == 0`). Active lanes write 4 bytes via `st.shared.u8` into the canonical dense smem layout matching the swizzle=0 descriptor (**landmine #3**).
- `L1_OUT_ROW_BYTES = kUseFp4Acts ? block_n/2 : block_n`; TMA-store N offset `out_n_idx = kUseFp4Acts ? n_block_idx*(L1_OUT_BLOCK_N/2) : n_block_idx*L1_OUT_BLOCK_N`.

### 6.4 Heuristics

`csrc/jit_kernels/heuristics/mega_moe.hpp`: thread `use_mxf4_kind` through `get_block_config_for_mega_moe` / `get_pipeline_config_for_mega_moe` / `get_mega_moe_config`. Effects: smallest-token-per-expert tier bumps `block_m` 16→32 (so `smem_a_per_stage` hits 1024B alignment); per-stage A/B smem halves for dense FP4 → more `num_stages`. `use_mxf4_kind` is derived here (auto-on when activations are MXFP4; may fall back if a shape cannot satisfy dense-smem constraints).

## 7. Math / PTX additions

- `deep_gemm/include/deep_gemm/common/math.cuh`:
  - `get_e2m1_sf_and_sf_inv<kUseUE8M0=true>(const float2& amax, float2& sf, float2& sf_inv)` — `1/6` divisor.
  - `cvt_pack_f32_to_e2m1x2(a, b)` (PTX `cvt.rn.satfinite.e2m1x2.f32`; low nibble = a) and, if useful, `cvt_pack_f32x4_to_e2m1x4(a,b,c,d)`.
  - Reuse existing `fast_log2_ceil`, `fast_pow2`, `get_e4m3_sf_and_sf_inv`.
- `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`:
  - `SM100_MMA_MXF4_2x1SM_SS` (cta_group::2 mxf4 block_scale; CUDA ≥12.9 uses `.block32`, else `.scale_vec::2X`). The 1-CTA `SM100_MMA_MXF4_SS` already exists.

## 8. File-by-file change map

| File | Change |
|---|---|
| `deep_gemm/mega/__init__.py` | `act_format`/`combine_dtype` on `SymmBuffer` + `get_symm_buffer_for_mega_moe`; pass through in `fp8_fp4_mega_moe`. |
| `csrc/apis/mega.hpp` | Size token/combine/combine-SF slots from `act_format`/`combine_dtype`; produce `kPackedFP4` views for FP4 acts; infer `use_fp4_acts` from tensor dtype; assert `!mxf4 || fp4_acts`; thread flags to kernel launch. |
| `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` | 3 new template `{}` slots + `Args` fields; build FP4/dense TMA descriptors (`fp4_unpacked_smem`, swizzle, `.view(kPackedFP4)`, swizzle=0 L1-out). |
| `csrc/jit_kernels/heuristics/mega_moe.hpp` | Thread `use_mxf4_kind`; mxf4 block_m bump; halve per-stage smem for dense FP4. |
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` | Core: 3 template bools; FP4 dtype switches; mxf4 K=64/dense/swizzle; raw 2SM TMA copies + `expect_tx` regimes; FP4 idesc enum; mxf4 MMA branch; FP4 L1 epilogue store; FP8-combine write + dequant reduce; device-side combine layout mirror. |
| `deep_gemm/include/deep_gemm/common/math.cuh` | `get_e2m1_sf_and_sf_inv`, `cvt_pack_f32_to_e2m1x2[ /_x4]`. |
| `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh` | `SM100_MMA_MXF4_2x1SM_SS`. |
| `tests/test_mega_moe.py` | Parametrize `(act_format, combine_dtype)`; FP4-quantize `x` for the MXFP4 case. |

## 9. Testing & verification (real, on GB200)

Hardware available: 4× NVIDIA GB200 (compute 10.0). Verification runs on-device, not just compile.

- **Matrix:** parametrize `tests/test_mega_moe.py` over `act_format ∈ {'fp8','mxfp4'}` × `combine_dtype ∈ {bf16, fp8e4m3}` = 4 configs. Quantize `x` with `per_token_cast_to_fp4(use_ue8m0=True, gran_k=32)` for MXFP4 (vs `per_token_cast_to_fp8(..., gran_k=32, use_packed_ue8m0=True)` today).
- **Methodology:** establish the FP8-acts / BF16-combine run-to-run noise floor first; then assert FP4-acts and FP8-combine outputs stay within an expected relative-RMSE band. PR #33 observed a ~0.5 noise floor on `y` once the L1 FP4 store layout was correct, vs 1.41 when the `__shfl_xor 4` packing was wrong — so the threshold doubles as a layout-correctness gate for landmine #3. FP8-combine contributes a modest additional error term.
- **Baseline invariance:** with defaults (`act_format='fp8'`, `combine_dtype=bf16`), confirm numerics are unchanged from `main` (the template branches must compile away). This guards backward compatibility.
- **Build:** the repo JIT-compiles kernels; confirm every instantiated template combination compiles cleanly.

## 10. Risks & landmines

1. **idesc E2M1 enum** — `float_e2m1_t` (mxf4, enum 1) vs `float_e2m1_unpacksmem_t` (mxf8f6f4, enum 5). Wrong choice ⇒ illegal-instruction at first MMA. Gate: any MXFP4 numerics test fails hard.
2. **SF TMEM half-word offset `k*2`** for `scale_vec::2X` — wrong offset ⇒ scrambled scales. Gate: MXFP4 rel-RMSE.
3. **FP4 L1 store `__shfl_xor 4` + half-warp donor gate** — wrong packing ⇒ rel-RMSE ≈ 1.4. Gate: the L1 sentinel-style threshold.
4. **16-lane (not full-warp) amax mask** in FP8-combine write — full-warp reduce reads lanes that `break` on padding rows. Gate: FP8-combine correctness on shapes with padding.

Mitigation: the two features are landed and tested sequentially (Phase 1 = FP8-combine, Phase 2 = MXFP4), so a regression is isolated to one feature's diff.

## 11. Implementation sequencing

1. **Phase 0 — primitives:** add `math.cuh` E2M1 helpers and the `SM100_MMA_MXF4_2x1SM_SS` PTX (no behavior change; compile-only).
2. **Phase 1 — FP8 combine:** `combine_dtype` API + buffer slots + L2 epilogue write + reduce. Test all-FP8-acts with bf16 vs fp8 combine. Land.
3. **Phase 2 — MXFP4 acts (+mxf4-kind):** `act_format` API + buffer/TMA + mainloop + MMA + L1 epilogue + heuristics. Test the full 4-config matrix. Land.

Backward compatibility is preserved throughout by the `false` template defaults and the unchanged default API values.
