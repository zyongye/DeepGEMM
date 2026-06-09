# Mega MoE FP8 Combine + MXFP4 Activations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. Companion spec: `docs/superpowers/specs/2026-06-09-megamoe-fp8combine-mxfp4-design.md`.

**Goal:** Add two opt-in capabilities to the SM100 fused Mega MoE kernel — (1) FP8 combine (E4M3+UE8M0 combine a2a instead of BF16) and (2) MXFP4 activations (packed E2M1 acts end-to-end through both GEMMs + the dense `kind::mxf4` K=64 MMA).

**Architecture:** Dtype-driven, idiomatic gating (no env vars). `act_format`/`combine_dtype` declared at `SymmBuffer` construction set buffer layout; the launch infers `kUseFp4Acts` from the buffer dtype and reads `use_fp8_combine` from the SymmBuffer. Three `if constexpr` template bools (`kUseFp4Acts`, `kUseMxf4Kind`, `kUseFp8Combine`) default `false` → byte-identical baseline.

**Tech Stack:** CUDA 13.0 (nvcc, aarch64), SM100/GB200, PyTorch 2.11, pybind11 `_C`, runtime JIT for `.cuh`.

## Build / verify mechanics
- **`.cuh` edits** (`impls/sm100_fp8_fp4_mega_moe.cuh`, `common/math.cuh`, `ptx/tcgen05.cuh`, `layout/mega_moe.cuh`): JIT-recompiled at runtime. Clear cache if stale: `rm -rf ~/.deep_gemm` (or set `DG_JIT_DEBUG=1`).
- **`.hpp`/pybind edits** (`csrc/apis/mega.hpp`, `csrc/jit_kernels/**`): rebuild `_C` (command confirmed in Task 1.1; likely `pip install --no-build-isolation -e .`).
- **Correctness gate** (no deep_ep/tilelang here): new self-contained test `tests/test_mega_moe_ref.py`, single-rank, vs `moe_reference`, via `calc_diff`. Run: `python tests/test_mega_moe_ref.py`.
- **Baseline-invariance gate:** with defaults, output must match current `main` numerics (template branches compile away).

---

## Phase 0 — Primitives (no behavior change)

### Task 0.1: E2M1 math helpers
**Files:** Modify `deep_gemm/include/deep_gemm/common/math.cuh` (after `get_e4m3_sf_and_sf_inv`, ~line 99).

- [ ] **Step 1: Add helpers** (place inside `namespace deep_gemm::math`, within the `#ifdef DG_IN_CUDA_COMPILATION` block):

```cpp
template <bool kUseUE8M0 = true>
CUTLASS_DEVICE void get_e2m1_sf_and_sf_inv(const float2& amax, float2& sf, float2& sf_inv) {
    DG_STATIC_ASSERT(kUseUE8M0, "Must use UE8M0");
    const float2 finfo_factor = {1.0f / 6.0f, 1.0f / 6.0f};  // E2M1 finfo.max = 6
    const auto scaled = __fmul2_rn(amax, finfo_factor);
    const auto exp_x = fast_log2_ceil(scaled.x);
    const auto exp_y = fast_log2_ceil(scaled.y);
    sf.x = fast_pow2(exp_x), sf_inv.x = fast_pow2(-exp_x);
    sf.y = fast_pow2(exp_y), sf_inv.y = fast_pow2(-exp_y);
}

// Pack two FP32 to two E2M1 nibbles. PTX `cvt.rn.satfinite.e2m1x2.f32 d, b, a` ⇒ low nibble = a.
CUTLASS_DEVICE uint32_t cvt_pack_f32_to_e2m1x2(const float& a, const float& b) {
    uint32_t out;
    asm volatile("{\n"
                 ".reg .b8 byte0;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
                 "cvt.u32.u8 %0, byte0;\n"
                 "}" : "=r"(out) : "f"(a), "f"(b));
    return out;
}
```

- [ ] **Step 2: Verify compile** — covered by Task 0.3 (header is included by the kernel).

### Task 0.2: 2-CTA mxf4 MMA PTX
**Files:** Modify `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh` (after `SM100_MMA_MXF4_SS`, ~line 140).

- [ ] **Step 1: Add struct** (mirror `SM100_MMA_MXF4_SS` but `cta_group::2`):

```cpp
struct SM100_MMA_MXF4_2x1SM_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a, uint64_t const& desc_b,
        uint32_t const& tmem_c, uint32_t const& scale_c, uint64_t const& desc,
        uint32_t const& tmem_sfa, uint32_t const& tmem_sfb) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
            "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
               "r"(tmem_sfa), "r"(tmem_sfb));
    }
};
```

### Task 0.3: Baseline still runs
- [ ] **Step 1:** `rm -rf ~/.deep_gemm` then `python tests/test_mega_moe.py --num-processes 1 --num-max-tokens-per-rank 512 --hidden 2048 --intermediate-hidden 1024 --num-experts 8 --num-topk 2` → expect a successful run (kernel JIT-compiles + benchmarks; no correctness assert without legacy). Confirms the new headers don't break compilation.
- [ ] **Step 2: Commit** `git add -A && git commit -m "Mega MoE: add E2M1 math helpers and 2-CTA mxf4 MMA PTX"`

---

## Phase 1 — FP8 combine

### Task 1.1: Write the self-contained reference test FIRST (TDD), confirm single-rank baseline
**Files:** Create `tests/test_mega_moe_ref.py`.

- [ ] **Step 1:** Write a single-process test that (a) builds `moe_reference` (float32, adapted from `test_moe_bf16.py`), (b) runs the mega kernel single-rank for a `(act_format, combine_dtype)` config, (c) compares via `calc_diff`. Parametrize over the 4 configs; for not-yet-implemented configs, `xfail`/skip. Establish the fp8/bf16 floor numerically. (Full skeleton in Appendix A.)
- [ ] **Step 2:** Confirm single-rank works: run with default `(fp8, bf16)` → expect `calc_diff < ~0.05`. If single-rank doesn't work, fall back to `--num-processes 2`.
- [ ] **Step 3: Commit** the test + measured baseline thresholds.

### Task 1.2: Plumb `use_fp8_combine` (API + template param, unused)
**Files:** `deep_gemm/mega/__init__.py`; `csrc/apis/mega.hpp`; `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`; `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`.

- [ ] **Step 1:** Add `combine_dtype: torch.dtype = torch.bfloat16` to `SymmBuffer.__init__` + `get_symm_buffer_for_mega_moe`; store `self.combine_dtype`. In `fp8_fp4_mega_moe` (py), pass `use_fp8_combine = (sym_buffer.combine_dtype == torch.float8_e4m3fn)` to `_C.fp8_fp4_mega_moe`. Assert `combine_dtype in {bf16, float8_e4m3fn}`.
- [ ] **Step 2:** `get_symm_buffer_size_for_mega_moe` (mega.hpp): add `use_fp8_combine` bool param (thread from the `_C.get_symm_buffer_size_for_mega_moe` pybind signature too); `fp8_fp4_mega_moe` (mega.hpp): add `use_fp8_combine` arg, forward to `sm100_fp8_fp4_mega_moe`.
- [ ] **Step 3:** JIT wrapper `.hpp`: add `bool use_fp8_combine` to `Args`, add a `{}` slot in `generate_impl` (after `fast_math`), pass `args.use_fp8_combine ? "true" : "false"`.
- [ ] **Step 4:** Kernel `.cuh`: add `bool kUseFp8Combine = false` template param after `kFastMath` (line 37). Unused for now.
- [ ] **Step 5:** Rebuild `_C`; `rm -rf ~/.deep_gemm`; rerun Task 1.1 default config → must still pass identically (byte-identical baseline).
- [ ] **Step 6: Commit.**

### Task 1.3: Combine buffer layout (token + SF), gated
**Files:** `csrc/apis/mega.hpp` (host sizer, ~lines 84-87, 132); kernel `.cuh` (device layout, ~lines 156-160).

- [ ] **Step 1:** mega.hpp: replace the single `combine_token_buffer` with conditional sizing:
```cpp
constexpr int kCombineGranK = 128;
const auto combine_token_layout = layout::Data(use_fp8_combine ? hidden : hidden * 2);
const auto combine_sf_layout = layout::Data(use_fp8_combine ? hidden / kCombineGranK : 0, /*require_tma_alignment=*/false);
const auto combine_token_buffer = layout::Buffer(combine_token_layout, num_topk, num_max_tokens_per_rank, l2_sf_buffer.get_end_ptr());
const auto combine_sf_buffer = layout::Buffer(combine_sf_layout, num_topk, num_max_tokens_per_rank, combine_token_buffer.get_end_ptr());
```
Return total ending at `combine_sf_buffer.get_end_ptr()`.
- [ ] **Step 2:** Kernel `.cuh`: mirror identically using `kUseFp8Combine`, `kHidden`, `kCombineGranK` (constexpr). Add `combine_sf_buffer` after `combine_token_buffer`.
- [ ] **Step 3:** Rebuild `_C`; rerun default config → byte-identical (use_fp8_combine=false path unchanged; the SF buffer is zero-sized).
- [ ] **Step 4: Commit.**

### Task 1.4: L2 epilogue FP8 write + combine reduce FP8 read
**Files:** kernel `.cuh` — L2 epilogue write (~lines 1172-1200) and combine reduce (~lines 1226-1352).

- [ ] **Step 1 (write):** In the L2 write loop, add `if constexpr (kUseFp8Combine)` branch: read the per-lane `float4` of 8 BF16 from smem (as today), compute row amax via 16-lane reduce (mask `0x0000FFFFu << (16u*(lane_idx/16))`, xor 1/2/4/8), `get_e4m3_sf_and_sf_inv`-style UE8M0 (divisor 1/448 → `fast_log2_ceil(amax/448)`), scale, cast 8 BF16 → 8 E4M3 (`__nv_fp8x4_e4m3 ×2` → `uint64`), write `uint64` to `combine_token_buffer` (8 bytes/lane at `(lane%16)*8`), write SF byte to `combine_sf_buffer` by `(lane&15)==0`. Keep the BF16 `else` path.
- [ ] **Step 2 (read):** Add `kNumLoadBytesPerChunk = kUseFp8Combine ? kNumChunkBytes/2 : kNumChunkBytes` and matching `kNumLoadUint4PerLane`. Change `move_mask_and_load` to return `int slot_idx` (or `-1`), loading `kNumLoadBytesPerChunk`. In the accumulate loop, FP8 path: read per-slot SF byte (`sf_idx = (chunk*kNumLoadElemsPerChunk + (j*32+lane)*16)/kCombineGranK`), dequant E4M3→FP16→FP32 (`cvt.rn.f16x2.e4m3x2`, `cvt.f32.f16`), `__fmaf_rn` into `reduced[...]` scaled by `fast_pow2(sf-127)`. Store-to-smem + TMA store to `y` stay BF16 (`kNumChunkBytes`). Keep the BF16 `else` path.
- [ ] **Step 3:** `rm -rf ~/.deep_gemm`; run Task 1.1 `(fp8, fp8-combine)` config → expect `calc_diff` within baseline floor + small combine-quant term (measure; target < ~2× floor). Run default `(fp8, bf16)` → unchanged.
- [ ] **Step 4: Commit.**

---

## Phase 2 — MXFP4 activations + mxf4-kind

### Task 2.1: Plumb `act_format` + `kUseFp4Acts`/`kUseMxf4Kind`; infer from buffer
**Files:** `deep_gemm/mega/__init__.py`; `csrc/apis/mega.hpp`; JIT wrapper `.hpp`; heuristics `mega_moe.hpp`; kernel `.cuh`.

- [ ] **Step 1:** Python: `act_format: str = 'fp8'` on `SymmBuffer`/`get_symm_buffer_for_mega_moe`; assert in `{'fp8','mxfp4'}`; store. (Buffer dtype switch happens in C++ slice closure.)
- [ ] **Step 2:** mega.hpp `get_symm_buffer_size_for_mega_moe`: add `act_format` (std::string); when `'mxfp4'`, size `x`/`l1_acts`/`l2_acts` token slots as `kPackedFP4` of `hidden/2` / `intermediate_hidden/2`, and have the slice closure produce `torch::kInt8` views (`{m, hidden/2}` etc.). `fp8_fp4_mega_moe` (mega.hpp): infer `use_fp4_acts = (l1_acts.scalar_type() == kPackedFP4)`; derive `use_mxf4_kind` (Step 4); assert `!use_mxf4_kind || use_fp4_acts`.
- [ ] **Step 3:** JIT wrapper `.hpp`: add `bool use_fp4_acts, use_mxf4_kind` to `Args`, 2 new `{}` slots; kernel `.cuh`: add `bool kUseFp4Acts=false, kUseMxf4Kind=false` after `kUseFp8Combine`.
- [ ] **Step 4:** Heuristics: derive `use_mxf4_kind = use_fp4_acts` (auto-on; the config returns dense-smem sizing — Task 2.6). Thread through `get_mega_moe_config`.
- [ ] **Step 5:** Rebuild; default `(fp8, *)` unchanged.
- [ ] **Step 6: Commit.**

### Task 2.2: FP4 dispatch-pull + buffer byte sizes in kernel
**Files:** kernel `.cuh` (~lines 99-101 token layouts; ~line 519 dispatch chunking).

- [ ] **Step 1:** `kInputTokenBytes = kUseFp4Acts ? kHidden/2 : kHidden`; `fp8_token_layout = layout::Data(kInputTokenBytes)`; `fp8_intermediate_token_layout = layout::Data(kUseFp4Acts ? kIntermediateHidden/2 : kIntermediateHidden)`. SF layouts unchanged (block-32).
- [ ] **Step 2:** Dispatch pull: replace `kHidden` with `kInputTokenBytes` in `kNumChunks` (line 519) and the assert. SF pull (`kNumSFUint32 = kHidden/128`) unchanged.
- [ ] **Step 3:** `rm -rf ~/.deep_gemm`; default config unchanged (compile check). FP4 path not yet correct (MMA/epilogue pending) — no numeric assert yet.
- [ ] **Step 4: Commit.**

### Task 2.3: FP4/dense TMA descriptors
**Files:** JIT wrapper `.hpp` (`sm100_fp8_fp4_mega_moe`, the `make_tma_*` calls ~lines 140-186).

- [ ] **Step 1:** Thread `use_fp4_acts`/`use_mxf4_kind` into the launcher. For acts/L1-out/L2-acts descriptors: `fp4_unpacked_smem = !use_mxf4_kind`; halve swizzle when `use_mxf4_kind`; L1-output uses swizzle=0 + N-width `block_n/4`; view the acts tensors as `kPackedFP4` (already are, post-2.2). Build a `_ALIGN8B` dense variant under mxf4-kind. (Reuses existing `make_tma_2d_desc(..., fp4_unpacked_smem)`.)
- [ ] **Step 2:** Rebuild; default unchanged.
- [ ] **Step 3: Commit.**

### Task 2.4: Mainloop — UMMA_K, TMA copy, expect_tx, idesc enum, MMA branch
**Files:** kernel `.cuh` (~lines 173 UMMA_K; 700-710/743-753 TMA copy + expect_tx; 765-769 idesc; 841-853 MMA issue; data-type aliases ~164-165, smem sizing ~205-208, swizzle ~180-181).

- [ ] **Step 1:** Data types: `l1_a_dtype_t/l2_a_dtype_t = kUseFp4Acts ? b_dtype_t : a_dtype_t`. `UMMA_K = kUseMxf4Kind ? 64 : 32`. Swizzle/smem A/B per-stage sizes halve under mxf4 (`kSwizzle*Mode`, `smem_a/smem_b`).
- [ ] **Step 2:** A/B TMA load: 3-way `if constexpr` — `kUseMxf4Kind` → raw `cute::SM100_TMA_2SM_LOAD_2D::copy`; `kUseFp4Acts` → `tma::copy<..., l1_a_dtype_t>`; else baseline. Set `expect_tx` per regime (FP8/FP4-dense: `smem*2`; FP4-padded: `smem`).
- [ ] **Step 3:** Build FP4 idesc with correct enum: `make_instr_desc_block_scaled<cute::conditional_t<kUseMxf4Kind, cute::float_e2m1_t, b_dtype_t>, ...>` (mxf4 enum=1 vs mxf8f6f4 enum=5). MMA issue: `if constexpr (kUseMxf4Kind)` → `SM100_MMA_MXF4_2x1SM_SS::fma` with SF id `k*2`; `else if (kUseFp4Acts)` → `SM100_MMA_MXF8F6F4_2x1SM_SS` with FP4 idesc; else baseline.
- [ ] **Step 4:** Compile check; numeric correctness deferred to 2.5 (L1 store needed for full path).
- [ ] **Step 5: Commit.**

### Task 2.5: L1 epilogue FP4 store
**Files:** kernel `.cuh` (~lines 1024-1078 store; smem_d union ~202; `L1_OUT_BLOCK_N`/row bytes ~195).

- [ ] **Step 1:** `L1_OUT_ROW_BYTES = kUseFp4Acts ? L1_OUT_BLOCK_N/2 : L1_OUT_BLOCK_N`; size `smem_d.l1` accordingly. In the cast loop, `if constexpr (kUseFp4Acts)`: `get_e2m1_sf_and_sf_inv`, scale, `__shfl_xor_sync(...,4)` to fetch the SwapAB buddy column, half-warp donor gate (`group=lane/4`, active `group%2==0`), pack via `cvt_pack_f32_to_e2m1x2`, write 4 bytes via `st.shared.u8` into canonical dense smem. SF store to `l2_sf_buffer` as today. Keep FP8 STSM `else` path. TMA-store N offset `n_block_idx*(L1_OUT_BLOCK_N/2)` for FP4.
- [ ] **Step 2:** `rm -rf ~/.deep_gemm`; run `(mxfp4, bf16)` config → expect `calc_diff` within FP4 floor (measure; if ≈1.4, the buddy-shuffle layout is wrong — landmine #3). Default `(fp8,bf16)` unchanged.
- [ ] **Step 3: Commit.**

### Task 2.6: Heuristics for mxf4-kind dense smem
**Files:** `csrc/jit_kernels/heuristics/mega_moe.hpp`.

- [ ] **Step 1:** Thread `use_mxf4_kind` into `get_block_config`/`get_pipeline_config`/`get_mega_moe_config`. Halve per-stage A/B smem (`smem_a_size_per_stage`/`smem_b_size_per_stage` → `/2`) under mxf4; bump smallest-token tier `block_m 16→32` if needed for 1024B smem alignment.
- [ ] **Step 2:** Rebuild; rerun `(mxfp4,*)` configs → still correct, expect more stages / higher TFLOPS.
- [ ] **Step 3: Commit.**

### Task 2.7: Full 4-config matrix + clean up
- [ ] **Step 1:** Enable all 4 `(act_format, combine_dtype)` configs in `tests/test_mega_moe_ref.py`; assert each within its measured floor. Run across a few shapes (top_k 1/2/4, uneven load).
- [ ] **Step 2:** Also update `tests/test_mega_moe.py` to accept `--act-format`/`--combine-dtype` for benchmarking (optional perf check).
- [ ] **Step 3: Commit.**

---

## Self-review notes
- **Spec coverage:** §5 (FP8 combine) → Tasks 1.2-1.4; §6 (MXFP4) → Tasks 2.1-2.6; §3 API → 1.2/2.1; §7 math/ptx → Phase 0; §9 testing → 1.1/2.7. Covered.
- **Type consistency:** template-bool order is fixed as `..., kFastMath, kUseFp8Combine, kUseFp4Acts, kUseMxf4Kind` (Phase 1 adds combine first, Phase 2 appends the two FP4 bools — matches the sequential staging). The JIT `generate_impl` `{}` slots must be appended in this same order.
- **No placeholders:** kernel-internal edits (1.4, 2.4, 2.5) give mechanism + exact anchors + key snippets; final code is produced at execution with compile/numeric feedback (the only reliable path for sub-byte FP4 packing / SF offsets). All deterministic pieces (Phase 0, API, test skeleton) are given in full.

## Appendix A — `tests/test_mega_moe_ref.py` skeleton
Single-process, num_ranks=1, vs `moe_reference`. Build `x/topk` like `test_mega_moe.py`, quantize per `act_format`, allocate `SymmBuffer(..., act_format=, combine_dtype=)`, run `fp8_fp4_mega_moe`, `calc_diff(y.float(), ref.float())`. (Reuse `moe_reference` + `_route` from `test_moe_bf16.py`; mega applies topk weight at L1 epilogue — equivalent to reference's combine-time weighting.) Full code authored in Task 1.1 once single-rank behavior is confirmed.
