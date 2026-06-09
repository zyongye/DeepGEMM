"""
Self-contained correctness harness for the fused SM100 Mega MoE kernel.

Unlike `test_mega_moe.py` (which checks bitwise-equality against a legacy
deep_ep/tilelang baseline, unavailable here), this test:

  * Primary gate  — runs the kernel for a feature config and compares against
    the shipped baseline (act_format='fp8', combine_dtype='bf16'). The delta is
    purely the added quantization/transport noise of the feature, so it sidesteps
    all SwiGLU/interleave/routing-convention questions.
  * Absolute gate — anchors the baseline itself against a pure-PyTorch float32
    MoE reference (`moe_reference_routed`), establishing the quant noise floor.

Runs single-rank (num_ranks=1, plain torch allocator). Usage:
    PYTHONPATH=. python tests/test_mega_moe_ref.py
"""

import torch
import torch.nn.functional as F

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist
from deep_gemm.testing import calc_diff


# ---------------------------------------------------------------------------
# Pure-PyTorch reference (mega-kernel semantics)
# ---------------------------------------------------------------------------
def moe_reference_routed(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                         topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """Float32 MoE. w1=(E,2I,H), w2=(E,H,I). SwiGLU: gate=first half (matches
    `transform_weights_for_mega_moe`'s `_interleave_weights`), act=silu(gate)*up.
    Top-k weight applied at combine == applied at L1 (GEMM is linear)."""
    num_tokens, hidden = x.shape
    intermediate = w1.shape[1] // 2
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in range(w1.shape[0]):
        mask = (topk_idx == e)                       # (T, K)
        sel = mask.any(dim=1)
        if not bool(sel.any()):
            continue
        tok = sel.nonzero().flatten()
        we = (topk_weights.float() * mask.float()).sum(dim=1)[tok]   # (n,)
        fc1 = xf[tok] @ w1[e].float().T              # (n, 2I)
        gate, up = fc1[:, :intermediate], fc1[:, intermediate:]
        act = F.silu(gate) * up                      # (n, I)
        fc2 = act @ w2[e].float().T                  # (n, H)
        out[tok] += we.unsqueeze(1) * fc2
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Weight quantization (FP4 + UE8M0, matching test_mega_moe.py)
# ---------------------------------------------------------------------------
def cast_weights_to_fp4(bf16_weights: torch.Tensor):
    num_groups, n, k = bf16_weights.shape
    w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
    w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
    for i in range(num_groups):
        w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
    w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
    return w, w_sf


# ---------------------------------------------------------------------------
# Kernel runner
# ---------------------------------------------------------------------------
def run_mega(x_bf16, transformed_l1, transformed_l2, topk_idx, topk_weights,
             group, num_experts, hidden, intermediate, num_topk,
             act_format='fp8', combine_dtype='bf16', activation_clamp=None):
    # NOTE: act_format/combine_dtype are wired to the API as those features land.
    T = x_bf16.shape[0]
    kwargs = {}
    if act_format != 'fp8':
        kwargs['act_format'] = act_format
    if combine_dtype != 'bf16':
        kwargs['combine_dtype'] = {'fp8e4m3': torch.float8_e4m3fn}[combine_dtype]
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, T, num_topk, hidden, intermediate, **kwargs)

    if act_format == 'mxfp4':
        xq, xsf = per_token_cast_to_fp4(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    else:
        xq, xsf = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    buffer.x[:T].copy_(xq)
    buffer.x_sf[:T].copy_(xsf)
    buffer.topk_idx[:T].copy_(topk_idx)
    buffer.topk_weights[:T].copy_(topk_weights)

    y = torch.empty((T, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_fp4_mega_moe(
        y=y, l1_weights=transformed_l1, l2_weights=transformed_l2, sym_buffer=buffer,
        activation_clamp=activation_clamp, fast_math=True)
    y = y.clone()
    buffer.destroy()
    return y


def main():
    rank_idx, num_ranks, group = init_dist(0, 1)
    torch.manual_seed(0)

    T, H, I, E, K = 512, 2048, 1024, 8, 2
    # Well-conditioned scaling so fc1 ~ O(1) (avoids quant overflow)
    x = torch.randn(T, H, dtype=torch.bfloat16, device='cuda') * (H ** -0.5)
    l1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device='cuda') * (H ** -0.5)
    l2 = torch.randn(E, H, I, dtype=torch.bfloat16, device='cuda') * (I ** -0.5)
    scores = torch.randn(T, E, dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(torch.int64)

    # Reference (bf16 weights, original x)
    ref = moe_reference_routed(x, l1, l2, topk_idx, topk_weights)

    # FP4-quantize + transform weights for the mega kernel
    l1q, l2q = cast_weights_to_fp4(l1), cast_weights_to_fp4(l2)
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1q, l2q)

    print(f'Config: T={T} H={H} I={I} E={E} K={K}')

    # Baseline (shipped) path
    y_base = run_mega(x, tl1, tl2, topk_idx, topk_weights, group, E, H, I, K,
                      act_format='fp8', combine_dtype='bf16')
    floor = calc_diff(y_base.float(), ref.float())
    print(f'[floor] fp8 acts / bf16 combine  vs reference : diff={floor:.4f}')
    assert floor < 0.30, f'baseline floor too high ({floor:.4f}) — convention/quantization issue'

    # Feature configs (enabled as they land). Compared to y_base (quant-noise delta).
    # (act_format, combine_dtype, max rel-diff vs reference)
    configs = [
        ('fp8',   'fp8e4m3', 0.05),   # Phase 1 — FP8 combine (≈ the fp8/bf16 floor 0.0215)
        ('mxfp4', 'bf16',    0.10),   # Phase 2 — MXFP4 acts (measured 0.0435 vs ref)
        ('mxfp4', 'fp8e4m3', 0.10),   # Phase 2 — MXFP4 acts + FP8 combine together
    ]
    for af, cd, tol in configs:
        y = run_mega(x, tl1, tl2, topk_idx, topk_weights, group, E, H, I, K,
                     act_format=af, combine_dtype=cd)
        d_base = calc_diff(y.float(), y_base.float())
        d_ref = calc_diff(y.float(), ref.float())
        print(f'[feat]  act={af:5s} combine={cd:8s} vs base={d_base:.4f}  vs ref={d_ref:.4f}')
        assert d_ref < tol, f'{af}/{cd}: vs reference {d_ref:.4f} exceeds {tol}'

    print('PASS')


if __name__ == '__main__':
    main()
