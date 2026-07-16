"""Single-rank correctness test for MegaMoE activation/combine formats.

This test does not depend on DeepEP or TileLang.  It compares the shipped
FP8-activation/BF16-combine path and the new MXFP4/FP8 paths against a direct
float32 PyTorch MoE implementation.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist


def moe_reference_routed(
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor) -> torch.Tensor:
    """Evaluate the routed SwiGLU MoE directly in float32."""
    num_tokens, hidden = x.shape
    intermediate = w1.shape[1] // 2
    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)
    x_float = x.float()
    for expert_idx in range(w1.shape[0]):
        mask = topk_idx == expert_idx
        selected = mask.any(dim=1)
        if not bool(selected.any()):
            continue
        token_idx = selected.nonzero().flatten()
        weights = (topk_weights.float() * mask.float()).sum(dim=1)[token_idx]
        fc1 = x_float[token_idx] @ w1[expert_idx].float().T
        gate, up = fc1[:, :intermediate], fc1[:, intermediate:]
        fc2 = (F.silu(gate) * up) @ w2[expert_idx].float().T
        out[token_idx] += weights.unsqueeze(1) * fc2
    return out.to(torch.bfloat16)


def cast_weights_to_fp4(weights: torch.Tensor):
    num_groups, n, k = weights.shape
    quantized = torch.empty(
        (num_groups, n, k // 2), dtype=torch.int8, device='cuda')
    scales = torch.empty(
        (num_groups, n, k // 32), dtype=torch.float, device='cuda')
    for group_idx in range(num_groups):
        quantized[group_idx], scales[group_idx] = per_token_cast_to_fp4(
            weights[group_idx], use_ue8m0=True, gran_k=32)
    scales = deep_gemm.transform_sf_into_required_layout(
        scales, n, k, (1, 32), num_groups)
    return quantized, scales


def run_mega(
        x_bf16: torch.Tensor,
        transformed_l1,
        transformed_l2,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        group: dist.ProcessGroup,
        num_experts: int,
        hidden: int,
        intermediate: int,
        num_topk: int,
        act_format: str = 'fp8',
        combine_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    num_tokens = x_bf16.shape[0]
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_tokens, num_topk, hidden, intermediate,
        act_format=act_format, combine_dtype=combine_dtype)
    if act_format == 'mxfp4':
        x_quantized, x_sf = per_token_cast_to_fp4(
            x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    else:
        x_quantized, x_sf = per_token_cast_to_fp8(
            x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    buffer.x[:num_tokens].copy_(x_quantized)
    buffer.x_sf[:num_tokens].copy_(x_sf)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)

    y = torch.empty(
        (num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_fp4_mega_moe(
        y=y,
        l1_weights=transformed_l1,
        l2_weights=transformed_l2,
        sym_buffer=buffer,
        fast_math=True)
    y = y.clone()
    buffer.destroy()
    return y


def main() -> None:
    _, _, group = init_dist(0, 1)
    torch.manual_seed(0)

    num_tokens, hidden, intermediate = 512, 2048, 1024
    num_experts, num_topk = 8, 2
    x = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device='cuda') * hidden ** -0.5
    l1 = torch.randn(
        num_experts, 2 * intermediate, hidden,
        dtype=torch.bfloat16, device='cuda') * hidden ** -0.5
    l2 = torch.randn(
        num_experts, hidden, intermediate,
        dtype=torch.bfloat16, device='cuda') * intermediate ** -0.5
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False)

    reference = moe_reference_routed(
        x, l1, l2, topk_idx, topk_weights)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
        cast_weights_to_fp4(l1), cast_weights_to_fp4(l2))

    baseline = run_mega(
        x, transformed_l1, transformed_l2, topk_idx, topk_weights,
        group, num_experts, hidden, intermediate, num_topk)
    baseline_diff = calc_diff(baseline.float(), reference.float())
    print(f'FP8 acts / BF16 combine: diff={baseline_diff:.4f}')
    assert baseline_diff < 0.30

    configs = (
        ('fp8', torch.float8_e4m3fn, 0.05),
        ('mxfp4', torch.bfloat16, 0.10),
        ('mxfp4', torch.float8_e4m3fn, 0.10),
    )
    for act_format, combine_dtype, tolerance in configs:
        result = run_mega(
            x, transformed_l1, transformed_l2, topk_idx, topk_weights,
            group, num_experts, hidden, intermediate, num_topk,
            act_format=act_format, combine_dtype=combine_dtype)
        baseline_delta = calc_diff(result.float(), baseline.float())
        reference_diff = calc_diff(result.float(), reference.float())
        print(
            f'{act_format} acts / {combine_dtype}: '
            f'vs baseline={baseline_delta:.4f}, vs reference={reference_diff:.4f}')
        assert reference_diff < tolerance

    dist.destroy_process_group()
    print('PASS')


if __name__ == '__main__':
    main()
