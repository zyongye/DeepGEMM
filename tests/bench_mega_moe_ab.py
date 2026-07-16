"""Revision-neutral distributed MegaMoE benchmark.

This file intentionally uses only the public API shared by ``e1e5123`` and the
PR-377 rewrite. Keep the benchmark script fixed and select the implementation
with ``PYTHONPATH`` so both revisions receive identical inputs::

    PYTHONPATH=/path/to/e1e5123 python /path/to/rewrite/tests/bench_mega_moe_ab.py \
        --label e1e5123 --output /tmp/e1e5123.json
    PYTHONPATH=/path/to/rewrite python /path/to/rewrite/tests/bench_mega_moe_ab.py \
        --label pr377-rewrite --output /tmp/pr377-rewrite.json

The default workload targets the low-concurrency regression: 32, 64, 128 and
256 tokens per rank, with optional rank-balanced local-expert skew.
"""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import bench_kineto
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist, uneven_all_gather


def _revision() -> str:
    try:
        root = Path(deep_gemm.__file__).resolve().parent.parent
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _cast_weights_to_fp4(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_groups, n, k = weights.shape
    quantized = torch.empty((num_groups, n, k // 2), dtype=torch.int8, device='cuda')
    scales = torch.empty((num_groups, n, k // 32), dtype=torch.float, device='cuda')
    for group_idx in range(num_groups):
        quantized[group_idx], scales[group_idx] = per_token_cast_to_fp4(
            weights[group_idx], use_ue8m0=True, gran_k=32)
    scales = deep_gemm.transform_sf_into_required_layout(
        scales, n, k, (1, 32), num_groups)
    return quantized, scales


def _make_routing(
        num_tokens: int,
        num_experts: int,
        num_experts_per_rank: int,
        num_topk: int,
        routing_skew: float,
        seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device='cuda')
    generator.manual_seed(seed)
    scores = torch.randn(
        (num_tokens, num_experts), dtype=torch.float, device='cuda', generator=generator)
    if routing_skew > 0:
        # Every rank receives the same local-index distribution. This isolates
        # expert imbalance from a trivial rank/NVLink traffic imbalance.
        local_expert_idx = torch.arange(num_experts, device='cuda') % num_experts_per_rank
        scores.sub_(routing_skew * torch.log1p(local_expert_idx.float()).unsqueeze(0))
    return torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)


def _routing_metrics(
        topk_idx: torch.Tensor,
        rank_idx: int,
        num_experts_per_rank: int,
        group: dist.ProcessGroup) -> Dict[str, float]:
    gathered = uneven_all_gather(topk_idx, group=group)
    first_expert = rank_idx * num_experts_per_rank
    local = gathered[(gathered >= first_expert) &
                     (gathered < first_expert + num_experts_per_rank)] - first_expert
    counts = torch.bincount(local, minlength=num_experts_per_rank)
    received = int(local.numel())
    active = int((counts > 0).sum().item())
    maximum = int(counts.max().item()) if counts.numel() else 0
    values = torch.tensor([received, active, maximum], dtype=torch.float64, device='cuda')
    all_values = [torch.empty_like(values) for _ in range(dist.get_world_size(group))]
    dist.all_gather(all_values, values, group=group)
    stacked = torch.stack(all_values).cpu()
    received_mean = float(stacked[:, 0].mean().item())
    imbalance = []
    for received_rank, _, maximum_rank in stacked.tolist():
        mean_expert_load = received_rank / num_experts_per_rank
        imbalance.append(maximum_rank / mean_expert_load if mean_expert_load else 0.0)
    return {
        'received_tokens_mean': received_mean,
        'received_tokens_min': int(stacked[:, 0].min().item()),
        'received_tokens_max': int(stacked[:, 0].max().item()),
        'active_experts_mean': float(stacked[:, 1].mean().item()),
        'expert_max_over_mean': max(imbalance),
    }


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert args.num_experts % num_ranks == 0
    num_experts_per_rank = args.num_experts // num_ranks
    max_tokens = max(args.tokens)
    assert max_tokens <= args.num_max_tokens_per_rank
    assert args.hidden % 128 == 0 and args.intermediate_hidden % 128 == 0

    torch.manual_seed(args.seed + rank_idx)
    os.environ['DG_MEGA_MXF4_KIND'] = str(int(args.mxf4_kind))
    combine_dtype = torch.float8_e4m3fn if args.combine_dtype == 'fp8' else torch.bfloat16

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        args.num_experts,
        args.num_max_tokens_per_rank,
        args.num_topk,
        args.hidden,
        args.intermediate_hidden,
        combine_dtype=combine_dtype,
        act_format=args.act_format)

    l1 = torch.randn(
        (num_experts_per_rank, args.intermediate_hidden * 2, args.hidden),
        dtype=torch.bfloat16, device='cuda')
    l1 = _cast_weights_to_fp4(l1)
    l2 = torch.randn(
        (num_experts_per_rank, args.hidden, args.intermediate_hidden),
        dtype=torch.bfloat16, device='cuda')
    l2 = _cast_weights_to_fp4(l2)
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)

    x = torch.randn((max_tokens, args.hidden), dtype=torch.bfloat16, device='cuda')
    if args.act_format == 'mxfp4':
        x_quantized, x_sf = per_token_cast_to_fp4(
            x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    else:
        x_quantized, x_sf = per_token_cast_to_fp8(
            x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    del x

    sync_value = torch.ones((), dtype=torch.int, device='cuda')
    results: List[Dict[str, object]] = []
    for num_tokens in args.tokens:
        topk_weights, topk_idx = _make_routing(
            num_tokens,
            args.num_experts,
            num_experts_per_rank,
            args.num_topk,
            args.routing_skew,
            args.seed * 1000003 + rank_idx * 1009 + num_tokens)
        routing = _routing_metrics(topk_idx, rank_idx, num_experts_per_rank, group)

        buffer.x[:num_tokens].copy_(x_quantized[:num_tokens])
        buffer.x_sf[:num_tokens].copy_(x_sf[:num_tokens])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        y = torch.empty((num_tokens, args.hidden), dtype=torch.bfloat16, device='cuda')

        def run() -> None:
            deep_gemm.fp8_fp4_mega_moe(
                y=y,
                l1_weights=l1,
                l2_weights=l2,
                sym_buffer=buffer,
                activation_clamp=args.activation_clamp,
                fast_math=bool(args.fast_math))

        for _ in range(args.warmups):
            run()
        torch.cuda.synchronize()

        elapsed_samples = []
        for _ in range(args.repeats):
            elapsed = bench_kineto(
                run,
                'mega_moe',
                num_tests=args.num_tests,
                suppress_kineto_output=True,
                flush_l2=False,
                barrier=lambda: dist.all_reduce(sync_value, group=group))
            elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device='cuda')
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX, group=group)
            elapsed_samples.append(float(elapsed_tensor.item()))

        elapsed_seconds = statistics.median(elapsed_samples)
        flops_per_rank = (
            2 * routing['received_tokens_mean'] * args.hidden * args.intermediate_hidden * 3)
        result = {
            'label': args.label,
            'revision': _revision(),
            'num_ranks': num_ranks,
            'tokens_per_rank': num_tokens,
            'capacity_requested': args.num_max_tokens_per_rank,
            'capacity_aligned': buffer.num_max_tokens_per_rank,
            'buffer_gib': buffer.buffer.nbytes / 2 ** 30,
            'hidden': args.hidden,
            'intermediate_hidden': args.intermediate_hidden,
            'num_experts': args.num_experts,
            'num_topk': args.num_topk,
            'act_format': args.act_format,
            'combine_dtype': args.combine_dtype,
            'mxf4_kind': bool(args.mxf4_kind),
            'routing_skew': args.routing_skew,
            'latency_us': elapsed_seconds * 1e6,
            'tflops_per_rank': flops_per_rank / elapsed_seconds / 1e12,
            **routing,
        }
        results.append(result)
        if rank_idx == 0:
            print('DG_BENCH_RESULT=' + json.dumps(result, sort_keys=True), flush=True)

    if rank_idx == 0 and args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')

    dist.barrier(group=group)
    buffer.destroy()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description='Cross-revision distributed MegaMoE A/B benchmark')
    parser.add_argument('--label', default='candidate')
    parser.add_argument('--output', default='')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--tokens', type=int, nargs='+', default=(32, 64, 128, 256))
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--intermediate-hidden', type=int, default=3072)
    parser.add_argument('--num-experts', type=int, default=384)
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--act-format', choices=('fp8', 'mxfp4'), default='mxfp4')
    parser.add_argument('--combine-dtype', choices=('bf16', 'fp8'), default='fp8')
    parser.add_argument('--mxf4-kind', type=int, choices=(0, 1), default=1)
    parser.add_argument('--routing-skew', type=float, default=0.0)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--fast-math', type=int, choices=(0, 1), default=1)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--num-tests', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        _worker, args=(args.num_processes, args),
        nprocs=args.num_processes)


if __name__ == '__main__':
    main()
