"""Benchmark the fused SM100 (Blackwell/GB200) Mega MoE kernel.

Times the full fused kernel (dispatch a2a + L1 GEMM + SwiGLU + L2 GEMM + combine) via
`bench_kineto`, for three compute configs:
  - fp8 acts x fp4 weights        (kind::mxf8f6f4, K=32)        -- baseline
  - mxfp4 acts                    (kind::mxf8f6f4, K=32)        -- DG_MEGA_MXF4_KIND=0
  - mxfp4 acts                    (kind::mxf4 dense, K=64)      -- DG_MEGA_MXF4_KIND=1 (default)

Single-GPU usage (SM100 required):
    PYTHONPATH=$PWD python tests/bench_mega_moe.py
    PYTHONPATH=$PWD python tests/bench_mega_moe.py --num-tokens 4096 --hidden 7168 \
        --intermediate 2048 --num-experts 8 --num-topk 8
"""
import argparse
import os
import torch

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist
from deep_gemm.testing import bench_kineto


def cast_weights_to_fp4(bf16_w):
    """(E, N, K) bf16 -> packed-FP4 (E, N, K/2) int8 + transformed UE8M0 SF."""
    E, N, Kd = bf16_w.shape
    w = torch.empty((E, N, Kd // 2), device='cuda', dtype=torch.int8)
    wsf = torch.empty((E, N, Kd // 32), device='cuda', dtype=torch.float)
    for e in range(E):
        w[e], wsf[e] = per_token_cast_to_fp4(bf16_w[e], use_ue8m0=True, gran_k=32)
    wsf = deep_gemm.transform_sf_into_required_layout(wsf, N, Kd, (1, 32), E)
    return w, wsf


def main():
    p = argparse.ArgumentParser(description='Mega MoE MXFP4 benchmark')
    p.add_argument('--num-tokens', type=int, default=512)
    p.add_argument('--hidden', type=int, default=2048)
    p.add_argument('--intermediate', type=int, default=1024)
    p.add_argument('--num-experts', type=int, default=8)
    p.add_argument('--num-topk', type=int, default=2)
    p.add_argument('--num-tests', type=int, default=60)
    p.add_argument('--clamp', type=float, default=None, help='activation clamp (default: none)')
    args = p.parse_args()

    rank, world, group = init_dist(0, 1)
    torch.manual_seed(0)
    T, H, I, E, K = args.num_tokens, args.hidden, args.intermediate, args.num_experts, args.num_topk
    assert H % 128 == 0 and I % 128 == 0 and E % world == 0, 'hidden/intermediate %128, experts %ranks'

    x  = torch.randn(T, H, dtype=torch.bfloat16, device='cuda') * (H ** -0.5)
    l1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device='cuda') * (H ** -0.5)
    l2 = torch.randn(E, H, I, dtype=torch.bfloat16, device='cuda') * (I ** -0.5)
    scores = torch.randn(T, E, dtype=torch.float, device='cuda')
    tw, ti = torch.topk(scores, K, dim=-1, largest=True, sorted=False)
    ti = ti.to(torch.int64)
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(cast_weights_to_fp4(l1), cast_weights_to_fp4(l2))

    num_routed = T * K
    flops = 2 * num_routed * (H * (2 * I)) + 2 * num_routed * (I * H)  # L1(gate+up) + L2

    def bench(label, act_format, mxf4_env):
        if mxf4_env is None:
            os.environ.pop('DG_MEGA_MXF4_KIND', None)
        else:
            os.environ['DG_MEGA_MXF4_KIND'] = mxf4_env
        buf = deep_gemm.get_symm_buffer_for_mega_moe(group, E, T, K, H, I, act_format=act_format)
        if act_format == 'mxfp4':
            xq, xsf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        else:
            xq, xsf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        y = torch.empty((T, H), dtype=torch.bfloat16, device='cuda')

        def run():
            buf.x[:T].copy_(xq); buf.x_sf[:T].copy_(xsf)
            buf.topk_idx[:T].copy_(ti); buf.topk_weights[:T].copy_(tw)
            deep_gemm.fp8_fp4_mega_moe(y=y, l1_weights=tl1, l2_weights=tl2, sym_buffer=buf,
                                       activation_clamp=args.clamp, fast_math=True)
        t = bench_kineto(run, 'mega_moe', num_tests=args.num_tests, suppress_kineto_output=True)
        print(f'{label:42s} {t*1e6:8.2f} us   {flops/t/1e12:7.1f} TFLOPS', flush=True)
        return t

    print(f'Config: T={T} H={H} I={I} E={E} K={K}  (3 matmuls, {flops/1e9:.1f} GFLOP)\n', flush=True)
    t_fp8  = bench('fp8 acts x fp4 wt  (mxf8f6f4 K=32)', 'fp8',   None)
    t_mxf8 = bench('mxfp4 acts         (mxf8f6f4 K=32)', 'mxfp4', '0')
    t_mxf4 = bench('mxfp4 acts         (mxf4-kind K=64)', 'mxfp4', '1')
    print(flush=True)
    print(f'mxf4-kind vs fp8xfp4:        {t_fp8 / t_mxf4:.3f}x', flush=True)
    print(f'mxf4-kind vs mxfp4-mxf8f6f4: {t_mxf8 / t_mxf4:.3f}x', flush=True)


if __name__ == '__main__':
    main()
