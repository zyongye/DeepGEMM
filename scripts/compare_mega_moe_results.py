"""Compare JSON outputs produced by tests/bench_mega_moe_ab.py."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare two MegaMoE benchmark result files')
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    args = parser.parse_args()

    baseline = {row['tokens_per_rank']: row for row in json.loads(Path(args.baseline).read_text())}
    candidate = {row['tokens_per_rank']: row for row in json.loads(Path(args.candidate).read_text())}
    assert baseline.keys() == candidate.keys(), 'Token cases differ between result files'

    first_token = min(baseline)
    old_first, new_first = baseline[first_token], candidate[first_token]
    print(
        f'{old_first["label"]} ({old_first["revision"][:12]}): '
        f'{old_first["buffer_gib"]:.3f} GiB symmetric buffer')
    print(
        f'{new_first["label"]} ({new_first["revision"][:12]}): '
        f'{new_first["buffer_gib"]:.3f} GiB symmetric buffer')
    print()

    print('| tokens/rank | baseline (us) | candidate (us) | speedup | expert max/mean |')
    print('|---:|---:|---:|---:|---:|')
    for tokens in sorted(baseline):
        old, new = baseline[tokens], candidate[tokens]
        for field in ('num_ranks', 'capacity_requested', 'capacity_aligned',
                      'hidden', 'intermediate_hidden', 'num_experts',
                      'num_topk', 'act_format', 'combine_dtype', 'routing_skew'):
            assert old[field] == new[field], f'{field} differs for {tokens} tokens/rank'
        speedup = old['latency_us'] / new['latency_us']
        print(f'| {tokens} | {old["latency_us"]:.2f} | {new["latency_us"]:.2f} | '
              f'{speedup:.3f}x | {new["expert_max_over_mean"]:.2f}x |')


if __name__ == '__main__':
    main()
