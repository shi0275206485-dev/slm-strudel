"""
anova_rank.py

One-way analysis of variance of a metric across LoRA ranks, with the
training seed as the replicate (thesis Section 6.4).

The unit of analysis is one trained adapter: for each (rank, train_seed)
the metric is first averaged over the generation seeds (Stage 1 of
aggregate_seed_stability.py, same weights for sound_sim and final_score).
The resulting per-adapter values -- three per rank -- are then compared
across ranks with scipy.stats.f_oneway.

Two input modes:

  1) From the per-file scores written by run_batch_metrics.py
       python anova_rank.py --per_file per_file_scores.jsonl [--metric final_score]

  2) From summary statistics only (mean, sample SD, n per rank), e.g. the
     values printed in a results table. One-way ANOVA needs nothing else,
     so this reproduces mode 1 up to the rounding of the inputs.
       python anova_rank.py --summary "16:0.7726,0.0140" "64:0.7822,0.0200" ... --n 3
"""

import argparse
import json
import math
from collections import defaultdict

from scipy import stats

from aggregate_seed_stability import SUBMETRICS, compute_composites, mean, parse_rank_seed


def per_adapter_values(per_file, metric, shot=0):
    groups = defaultdict(list)
    with open(per_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if shot is not None and int(row.get("shot", 0)) != shot:
                continue
            rank, train_seed = parse_rank_seed(row)
            if rank is None or train_seed is None:
                continue
            groups[(rank, train_seed)].append(row)
    by_rank = defaultdict(dict)
    for (rank, train_seed), rows in groups.items():
        avg = compute_composites({m: mean([r[m] for r in rows]) for m in SUBMETRICS})
        by_rank[rank][train_seed] = avg[metric]
    return {r: [v for _, v in sorted(seeds.items())] for r, seeds in sorted(by_rank.items())}


def anova_from_summary(groups, n):
    """groups: list of (mean, sd). Standard one-way ANOVA identities."""
    k = len(groups)
    grand = sum(m for m, _ in groups) / k
    ss_between = sum(n * (m - grand) ** 2 for m, _ in groups)
    ss_within = sum((n - 1) * sd ** 2 for _, sd in groups)
    df1, df2 = k - 1, k * n - k
    F = (ss_between / df1) / (ss_within / df2)
    return F, df1, df2, stats.f.sf(F, df1, df2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_file", help="per_file_scores.jsonl from run_batch_metrics.py")
    ap.add_argument("--metric", default="final_score",
                    help="one of %s, sound_sim, final_score" % ", ".join(SUBMETRICS))
    ap.add_argument("--shot", type=int, default=0,
                    help="shot count K to analyse (per_file mode); the rank sweep uses K=0")
    ap.add_argument("--summary", nargs="+", metavar="RANK:MEAN,SD",
                    help="summary statistics per rank instead of --per_file")
    ap.add_argument("--n", type=int, default=3, help="replicates per rank (summary mode)")
    args = ap.parse_args()

    if args.per_file:
        vals = per_adapter_values(args.per_file, args.metric, args.shot)
        print(f"metric: {args.metric}, K={args.shot}")
        for r, v in vals.items():
            sd = math.sqrt(sum((x - mean(v)) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
            print(f"  r={r:<5} n={len(v)}  values={', '.join(f'{x:.4f}' for x in v)}  "
                  f"mean={mean(v):.4f}  sd={sd:.4f}")
        if len(vals) < 2:
            raise SystemExit("need at least two ranks in the input to compare")
        F, p = stats.f_oneway(*vals.values())
        k, N = len(vals), sum(len(v) for v in vals.values())
        print(f"one-way ANOVA: F({k - 1}, {N - k}) = {F:.3f}, p = {p:.4f}")
    elif args.summary:
        groups = []
        for item in args.summary:
            rank, ms = item.split(":")
            m, sd = (float(x) for x in ms.split(","))
            groups.append((m, sd))
        F, df1, df2, p = anova_from_summary(groups, args.n)
        print(f"one-way ANOVA from summary statistics (n={args.n} per rank): "
              f"F({df1}, {df2}) = {F:.3f}, p = {p:.4f}")
    else:
        ap.error("give --per_file or --summary")


if __name__ == "__main__":
    main()
