"""
aggregate_seed_stability.py

Second-stage aggregation for the training-seed stability experiment.

Context: run_batch_metrics.py already produces per_file_scores.jsonl with one
row per candidate file. For the seed-stability sweep, each candidate file is
one (rank, train_seed, generation_seed) combination. This script does the two
nested aggregations that run_batch_metrics.py's own summary.csv does NOT do:

  Stage 1: group by (rank, train_seed), average over the 5 GENERATION seeds
           -> one mean row per trained model (this is what summary.csv gives,
              but here we also compute sound_sim / final_score per group).
  Stage 2: group by rank, aggregate the Stage-1 per-train-seed means across
           the 3 TRAINING seeds -> mean, std, and coefficient of variation.
           The Stage-2 std is the "is this rank's result reproducible across
           retraining, or just one seed's luck?" number.

It reads rank / train_seed from the JSONL rows' own fields if present (the
generation notebook writes "rank" and "train_seed" into each row), and falls
back to parsing them out of source_file if not.

Usage:
    python aggregate_seed_stability.py --per_file per_file_scores.jsonl --out_csv seed_stability.csv
"""

import argparse
import json
import re
import math
from collections import defaultdict


SUBMETRICS = ["syntax_pass_rate", "execution_pass_rate", "text_similarity",
              "mel_pearson", "mel_ssim", "onset_f1", "hap_f1"]


def parse_rank_seed(row):
    """Prefer explicit fields; fall back to filename parsing."""
    rank = row.get("rank")
    train_seed = row.get("train_seed")
    if rank is not None and train_seed is not None:
        return int(rank), int(train_seed)

    # fall back: pull _r<rank>_ and _trainseed<n>_ out of source_file / model
    src = row.get("source_file", "") or str(row.get("model", ""))
    r_m = re.search(r"_r(\d+)_", src)
    s_m = re.search(r"trainseed(\d+)", src)
    rank = int(r_m.group(1)) if r_m else None
    train_seed = int(s_m.group(1)) if s_m else None
    return rank, train_seed


def compute_composites(d):
    """Add sound_sim and final_score to a dict of the 7 sub-metrics."""
    d["sound_sim"] = (0.2 * d["mel_pearson"] + 0.1 * d["mel_ssim"]
                      + 0.2 * d["onset_f1"] + 0.5 * d["hap_f1"])
    d["final_score"] = (0.1 * d["syntax_pass_rate"] + 0.5 * d["execution_pass_rate"]
                        + 0.1 * d["text_similarity"] + 0.3 * d["sound_sim"])
    return d


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std_sample(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_file", required=True, help="per_file_scores.jsonl from run_batch_metrics.py")
    ap.add_argument("--out_csv", default="seed_stability.csv")
    args = ap.parse_args()

    with open(args.per_file, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    # ---- Stage 1: group by (rank, train_seed), average over generation seeds ----
    by_rank_seed = defaultdict(list)
    for row in rows:
        rank, train_seed = parse_rank_seed(row)
        if rank is None or train_seed is None:
            print(f"[warn] could not parse rank/train_seed from a row (source_file="
                  f"{row.get('source_file')!r}); skipping it.")
            continue
        by_rank_seed[(rank, train_seed)].append(row)

    stage1 = {}  # (rank, train_seed) -> dict of averaged metrics (+ composites)
    for (rank, train_seed), grp in sorted(by_rank_seed.items()):
        avg = {m: mean([r[m] for r in grp]) for m in SUBMETRICS}
        avg = compute_composites(avg)
        avg["n_generation_seeds"] = len(grp)
        stage1[(rank, train_seed)] = avg

    # ---- Stage 2: for each rank, aggregate across training seeds ----
    by_rank = defaultdict(dict)
    for (rank, train_seed), avg in stage1.items():
        by_rank[rank][train_seed] = avg

    report_metrics = SUBMETRICS + ["sound_sim", "final_score"]

    with open(args.out_csv, "w", encoding="utf-8") as f:
        header = (["rank", "n_train_seeds"]
                  + [f"{m}_mean" for m in report_metrics]
                  + [f"{m}_std" for m in report_metrics]
                  + [f"{m}_cv_pct" for m in report_metrics])
        f.write(",".join(header) + "\n")

        for rank in sorted(by_rank):
            seed_means = by_rank[rank]
            n_ts = len(seed_means)
            means, stds, cvs = [], [], []
            for m in report_metrics:
                vals = [seed_means[ts][m] for ts in sorted(seed_means)]
                mu = mean(vals)
                sd = std_sample(vals)
                cv = (sd / mu * 100) if mu else 0.0
                means.append(mu); stds.append(sd); cvs.append(cv)
            row = ([str(rank), str(n_ts)]
                   + [f"{v:.4f}" for v in means]
                   + [f"{v:.4f}" for v in stds]
                   + [f"{v:.2f}" for v in cvs])
            f.write(",".join(row) + "\n")

    # ---- also print a readable stability summary to stdout ----
    print("\n=== Per (rank, train_seed) means (Stage 1) ===")
    print(f"{'rank':<6}{'train_seed':<12}{'n_gen':<7}{'execution':<11}{'sound_sim':<11}{'final_score':<12}")
    for (rank, train_seed) in sorted(stage1):
        a = stage1[(rank, train_seed)]
        print(f"{rank:<6}{train_seed:<12}{a['n_generation_seeds']:<7}"
              f"{a['execution_pass_rate']:<11.4f}{a['sound_sim']:<11.4f}{a['final_score']:<12.4f}")

    print("\n=== Cross-training-seed stability (Stage 2) ===")
    print(f"{'rank':<6}{'final_mean':<12}{'final_std':<12}{'final_CV%':<11}{'exec_std':<11}{'verdict'}")
    for rank in sorted(by_rank):
        seed_means = by_rank[rank]
        fvals = [seed_means[ts]["final_score"] for ts in sorted(seed_means)]
        evals = [seed_means[ts]["execution_pass_rate"] for ts in sorted(seed_means)]
        fmu, fsd = mean(fvals), std_sample(fvals)
        fcv = (fsd / fmu * 100) if fmu else 0.0
        esd = std_sample(evals)
        verdict = ("stable" if fcv < 3 else "moderate spread" if fcv < 6 else "HIGH spread — likely seed-driven")
        print(f"{rank:<6}{fmu:<12.4f}{fsd:<12.4f}{fcv:<11.2f}{esd:<11.4f}{verdict}")

    print(f"\nWritten to {args.out_csv}")


if __name__ == "__main__":
    main()
