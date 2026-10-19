"""
diagnose_pipeline_leakage.py

Quantifies how much of the sound_sim signal is lost to evaluation-pipeline
failures rather than to genuine model error.

Motivation: on baseline_qwen2_5-coder-0_5b-instruct_0shot_seed10, 13/150
candidates executed successfully in the browser, but only 1/13 produced a
queryable Hap list and only 4/13 produced a .wav. If that ratio holds on
high-execution (fine-tuned) files, then hap_f1 / mel_* / onset_f1 are being
systematically depressed by tooling, not by the model -- which would make
sound_sim a lower bound on capability rather than a measurement of it.

For every <out_dir>/work/<file_stem>/ directory this reports, per file:
  exec_ok            : candidates that ran in the browser
  hap_ok  / hap_loss : of those, how many yielded Haps (and how many did not)
  wav_ok  / wav_loss : of those, how many yielded audio
plus a breakdown of WHY the hap and audio steps failed, aggregated across
all files so systematic causes stand out from one-off errors.

Usage:
    python diagnose_pipeline_leakage.py --results_dir batch_results
    python diagnose_pipeline_leakage.py --results_dir batch_results --csv leakage.csv

Read-only: opens manifests, writes nothing except the optional --csv.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

SYNTAX_PASS = {"SYNTAX_OK", "SUCCESS", "OK"}
EXEC_PASS = {"SUCCESS", "OK"}


def load_manifest(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def index_by_entry(rows):
    return {r["entry_index"]: r for r in rows if isinstance(r, dict) and "entry_index" in r}


def shorten(msg, n=60):
    """Collapse an error message to a comparable signature: drop line/column
    numbers and quoted identifiers so 'Unexpected token (1:5)' and
    'Unexpected token (61:4)' group together."""
    if not msg:
        return "(no message)"
    msg = str(msg).split("\n")[0]
    msg = re.sub(r"\(\d+:\d+\)", "(L:C)", msg)
    msg = re.sub(r"\b\d+\b", "N", msg)
    return msg[:n]


def analyse_dir(work_subdir):
    syn = index_by_entry(load_manifest(os.path.join(work_subdir, "syntax_manifest.jsonl")))
    exe = index_by_entry(load_manifest(os.path.join(work_subdir, "execution_manifest.jsonl")))
    hap = index_by_entry(load_manifest(os.path.join(work_subdir, "hap_manifest.jsonl")))
    aud = index_by_entry(load_manifest(os.path.join(work_subdir, "audio_manifest.jsonl")))

    if not exe:
        return None

    n = len(exe)
    syn_ok = {i for i, r in syn.items() if str(r.get("status", "")).upper() in SYNTAX_PASS}
    exec_ok = {i for i, r in exe.items() if str(r.get("status", "")).upper() in EXEC_PASS}
    hap_ok = {i for i, r in hap.items()
              if str(r.get("status", "")).upper() == "SUCCESS" and r.get("haps")}
    wav_ok = {i for i, r in aud.items()
              if str(r.get("status", "")).upper() == "SUCCESS" and r.get("wav_path")}

    hap_lost = exec_ok - hap_ok
    wav_lost = exec_ok - wav_ok

    hap_reasons = Counter(shorten(hap.get(i, {}).get("message")) for i in hap_lost)
    wav_reasons = Counter(str(aud.get(i, {}).get("status", "MISSING")) for i in wav_lost)

    # candidates the two front-end checkers disagree about, in both directions
    exec_not_syn = sorted(exec_ok - syn_ok)
    syn_not_exec_n = len(syn_ok - exec_ok)

    return {
        "n": n,
        "syntax_ok": len(syn_ok),
        "exec_ok": len(exec_ok),
        "hap_ok": len(hap_ok),
        "wav_ok": len(wav_ok),
        "hap_lost": len(hap_lost),
        "wav_lost": len(wav_lost),
        "hap_keep_rate": len(hap_ok) / len(exec_ok) if exec_ok else None,
        "wav_keep_rate": len(wav_ok) / len(exec_ok) if exec_ok else None,
        "exec_but_not_syntax": exec_not_syn,
        "syntax_but_not_exec": syn_not_exec_n,
        "hap_reasons": hap_reasons,
        "wav_reasons": wav_reasons,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="the --out_dir you passed to run_batch_metrics.py (contains work/)")
    ap.add_argument("--csv", default=None, help="optional path to write the per-file table")
    args = ap.parse_args()

    work_root = os.path.join(args.results_dir, "work")
    if not os.path.isdir(work_root):
        work_root = args.results_dir
        print(f"[diagnose] no work/ subdir, scanning {work_root} directly")

    subdirs = sorted(
        d for d in (os.path.join(work_root, x) for x in os.listdir(work_root))
        if os.path.isdir(d)
    )
    if not subdirs:
        print(f"[diagnose] no subdirectories under {work_root}")
        return

    rows = []
    all_hap_reasons = Counter()
    all_wav_reasons = Counter()
    disagreements = []

    for d in subdirs:
        res = analyse_dir(d)
        if res is None:
            continue
        stem = os.path.basename(d)
        rows.append((stem, res))
        all_hap_reasons.update(res["hap_reasons"])
        all_wav_reasons.update(res["wav_reasons"])
        if res["exec_but_not_syntax"]:
            disagreements.append((stem, res["exec_but_not_syntax"]))

    if not rows:
        print("[diagnose] found no readable manifests")
        return

    print()
    print("PER-FILE LEAKAGE  (of the candidates that executed, how many survived to each metric)")
    print("=" * 104)
    print(f"{'file':<52}{'exec':>6}{'hap':>6}{'keep':>8}{'wav':>6}{'keep':>8}")
    print("-" * 104)
    for stem, r in rows:
        hk = f"{100*r['hap_keep_rate']:.0f}%" if r["hap_keep_rate"] is not None else "  -"
        wk = f"{100*r['wav_keep_rate']:.0f}%" if r["wav_keep_rate"] is not None else "  -"
        print(f"{stem[:50]:<52}{r['exec_ok']:>6}{r['hap_ok']:>6}{hk:>8}{r['wav_ok']:>6}{wk:>8}")

    tot_exec = sum(r["exec_ok"] for _, r in rows)
    tot_hap = sum(r["hap_ok"] for _, r in rows)
    tot_wav = sum(r["wav_ok"] for _, r in rows)
    print("-" * 104)
    print(f"{'TOTAL':<52}{tot_exec:>6}{tot_hap:>6}"
          f"{(f'{100*tot_hap/tot_exec:.0f}%' if tot_exec else '  -'):>8}{tot_wav:>6}"
          f"{(f'{100*tot_wav/tot_exec:.0f}%' if tot_exec else '  -'):>8}")

    print()
    print("WHY HAP QUERIES FAILED ON EXECUTABLE CODE")
    print("=" * 104)
    if all_hap_reasons:
        for msg, c in all_hap_reasons.most_common(15):
            print(f"{c:>6}  {msg}")
    else:
        print("  (none -- every executable candidate yielded Haps)")

    print()
    print("WHY AUDIO CAPTURE FAILED ON EXECUTABLE CODE")
    print("=" * 104)
    if all_wav_reasons:
        for msg, c in all_wav_reasons.most_common(15):
            print(f"{c:>6}  {msg}")
    else:
        print("  (none -- every executable candidate yielded audio)")

    if disagreements:
        print()
        print("CHECKER DISAGREEMENT: parsed as SYNTAX_ERROR yet executed successfully")
        print("=" * 104)
        for stem, idxs in disagreements[:20]:
            print(f"  {stem[:60]}: entries {idxs}")
        print("  -> the transpiler check and the browser check are not applying the same")
        print("     preprocessing; worth reconciling before quoting a syntax pass rate.")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["file", "n", "syntax_ok", "exec_ok", "hap_ok", "wav_ok",
                        "hap_keep_rate", "wav_keep_rate", "syntax_but_not_exec",
                        "exec_but_not_syntax_n"])
            for stem, r in rows:
                w.writerow([stem, r["n"], r["syntax_ok"], r["exec_ok"], r["hap_ok"], r["wav_ok"],
                            f"{r['hap_keep_rate']:.4f}" if r["hap_keep_rate"] is not None else "",
                            f"{r['wav_keep_rate']:.4f}" if r["wav_keep_rate"] is not None else "",
                            r["syntax_but_not_exec"], len(r["exec_but_not_syntax"])])
        print(f"\n[diagnose] per-file table written to {args.csv}")


if __name__ == "__main__":
    main()
