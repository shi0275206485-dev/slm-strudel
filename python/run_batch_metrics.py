"""
run_batch_metrics.py

Orchestrates ALL metrics -- Syntax, Execution, Text similarity, Mel
similarity (Pearson + SSIM), Onset F1, and Hap F1 -- for a whole directory
of candidate output files in one pass, instead of running each tool
separately and hand-combining the results. Built entirely on top of the
already-verified pieces:
  - epoch_syntax_check.mjs   (Node, @strudel/transpiler)
  - epoch_execution_check.py (Playwright, no audio)
  - epoch_audio_check.py     (Playwright, WITH audio -- reuses t9_audio_capture.py)
  - epoch_hap_check.mjs      (Node, queries Haps directly -- NO audio rendering,
                               ~2ms/item -- needs hap_query_bundle.mjs built
                               per hap_query_entry.mjs's docstring, sitting in
                               the same NODE_PROJECT_DIR)
  - mel_similarity_score.py  (Pearson + SSIM + Onset F1 + Hap F1, imported directly)

Meant for the few-shot/baseline matrix (e.g. test_set_150_new track), but
works on any directory of files matching the (?:finetuned_)?(baseline|ablation)_
<model>_<k>shot_seed<n>.jsonl naming convention already used across this
project (same regex as mel_similarity_score.py's extract_model_shot_seed).

Usage:
    python run_batch_metrics.py \
        --input_dir path/to/few_shot_outputs \
        --file_pattern "baseline_*_*.jsonl" \
        --ref_test_file test_set_150_new.jsonl \
        --ref_audio_manifest audio_capture/reference/test_set_150_new/reference_manifest.jsonl \
        --out_dir batch_results

Produces, in --out_dir:
  - per_file_scores.jsonl : one row per input file, all metrics
  - summary.csv           : mean +/- std across seeds, per (model, shot),
                             same convention as mel_similarity_score.py
  - <out_dir>/work/<file_stem>/ : raw manifests per input file (syntax,
    execution, audio, hap) kept for debugging -- NOT auto-deleted like the
    training callback's per-epoch audio, since this is a one-shot
    evaluation run you'll likely want to spot-check, not something that
    repeats dozens of times. Pass --cleanup_audio to delete the .wav files
    after scoring if disk space is a concern (hap manifests are tiny JSON,
    never cleaned up).

** BEFORE RUNNING: fill in NODE_PROJECT_DIR below (same value you already
set in training_callback.py), and make sure hap_query_bundle.mjs and
epoch_hap_check.mjs are both present there. **
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from difflib import SequenceMatcher

from mel_similarity_score import (
    load_manifest, get_resized_spec, cosine_sim, ssim_sim, is_degenerate, onset_f1, hap_f1,
    extract_model_shot_seed,
)

# ================= PATH CONFIG -- same rules as training_callback.py =================
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
NODE_PROJECT_DIR = os.environ.get("STRUDEL_NODE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "node"))  # override with env STRUDEL_NODE_DIR
NODE_BIN = "node"  # replace with absolute path if `node` isn't reliably on PATH

NODE_SCRIPT_PATH = os.path.join(NODE_PROJECT_DIR, "epoch_syntax_check.mjs")
HAP_SCRIPT_PATH = os.path.join(NODE_PROJECT_DIR, "epoch_hap_check.mjs")  # also needs hap_query_bundle.mjs
                                                                          # to be sitting next to it
EXEC_SCRIPT_PATH = os.path.join(SCRIPTS_DIR, "epoch_execution_check.py")
AUDIO_SCRIPT_PATH = os.path.join(SCRIPTS_DIR, "epoch_audio_check.py")
# =======================================================================================


def load_ref_codes(ref_test_file):
    """index -> reference/target code, for text-similarity comparison."""
    with open(ref_test_file, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return {i: row["messages"][-1]["content"] for i, row in enumerate(rows)}


def text_similarity(candidate_file, ref_codes):
    avg, _per_item = text_similarity_detailed(candidate_file, ref_codes)
    return avg


def text_similarity_detailed(candidate_file, ref_codes):
    """Same computation as before, but also returns
    {entry_index: {"text_similarity": float, "gen_code": str, "ref_code": str}}.
    The returned average is bit-for-bit what text_similarity() used to return."""
    with open(candidate_file, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    scores = []
    per_item = {}
    for idx, row in enumerate(rows):
        cand_code = row["messages"][-1]["content"]
        ref_code = ref_codes.get(idx)
        if ref_code is not None:
            s = SequenceMatcher(None, cand_code, ref_code).ratio()
            scores.append(s)
            per_item[idx] = {"text_similarity": s, "gen_code": cand_code, "ref_code": ref_code}
        else:
            per_item[idx] = {"text_similarity": None, "gen_code": cand_code, "ref_code": None}
    return (sum(scores) / len(scores) if scores else 0.0), per_item


PASS_STATUSES = {"SUCCESS", "OK", "PASS", "VALID", "SYNTAX_OK"}


def load_pass_manifest(manifest_path):
    """entry_index -> 1.0/0.0 for the syntax and execution manifests.

    The two checkers report different pass tokens -- epoch_syntax_check.mjs
    emits SYNTAX_OK / SYNTAX_ERROR while epoch_execution_check.py emits
    SUCCESS / SYNTAX_ERROR -- so both are accepted here, along with any
    *_OK variant. Never raises: a missing or unreadable manifest yields {}
    and the per-item column comes out as None rather than a wrong 0."""
    if not os.path.exists(manifest_path):
        return {}
    out = {}
    try:
        entries = load_manifest(manifest_path)
    except Exception:
        return {}
    for e in entries:
        if not isinstance(e, dict) or "entry_index" not in e:
            continue
        ok = None
        if "status" in e:
            st = str(e["status"]).upper()
            ok = st in PASS_STATUSES or st.endswith("_OK")
        else:
            for key in ("valid", "passed", "pass", "is_valid", "success"):
                if key in e:
                    ok = bool(e[key])
                    break
        if ok is not None:
            out[e["entry_index"]] = 1.0 if ok else 0.0
    return out


def run_syntax_check(input_file, manifest_path):
    result = subprocess.run(
        [NODE_BIN, NODE_SCRIPT_PATH, input_file, manifest_path],
        capture_output=True, text=True, cwd=NODE_PROJECT_DIR,
    )
    for line in result.stdout.splitlines():
        if line.startswith("SYNTAX_PASS_RATE="):
            return float(line.split("=", 1)[1])
    print(f"[run_batch_metrics] WARNING: syntax check failed for {input_file}\n"
          f"stdout: {result.stdout}\nstderr: {result.stderr}")
    return 0.0


def run_execution_check(input_file, manifest_path):
    result = subprocess.run(
        [sys.executable, EXEC_SCRIPT_PATH, "--input", input_file, "--output_manifest", manifest_path],
        capture_output=True, text=True, cwd=SCRIPTS_DIR,
    )
    for line in result.stdout.splitlines():
        if line.startswith("EXECUTION_PASS_RATE="):
            return float(line.split("=", 1)[1])
    print(f"[run_batch_metrics] WARNING: execution check failed for {input_file}\n"
          f"stdout: {result.stdout}\nstderr: {result.stderr}")
    return 0.0


def run_audio_check(input_file, audio_dir, manifest_path):
    result = subprocess.run(
        [sys.executable, AUDIO_SCRIPT_PATH, "--input", input_file,
         "--out_dir", audio_dir, "--out_manifest", manifest_path],
        capture_output=True, text=True, cwd=SCRIPTS_DIR,
    )
    if "AUDIO_CAPTURE_DONE=" not in result.stdout:
        print(f"[run_batch_metrics] WARNING: audio check failed for {input_file}\n"
              f"stdout: {result.stdout}\nstderr: {result.stderr}")


def run_hap_check(input_file, manifest_path):
    """Queries Haps directly from the code (no audio rendering) -- ~2ms/item,
    so this is cheap even for large batches. Requires hap_query_bundle.mjs
    (built from hap_query_entry.mjs) to be sitting next to epoch_hap_check.mjs
    in NODE_PROJECT_DIR."""
    result = subprocess.run(
        [NODE_BIN, HAP_SCRIPT_PATH, input_file, manifest_path],
        capture_output=True, text=True, cwd=NODE_PROJECT_DIR,
    )
    if "HAP_QUERY_DONE=" not in result.stdout:
        print(f"[run_batch_metrics] WARNING: hap check failed for {input_file}\n"
              f"stdout: {result.stdout}\nstderr: {result.stderr}")


def build_reference_haps(ref_test_file, work_dir):
    """Writes the reference codes out as a completions-shaped JSONL, queries
    their Haps ONCE (cheap, ~2ms/item), and returns {entry_index: haps}.
    Reuses the exact same ref_test_file already used for text-similarity
    reference codes, so no separate reference file is needed."""
    with open(ref_test_file, "r", encoding="utf-8") as f:
        ref_rows = [json.loads(l) for l in f if l.strip()]

    ref_completions_path = os.path.join(work_dir, "_reference_for_hap.jsonl")
    with open(ref_completions_path, "w", encoding="utf-8") as f:
        for row in ref_rows:
            f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")

    ref_manifest_path = os.path.join(work_dir, "_reference_hap_manifest.jsonl")
    run_hap_check(ref_completions_path, ref_manifest_path)
    ref_entries = load_manifest(ref_manifest_path)

    ref_haps_by_index = {
        e["entry_index"]: e["haps"] for e in ref_entries
        if e["status"] == "SUCCESS" and e.get("haps")
    }
    print(f"[run_batch_metrics] built {len(ref_haps_by_index)} reference Hap lists from {ref_test_file}")
    return ref_haps_by_index


def score_hap(manifest_path, ref_haps_by_index):
    avg, _per_item = score_hap_detailed(manifest_path, ref_haps_by_index)
    return avg


def score_hap_detailed(manifest_path, ref_haps_by_index):
    """Same computation as before, plus {entry_index: hap_f1}."""
    entries = load_manifest(manifest_path)
    scores = []
    per_item = {}
    for e in entries:
        idx = e["entry_index"]
        if e["status"] != "SUCCESS" or not e.get("haps") or idx not in ref_haps_by_index:
            scores.append(0.0)
            per_item[idx] = 0.0
            continue
        try:
            s = hap_f1(e["haps"], ref_haps_by_index[idx])
        except Exception:
            s = 0.0
        scores.append(s)
        per_item[idx] = s
    return (sum(scores) / len(scores) if scores else 0.0), per_item


def score_audio_entries(entries, ref_by_index):
    """Core scoring logic, operating on an already-loaded list of manifest
    entries (not a path) -- shared by both the fresh-capture path and the
    existing-manifest path. Returns (pearson_avg, ssim_avg, onset_avg)."""
    pearson_scores, ssim_scores, onset_scores = [], [], []
    per_item = {}
    for e in entries:
        idx = e["entry_index"]
        if e["status"] != "SUCCESS" or not e.get("wav_path") or idx not in ref_by_index:
            pearson_scores.append(0.0)
            ssim_scores.append(0.0)
            onset_scores.append(0.0)
            per_item[idx] = {"mel_pearson": 0.0, "mel_ssim": 0.0, "onset_f1": 0.0,
                             "gen_wav_path": e.get("wav_path"), "ref_wav_path": ref_by_index.get(idx),
                             "audio_status": e.get("status")}
            continue
        try:
            cand_spec = get_resized_spec(e["wav_path"])
            ref_spec = get_resized_spec(ref_by_index[idx])
            if is_degenerate(cand_spec) or is_degenerate(ref_spec):
                p, s, o = 0.0, 0.0, 0.0
            else:
                p = max(0.0, cosine_sim(cand_spec, ref_spec))
                s = ssim_sim(cand_spec, ref_spec)
                o = onset_f1(e["wav_path"], ref_by_index[idx])
        except Exception:
            p, s, o = 0.0, 0.0, 0.0
        pearson_scores.append(p)
        ssim_scores.append(s)
        onset_scores.append(o)
        per_item[idx] = {"mel_pearson": p, "mel_ssim": s, "onset_f1": o,
                         "gen_wav_path": e.get("wav_path"), "ref_wav_path": ref_by_index.get(idx),
                         "audio_status": e.get("status")}
    pearson_avg = sum(pearson_scores) / len(pearson_scores) if pearson_scores else 0.0
    ssim_avg = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0.0
    onset_avg = sum(onset_scores) / len(onset_scores) if onset_scores else 0.0
    return pearson_avg, ssim_avg, onset_avg, per_item


def score_audio(audio_manifest_path, ref_by_index):
    return score_audio_entries(load_manifest(audio_manifest_path), ref_by_index)


def process_file(input_file, work_dir, ref_codes, ref_by_index, ref_haps_by_index, cleanup_audio,
                  existing_audio_by_file=None, per_item_writer=None, reuse_work_audio=False):
    fname = os.path.basename(input_file)
    stem = os.path.splitext(fname)[0]
    file_work_dir = os.path.join(work_dir, stem)
    os.makedirs(file_work_dir, exist_ok=True)

    syntax_manifest = os.path.join(file_work_dir, "syntax_manifest.jsonl")
    exec_manifest = os.path.join(file_work_dir, "execution_manifest.jsonl")
    hap_manifest = os.path.join(file_work_dir, "hap_manifest.jsonl")

    print(f"[run_batch_metrics] {fname}: text similarity...")
    text_sim, text_per_item = text_similarity_detailed(input_file, ref_codes)

    print(f"[run_batch_metrics] {fname}: syntax check (node)...")
    syntax_rate = run_syntax_check(input_file, syntax_manifest)

    print(f"[run_batch_metrics] {fname}: execution check (playwright)...")
    exec_rate = run_execution_check(input_file, exec_manifest)

    print(f"[run_batch_metrics] {fname}: hap query (node, no audio -- fast)...")
    run_hap_check(input_file, hap_manifest)
    hap_avg, hap_per_item = score_hap_detailed(hap_manifest, ref_haps_by_index)

    audio_dir = os.path.join(file_work_dir, "audio")
    audio_manifest = os.path.join(file_work_dir, "audio_manifest.jsonl")

    if existing_audio_by_file is not None:
        # Audio was already captured by a separate t9_audio_capture.py run --
        # use those entries directly, don't re-capture.
        entries = existing_audio_by_file.get(fname, [])
        if not entries:
            print(f"[run_batch_metrics] WARNING: no entries found for {fname} in the "
                  f"existing audio manifest -- check source_file naming matches exactly.")
        print(f"[run_batch_metrics] {fname}: scoring {len(entries)} pre-captured audio entries...")
        pearson_avg, ssim_avg, onset_avg, audio_per_item = score_audio_entries(entries, ref_by_index)
    elif reuse_work_audio and os.path.exists(audio_manifest):
        # This file's own audio_manifest.jsonl from a previous run of THIS script
        # is sitting in the work dir already -- rescore it in place rather than
        # spending hours re-rendering .wav files that have not changed.
        entries = load_manifest(audio_manifest)
        n_wav = sum(1 for e in entries if e.get("wav_path") and os.path.exists(e["wav_path"]))
        print(f"[run_batch_metrics] {fname}: reusing {len(entries)} entries from existing "
              f"audio_manifest.jsonl ({n_wav} .wav files still on disk)...")
        if entries and n_wav == 0:
            print(f"[run_batch_metrics] WARNING: none of the .wav paths in {audio_manifest} "
                  f"exist on disk -- mel/onset scores will all be 0. Drop --reuse_work_audio "
                  f"to re-capture, or check the paths.")
        pearson_avg, ssim_avg, onset_avg, audio_per_item = score_audio_entries(entries, ref_by_index)
    else:
        print(f"[run_batch_metrics] {fname}: audio capture + scoring (playwright, several minutes)...")
        run_audio_check(input_file, audio_dir, audio_manifest)
        pearson_avg, ssim_avg, onset_avg, audio_per_item = score_audio(audio_manifest, ref_by_index)
        if cleanup_audio:
            try:
                shutil.rmtree(audio_dir, ignore_errors=False)
            except Exception as e:
                print(f"[run_batch_metrics] WARNING: could not delete {audio_dir}: {e}")

    model, shot, seed = extract_model_shot_seed(fname)

    if per_item_writer is not None:
        syntax_by_index = load_pass_manifest(syntax_manifest)
        exec_by_index = load_pass_manifest(exec_manifest)
        all_indices = sorted(
            set(text_per_item) | set(hap_per_item) | set(audio_per_item)
            | set(syntax_by_index) | set(exec_by_index)
        )
        for idx in all_indices:
            t = text_per_item.get(idx, {})
            a = audio_per_item.get(idx, {})
            mel_p = a.get("mel_pearson")
            mel_s = a.get("mel_ssim")
            ons = a.get("onset_f1")
            hp = hap_per_item.get(idx)
            if None not in (mel_p, mel_s, ons, hp):
                sound_sim = 0.2 * mel_p + 0.1 * mel_s + 0.2 * ons + 0.5 * hp
            else:
                sound_sim = None
            per_item_writer.write(json.dumps({
                "source_file": fname, "model": model, "shot": shot, "seed": seed,
                "entry_index": idx,
                "syntax": syntax_by_index.get(idx),
                "execution": exec_by_index.get(idx),
                "text_similarity": t.get("text_similarity"),
                "mel_pearson": mel_p, "mel_ssim": mel_s,
                "onset_f1": ons, "hap_f1": hp,
                "sound_sim": sound_sim,
                "gen_wav_path": a.get("gen_wav_path"),
                "ref_wav_path": a.get("ref_wav_path"),
                "audio_status": a.get("audio_status"),
                "gen_code": t.get("gen_code"),
                "ref_code": t.get("ref_code"),
            }, ensure_ascii=False) + "\n")
        per_item_writer.flush()

    row = {
        "source_file": fname, "model": model, "shot": shot, "seed": seed,
        "syntax_pass_rate": syntax_rate, "execution_pass_rate": exec_rate,
        "text_similarity": text_sim, "mel_pearson": pearson_avg, "mel_ssim": ssim_avg,
        "onset_f1": onset_avg, "hap_f1": hap_avg,
    }
    print(f"[run_batch_metrics] {fname}: "
          f"syntax={syntax_rate:.4f} execution={exec_rate:.4f} text_sim={text_sim:.4f} "
          f"mel_pearson={pearson_avg:.4f} mel_ssim={ssim_avg:.4f} onset_f1={onset_avg:.4f} "
          f"hap_f1={hap_avg:.4f}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--file_pattern", default="*baseline_*.jsonl")
    parser.add_argument("--ref_test_file", required=False,
                         help="e.g. test_set_150_new.jsonl -- used for text-similarity reference codes",default="test_set_150_new.jsonl")
    parser.add_argument("--ref_audio_manifest", required=False,
                         help="reference_manifest.jsonl for the matching test set's rendered audio",default="audio_capture/reference/test_set_150_new/reference_manifest.jsonl")
    parser.add_argument("--out_dir", default="batch_results")
    parser.add_argument("--cleanup_audio", action="store_true",
                         help="delete each file's captured .wav files after scoring (off by default)")
    parser.add_argument("--reuse_work_audio", action="store_true",
                         help="if <out_dir>/work/<stem>/audio_manifest.jsonl already exists from a "
                              "previous run, rescore those .wav files instead of re-capturing audio. "
                              "Use this when re-running only to produce per-item scores -- it skips "
                              "the slow Playwright audio pass and leaves existing .wav files intact")
    parser.add_argument("--no_per_item", action="store_true",
                         help="skip writing per_item_scores.jsonl (per-item rows are written by default)")
    parser.add_argument("--existing_audio_manifest", default=None,
                         help="path to a capture_manifest.jsonl already produced by t9_audio_capture.py "
                              "(covering some or all of the input files) -- if given, audio capture is "
                              "SKIPPED entirely and these pre-captured entries are scored directly")
    args = parser.parse_args()

    if NODE_PROJECT_DIR.startswith("PUT_THE_DIRECTORY"):
        raise RuntimeError("Set NODE_PROJECT_DIR at the top of this script before running "
                            "(same value as in training_callback.py).")

    os.makedirs(args.out_dir, exist_ok=True)
    args.out_dir = os.path.abspath(args.out_dir)  # absolute -- avoids the same
                                                   # subprocess-cwd path mismatch
                                                   # already fixed in training_callback.py
    work_dir = os.path.join(args.out_dir, "work")
    os.makedirs(work_dir, exist_ok=True)

    ref_codes = load_ref_codes(args.ref_test_file)
    ref_entries = load_manifest(args.ref_audio_manifest)
    ref_by_index = {
        e["entry_index"]: e["wav_path"] for e in ref_entries
        if e["status"] == "SUCCESS" and e.get("wav_path")
    }
    print(f"[run_batch_metrics] loaded {len(ref_codes)} reference codes, "
          f"{len(ref_by_index)} reference audios")

    ref_haps_by_index = build_reference_haps(args.ref_test_file, work_dir)

    existing_audio_by_file = None
    if args.existing_audio_manifest:
        existing_entries = load_manifest(args.existing_audio_manifest)
        existing_audio_by_file = {}
        for e in existing_entries:
            existing_audio_by_file.setdefault(e["source_file"], []).append(e)
        print(f"[run_batch_metrics] loaded existing audio manifest {args.existing_audio_manifest}: "
              f"{len(existing_entries)} entries across {len(existing_audio_by_file)} source files "
              f"-- audio capture will be SKIPPED, using these directly.")

    input_files = sorted(glob.glob(os.path.join(args.input_dir, args.file_pattern)))
    if not input_files:
        print(f"No files matched {args.file_pattern} in {args.input_dir}")
        return
    print(f"[run_batch_metrics] found {len(input_files)} candidate files")

    per_file_rows = []
    per_file_path = os.path.join(args.out_dir, "per_file_scores.jsonl")
    per_item_path = os.path.join(args.out_dir, "per_item_scores.jsonl")
    if args.cleanup_audio and not args.no_per_item:
        print("[run_batch_metrics] NOTE: --cleanup_audio will delete the .wav files that "
              "per_item_scores.jsonl points at. The scores stay valid, but you won't be able "
              "to play back the examples afterwards.")
    per_item_f = None if args.no_per_item else open(per_item_path, "w", encoding="utf-8")
    try:
        with open(per_file_path, "w", encoding="utf-8") as f:
            for input_file in input_files:
                row = process_file(input_file, work_dir, ref_codes, ref_by_index, ref_haps_by_index,
                                    args.cleanup_audio, existing_audio_by_file=existing_audio_by_file,
                                    per_item_writer=per_item_f, reuse_work_audio=args.reuse_work_audio)
                per_file_rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        if per_item_f is not None:
            per_item_f.close()
    print(f"[run_batch_metrics] per-file scores written to {per_file_path}")
    if not args.no_per_item:
        print(f"[run_batch_metrics] per-item scores written to {per_item_path}")

    # aggregate: mean +/- std (ddof=1) across seeds, per (model, shot) --
    # same convention as mel_similarity_score.py
    import numpy as np
    from collections import defaultdict

    by_model_shot = defaultdict(lambda: defaultdict(list))
    metric_names = ["syntax_pass_rate", "execution_pass_rate", "text_similarity",
                     "mel_pearson", "mel_ssim", "onset_f1", "hap_f1"]
    for row in per_file_rows:
        key = (row["model"], row["shot"])
        for metric in metric_names:
            by_model_shot[key][metric].append(row[metric])

    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("model,shot,n_seeds,syntax_mean,syntax_std,execution_mean,execution_std,"
                "text_sim_mean,text_sim_std,mel_pearson_mean,mel_pearson_std,mel_ssim_mean,mel_ssim_std,"
                "onset_f1_mean,onset_f1_std,hap_f1_mean,hap_f1_std\n")
        for (model, shot), metrics in sorted(by_model_shot.items(), key=lambda x: (x[0][0], x[0][1] if x[0][1] is not None else -1)):
            n = len(metrics["syntax_pass_rate"])
            vals = []
            for metric in metric_names:
                arr = metrics[metric]
                mean = np.mean(arr)
                std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
                vals.extend([f"{mean:.4f}", f"{std:.4f}"])
            f.write(f"{model},{shot},{n}," + ",".join(vals) + "\n")
    print(f"[run_batch_metrics] summary written to {summary_path}")


if __name__ == "__main__":
    main()
