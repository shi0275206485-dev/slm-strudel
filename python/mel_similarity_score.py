"""
mel_similarity_score.py

Consumes:
  - a reference manifest (from running t9_audio_capture.py's capture pipeline
    ONCE over the 150 frozen test-set target/reference codes)
  - one or more candidate manifests (from running it over model outputs)

Produces:
  - per-entry scores (JSONL)
  - mean +/- std (ddof=1) aggregated per (model, shot), matching the
    reporting convention used throughout this project

ALIGNMENT ASSUMPTION (please confirm)
--------------------------------------
Reference audio is matched to a candidate entry by `entry_index`, i.e. the
Nth line of a candidate output file is assumed to correspond to the Nth line
of the frozen 150-item test set (same order the reference manifest was built
from). If your pipeline doesn't guarantee this ordering, tell me and I'll
switch to matching on an explicit test-item id field instead.

METHOD
------
1. Load both wavs at a fixed sample rate (22050 Hz, matches the ffmpeg
   conversion in t9_audio_capture.py).
2. Compute log-mel spectrogram (n_mels=64, n_fft=2048, hop_length chosen so
   the frame count is roughly comparable, then explicitly resized).
3. Resize both spectrograms along the time axis to a FIXED frame count
   (FRAME_COUNT=128) via linear interpolation. This is what makes the
   comparison fair across candidates whose absolute audio length differs
   even though they all represent N_CYCLES cycles (see t9_audio_capture.py's
   cps-based duration alignment) -- the resize normalizes any residual
   length mismatch (e.g. from cps-detection error) down to the same shape.
4. Pearson correlation (cosine similarity on MEAN-CENTERED spectrograms)
   between the two, mapped to a 0-100 scale to match the other three
   metrics' convention.
   NOTE: this was originally plain cosine similarity, but empirical +
   simulated testing showed it was inflated to 80%+ even for audibly
   unrelated clips, because sparse Strudel patterns leave a lot of the
   dB-scale spectrogram at a shared silence floor (~-80dB), which
   dominates a raw flatten-and-cosine comparison. Mean-centering first
   (equivalent to Pearson correlation) removes that shared constant and
   restored expected behavior (near-0 correlation for a synthetic
   zero-overlap test case, vs. 0.99 under plain cosine).
5. SSIM (skimage) on the same resized spectrograms -- a complementary,
   locally-windowed signal. Verified discriminating in the right direction,
   but with a much narrower absolute range than Pearson (0.90 vs 0.98 on a
   synthetic zero-overlap-vs-near-identical test, compared to Pearson's
   -0.01 vs 0.66) -- read as a relative ranking signal, not an absolute one.
6. Onset F1: precision/recall/F1 between the candidate's and reference's
   detected onset times (librosa.onset.onset_detect), each normalized to a
   [0,1] fraction of that clip's own (trimmed) duration so tempo differences
   don't bias alignment, matched within a tolerance window. UNLIKE Pearson
   and SSIM (both computed on the same log-mel spectrogram, which encodes
   timbre/frequency content as much as timing), Onset F1 only looks at WHEN
   onsets occur, not their frequency content -- verified on a synthetic
   test: two clips with identical rhythm but different pitch scored 1.0
   (vs. Pearson 55/SSIM 65 on the same pair), while a clip with genuinely
   different rhythm (same pitch) scored 0.56. This is the metric to use
   when you specifically want a rhythm-only signal, isolated from timbre.

Entries with missing/failed audio (UNSAFE_CODE, TIMEOUT, SYNTAX_ERROR,
AUDIO_RUNTIME_ERROR, AUDIO_CAPTURE_FAILED, FFMPEG_CONVERT_FAILED) score 0 --
same convention as Execution Success treating those as failures.

DEPENDENCIES: librosa, numpy, scipy, scikit-image
    pip install librosa numpy scipy scikit-image --break-system-packages   (if on this sandbox)
    pip install librosa numpy scipy scikit-image                            (on your machine)
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import librosa
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim_fn

SR = 22050
N_MELS = 64
N_FFT = 2048
FRAME_COUNT = 128  # fixed target frame count after resize
SSIM_DATA_RANGE = 80.0  # matches librosa.power_to_db's default top_db=80 dynamic range
ONSET_TOLERANCE = 0.03  # fraction of (trimmed) clip duration; roughly a 16th-note-ish
                        # window at typical tempo over a 4-cycle recording -- tune if
                        # this feels too strict/lenient once you see real distributions


def load_manifest(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def log_mel(wav_path):
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    # Trim leading/trailing near-silence -- t9_audio_capture.py over-records
    # with pre-roll/trailing padding to avoid clipping the first beat, so we
    # need to trim back down to the real content before comparing.
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db  # shape (N_MELS, T)


def resize_frames(S_db, frame_count=FRAME_COUNT):
    t = S_db.shape[1]
    if t == frame_count:
        return S_db
    factor = frame_count / t
    return zoom(S_db, (1.0, factor), order=1)  # linear interp along time axis


def cosine_sim(a, b):
    """Pearson correlation (cosine similarity on MEAN-CENTERED vectors).
    Plain cosine similarity on dB-scale spectrograms is dominated by the
    shared silence floor (~-80dB after power_to_db(ref=np.max)) -- sparse
    Strudel patterns have a lot of near-silence frames even after trimming,
    and two clips' silence regions look nearly identical regardless of
    whether their actual content matches. Mean-centering removes that
    shared constant baseline so the score reflects where/how the energy
    actually differs, not how much silence both clips share.
    Verified: two spectrograms with zero overlapping content gave raw
    cosine = 0.99 but Pearson correlation = -0.01 on a synthetic test.
    """
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def ssim_sim(a, b):
    """Structural similarity (skimage), applied to the same resized log-mel
    spectrograms used for the Pearson metric. Computed per local window
    rather than globally, so it's less prone to the shared-silence inflation
    that affected raw cosine similarity -- but not immune to it: verified on
    the same synthetic test used for the Pearson fix, SSIM gave 0.90 for two
    spectrograms with ZERO overlapping content vs 0.98 for a near-identical
    pair -- correct direction, but a much narrower absolute range than
    Pearson's -0.01 vs 0.66 on the same pair. Report/interpret this as a
    RELATIVE ranking signal, not an absolute-magnitude one.
    """
    return float(ssim_fn(a, b, data_range=SSIM_DATA_RANGE))


_SPEC_CACHE = {}

DEGENERATE_STD_THRESHOLD = 1e-6  # spectrogram std below this = effectively flat/silent
# Only 1-in-a-billion-ish odds a genuine continuous correlation lands exactly
# on 0.0 -- if we see that, it's almost certainly this branch firing because
# the code ran without error but produced no real audio (e.g. an empty
# pattern, or something never routed to output), not "ran and sounds
# unrelated to the reference." Worth distinguishing for interpretability.


def is_degenerate(spec):
    return float(np.std(spec)) < DEGENERATE_STD_THRESHOLD


def get_resized_spec(wav_path):
    if wav_path not in _SPEC_CACHE:
        _SPEC_CACHE[wav_path] = resize_frames(log_mel(wav_path))
    return _SPEC_CACHE[wav_path]


_ONSET_CACHE = {}


def get_normalized_onsets(wav_path):
    """Onset times detected via librosa, normalized to [0,1] as a fraction
    of that clip's own (trimmed) duration -- this is what makes onset
    patterns comparable between candidate and reference even if their
    absolute durations differ slightly (e.g. from imperfect cps detection),
    the same principle as resize_frames() for the spectrogram metrics."""
    if wav_path not in _ONSET_CACHE:
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)
        duration = len(y) / sr
        if duration <= 0:
            _ONSET_CACHE[wav_path] = np.array([])
        else:
            onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time")
            _ONSET_CACHE[wav_path] = onset_times / duration
    return _ONSET_CACHE[wav_path]


def onset_f1(cand_wav_path, ref_wav_path, tolerance=ONSET_TOLERANCE):
    """Precision/recall/F1 between candidate and reference onset times
    (each normalized to [0,1] of its own clip duration), matched via greedy
    nearest-neighbor within `tolerance`. Unlike cosine_sim/ssim_sim (which
    operate on the log-mel spectrogram and therefore reflect both timing AND
    frequency/timbre content), this ONLY looks at onset timing -- verified
    timbre-invariant on a synthetic same-rhythm-different-pitch test (scored
    1.0, vs. 55/65 for Pearson/SSIM on the same pair), while correctly
    scoring lower (0.56) when the rhythm itself genuinely differs."""
    cand = get_normalized_onsets(cand_wav_path)
    ref = get_normalized_onsets(ref_wav_path)
    if len(ref) == 0 or len(cand) == 0:
        return 0.0

    used = np.zeros(len(cand), dtype=bool)
    tp = 0
    for r in ref:
        diffs = np.abs(cand - r)
        diffs[used] = np.inf
        j = np.argmin(diffs)
        if diffs[j] <= tolerance:
            used[j] = True
            tp += 1
    fp = int(np.sum(~used))
    fn = len(ref) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---- Hap-based similarity (weighted partial matching) ---------------------
# Queries the PATTERN directly (no audio rendering at all) via Strudel's own
# Hap/queryArc mechanism -- see hap_query_entry.mjs for how these are
# extracted. Verified in isolation: ~2ms/query vs ~8-10s for the audio
# pipeline (roughly 4000x faster), and setcps/setcpm/setbpm are irrelevant
# here since queryArc operates in cycle-space regardless of tempo.
CORE_HAP_KEYS = {"s", "n", "note", "bank"}  # "what sound/pitch is this" -- weighted higher.
                                             # This split (vs. decorative params like
                                             # gain/cutoff/pan) is a judgment call, not an
                                             # empirical finding -- revisit if scores don't
                                             # match intuition once you see real data.
HAP_CORE_WEIGHT = 0.7  # core keys get 70% of the value-similarity weight combined
HAP_TIME_TOLERANCE = 0.03  # same convention/magnitude as ONSET_TOLERANCE, but here it's a
                           # fraction of the query window in CYCLES, not clip duration
HAP_NUMERIC_RELATIVE_TOLERANCE = 0.25  # numeric params: linear partial credit down to 0
                                        # at >=25% relative difference


def _hap_key_score(a, b):
    if a is None or b is None:
        return 0.0
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        denom = max(abs(a), abs(b), 1e-6)
        rel_diff = abs(a - b) / denom
        return max(0.0, 1.0 - rel_diff / HAP_NUMERIC_RELATIVE_TOLERANCE)
    return 1.0 if a == b else 0.0


def hap_value_similarity(val_a, val_b):
    """Weighted partial match between two Hap `value` dicts: core identity
    keys (what sound/note/bank) count for HAP_CORE_WEIGHT of the score,
    everything else (gain, cutoff, pan, ...) splits the remainder. Numeric
    params get linear partial credit based on relative closeness rather than
    exact-match-only. Verified: identical values -> 1.0; only a decorative
    param (cutoff) off by ~6% -> ~0.93; a core param (bank) different ->
    ~0.65; both empty -> 1.0 (trivially equal)."""
    keys = set(val_a.keys()) | set(val_b.keys())
    if not keys:
        return 1.0
    core_keys = keys & CORE_HAP_KEYS
    other_keys = keys - CORE_HAP_KEYS
    core_w = HAP_CORE_WEIGHT if core_keys else 0.0
    other_w = (1.0 - HAP_CORE_WEIGHT) if other_keys else 0.0
    total_w = core_w + other_w
    if total_w == 0:
        return 1.0
    core_score = (sum(_hap_key_score(val_a.get(k), val_b.get(k)) for k in core_keys) / len(core_keys)
                  if core_keys else 0.0)
    other_score = (sum(_hap_key_score(val_a.get(k), val_b.get(k)) for k in other_keys) / len(other_keys)
                   if other_keys else 0.0)
    return (core_w * core_score + other_w * other_score) / total_w


def hap_f1(cand_haps, ref_haps, time_tolerance=HAP_TIME_TOLERANCE):
    """Weighted partial-match F1 between two Hap lists (each a list of
    {begin, end, value} dicts from querying the same number of cycles).
    Haps are matched by time proximity (like onset_f1's greedy nearest-
    neighbor), but instead of binary match/no-match, each matched pair
    contributes its hap_value_similarity() score as PARTIAL credit -- a
    pair that's perfectly time-aligned but has unrelated values contributes
    ~0, not a full match."""
    if not ref_haps or not cand_haps:
        return 0.0
    used = [False] * len(cand_haps)
    soft_tp = 0.0
    for ref_hap in ref_haps:
        best_j, best_dist = None, None
        for j, ch in enumerate(cand_haps):
            if used[j] or ch.get("begin") is None or ref_hap.get("begin") is None:
                continue
            dist = abs(ch["begin"] - ref_hap["begin"])
            if dist <= time_tolerance and (best_dist is None or dist < best_dist):
                best_j, best_dist = j, dist
        if best_j is not None:
            used[best_j] = True
            soft_tp += hap_value_similarity(ref_hap["value"], cand_haps[best_j]["value"])
    precision = soft_tp / len(cand_haps) if cand_haps else 0.0
    recall = soft_tp / len(ref_haps) if ref_haps else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


FAILURE_STATUSES = {
    "UNSAFE_CODE", "TIMEOUT", "SYNTAX_ERROR", "AUDIO_RUNTIME_ERROR",
    "AUDIO_CAPTURE_FAILED", "FFMPEG_CONVERT_FAILED", "PLAYWRIGHT_DOM_ERROR",
}


def extract_model_shot_seed(source_file):
    """finetuned_baseline_<model>_<k>shot_seed<n>.jsonl -> (model, k, seed)
    Adjust this regex if your candidate filenames follow a different pattern
    (e.g. plain baseline_<model>_<k>shot_seed<n>.jsonl)."""
    m = re.match(r"(?:finetuned_)?(?:baseline|ablation)_(.+)_(\d+)shot_seed(\d+)\.jsonl", source_file)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return source_file, None, None


def score_candidates(ref_manifest_path, candidate_manifest_paths, out_jsonl, out_summary_csv):
    ref_entries = load_manifest(ref_manifest_path)
    ref_by_index = {}
    for e in ref_entries:
        if e["status"] == "SUCCESS" and e.get("wav_path"):
            ref_by_index[e["entry_index"]] = e["wav_path"]
    print(f"Loaded {len(ref_by_index)} usable reference audios out of {len(ref_entries)}.")

    per_entry_rows = []
    for cand_path in candidate_manifest_paths:
        cand_entries = load_manifest(cand_path)
        for e in cand_entries:
            idx = e["entry_index"]
            source_file = e["source_file"]
            model, k, seed = extract_model_shot_seed(source_file)

            if e["status"] in FAILURE_STATUSES or not e.get("wav_path"):
                score = 0.0
                ssim_score = 0.0
                onset_score = 0.0
                reason = e["status"]
            elif idx not in ref_by_index:
                score = None
                ssim_score = None
                onset_score = None
                reason = "NO_REFERENCE_AUDIO"
            else:
                try:
                    cand_spec = get_resized_spec(e["wav_path"])
                    ref_spec = get_resized_spec(ref_by_index[idx])
                    if is_degenerate(cand_spec):
                        # Code ran without error and audio was captured, but the
                        # result is (near-)flat -- i.e. no real audio content,
                        # not "audio that doesn't match the reference." Score 0
                        # either way, but tag it so this doesn't get conflated
                        # with genuine low-correlation cases during analysis.
                        score = 0.0
                        ssim_score = 0.0
                        onset_score = 0.0
                        reason = "DEGENERATE_SILENT_CANDIDATE"
                    elif is_degenerate(ref_spec):
                        # Shouldn't normally happen (all 150 reference renders
                        # were verified non-silent), but flag rather than silently
                        # mis-score if it ever does.
                        score = None
                        ssim_score = None
                        onset_score = None
                        reason = "DEGENERATE_REFERENCE_AUDIO"
                    else:
                        corr = cosine_sim(cand_spec, ref_spec)
                        score = max(0.0, corr) * 100.0  # clip negative correlation to 0, scale to 0-100
                        ssim_score = ssim_sim(cand_spec, ref_spec) * 100.0  # SSIM is already ~[0,1] for these inputs
                        onset_score = onset_f1(e["wav_path"], ref_by_index[idx]) * 100.0
                        reason = "OK"
                except Exception as ex:
                    score = 0.0
                    ssim_score = 0.0
                    onset_score = 0.0
                    reason = f"SCORE_ERROR: {ex}"

            per_entry_rows.append({
                "source_file": source_file, "model": model, "shot": k, "seed": seed,
                "entry_index": idx, "mel_similarity": score, "ssim_similarity": ssim_score,
                "onset_f1": onset_score, "reason": reason,
            })

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in per_entry_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Per-entry scores written to {out_jsonl}")

    # aggregate: mean +/- std (ddof=1) per (model, shot), across seeds first
    # (seed-level mean, matching this project's existing convention), then
    # report those per-seed means' own mean +/- std across seeds.
    # Done for all three metrics (mel_similarity = Pearson, ssim_similarity = SSIM,
    # onset_f1 = timbre-invariant rhythm-only signal).
    def aggregate(metric_key):
        by_model_shot_seed = defaultdict(list)
        for row in per_entry_rows:
            if row[metric_key] is not None:
                by_model_shot_seed[(row["model"], row["shot"], row["seed"])].append(row[metric_key])
        seed_means = defaultdict(list)
        for (model, shot, seed), scores in by_model_shot_seed.items():
            seed_means[(model, shot)].append(np.mean(scores))
        return seed_means

    mel_seed_means = aggregate("mel_similarity")
    ssim_seed_means = aggregate("ssim_similarity")
    onset_seed_means = aggregate("onset_f1")
    all_keys = sorted(set(mel_seed_means.keys()) | set(ssim_seed_means.keys()) | set(onset_seed_means.keys()),
                       key=lambda x: (x[0], x[1] if x[1] is not None else -1))

    with open(out_summary_csv, "w", encoding="utf-8") as f:
        f.write("model,shot,mel_mean,mel_std,ssim_mean,ssim_std,onset_f1_mean,onset_f1_std,n_seeds\n")
        for model, shot in all_keys:
            mel_means = mel_seed_means.get((model, shot), [])
            ssim_means = ssim_seed_means.get((model, shot), [])
            onset_means = onset_seed_means.get((model, shot), [])
            mel_mean = np.mean(mel_means) if mel_means else float("nan")
            mel_std = np.std(mel_means, ddof=1) if len(mel_means) > 1 else 0.0
            ssim_mean = np.mean(ssim_means) if ssim_means else float("nan")
            ssim_std = np.std(ssim_means, ddof=1) if len(ssim_means) > 1 else 0.0
            onset_mean = np.mean(onset_means) if onset_means else float("nan")
            onset_std = np.std(onset_means, ddof=1) if len(onset_means) > 1 else 0.0
            n_seeds = max(len(mel_means), len(ssim_means), len(onset_means))
            f.write(f"{model},{shot},{mel_mean:.4f},{mel_std:.4f},{ssim_mean:.4f},{ssim_std:.4f},"
                    f"{onset_mean:.4f},{onset_std:.4f},{n_seeds}\n")
    print(f"Summary written to {out_summary_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_manifest", required=True, help="manifest.jsonl from capturing the 150 reference/target codes")
    parser.add_argument("--candidate_manifests", required=True, nargs="+", help="one or more manifest.jsonl files from candidate model outputs")
    parser.add_argument("--out_jsonl", default="mel_similarity_per_entry.jsonl")
    parser.add_argument("--out_summary_csv", default="mel_similarity_summary.csv")
    args = parser.parse_args()

    score_candidates(args.ref_manifest, args.candidate_manifests, args.out_jsonl, args.out_summary_csv)
