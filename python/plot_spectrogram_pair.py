"""
plot_spectrogram_pair.py

Side-by-side log-mel spectrogram comparison for a single candidate/reference
wav pair, annotated with the same Pearson correlation + SSIM scores
mel_similarity_score.py would compute for that pair. Meant for manual
spot-checking (pair a listen with a look) and as a source figure for the
thesis -- reuses the exact same preprocessing (log_mel, resize_frames) and
scoring functions (cosine_sim, ssim_sim) from mel_similarity_score.py, so
what you see here always matches what the batch scoring actually computed.

Usage:
    python plot_spectrogram_pair.py --candidate_wav path/to/cand.wav --reference_wav path/to/ref.wav --out comparison.png
    python plot_spectrogram_pair.py --candidate_wav ... --reference_wav ... --out comparison.png --title "epoch 12, entry 47"
"""

import argparse

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from mel_similarity_score import (
    log_mel, resize_frames, cosine_sim, ssim_sim, is_degenerate, SR, SSIM_DATA_RANGE,
)


def plot_pair(candidate_wav, reference_wav, out_path, title=None):
    cand_raw = log_mel(candidate_wav)
    ref_raw = log_mel(reference_wav)
    cand_spec = resize_frames(cand_raw)
    ref_spec = resize_frames(ref_raw)

    if is_degenerate(cand_spec):
        corr, ssim_score, note = None, None, "CANDIDATE IS DEGENERATE (near-silent/flat) -- scores would be 0"
    elif is_degenerate(ref_spec):
        corr, ssim_score, note = None, None, "REFERENCE IS DEGENERATE (near-silent/flat) -- unexpected, check this file"
    else:
        corr = max(0.0, cosine_sim(cand_spec, ref_spec)) * 100.0
        ssim_score = ssim_sim(cand_spec, ref_spec) * 100.0
        note = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, spec, label in [(axes[0], ref_spec, "Reference"), (axes[1], cand_spec, "Candidate")]:
        img = librosa.display.specshow(
            spec, sr=SR, x_axis="time", y_axis="mel", ax=ax, cmap="magma",
            vmin=-SSIM_DATA_RANGE, vmax=0,
        )
        ax.set_title(label)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    if note:
        header = note
    else:
        header = f"Pearson (mel_similarity) = {corr:.1f}   |   SSIM = {ssim_score:.1f}"
    full_title = f"{title}\n{header}" if title else header
    fig.suptitle(full_title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if not note:
        print(f"Pearson (mel_similarity): {corr:.2f}")
        print(f"SSIM (ssim_similarity):   {ssim_score:.2f}")
    else:
        print(note)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_wav", required=True)
    parser.add_argument("--reference_wav", required=True)
    parser.add_argument("--out", default="spectrogram_comparison.png")
    parser.add_argument("--title", default=None, help="optional label, e.g. 'entry 47, seed 2'")
    args = parser.parse_args()

    plot_pair(args.candidate_wav, args.reference_wav, args.out, args.title)
