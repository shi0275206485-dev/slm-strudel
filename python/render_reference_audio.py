"""
render_reference_audio.py

Renders the 150 frozen test-set reference/target codes into reference audio,
producing a manifest in the same format t9_audio_capture.py produces for
candidates -- mel_similarity_score.py's --ref_manifest expects this.

This is INDEPENDENT of the finetuned-model generation status: the reference
codes are fixed (they're the ground-truth "assistant" completions in your
frozen 150-item test set), so this can run now, in parallel with generating
the remaining K=1,3,5,10,11 finetuned outputs.

ASSUMPTION (please confirm/adjust):
Reference file is JSONL, one entry per line, same "messages" chat schema as
your candidate output files -- i.e. the reference code is at
data["messages"][-1]["content"]. If your frozen test-set file uses a
different field name for the target code, change REF_CODE_FIELD_EXTRACTOR
below.

Usage:
    python render_reference_audio.py --ref_file path/to/test_set_150.jsonl
"""

import argparse
import asyncio
import json
import os

from tqdm import tqdm

from t9_audio_capture import make_page, capture_entry_audio, PlaywrightMonitor
from playwright.async_api import async_playwright

DEFAULT_AUDIO_DIR = "./audio_capture/reference"


def extract_ref_code(data):
    # Adjust here if your test-set schema differs.
    return data["messages"][-1]["content"]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file", required=True, help="path to the frozen 150-item test set JSONL")
    parser.add_argument("--out_dir", default=None, help="where to write wavs + manifest; defaults to a folder named after --ref_file so multiple test sets don't collide")
    parser.add_argument("--out_manifest", default=None, help="defaults to <out_dir>/reference_manifest.jsonl")
    args = parser.parse_args()

    if args.out_dir:
        audio_dir = args.out_dir
    else:
        tag = os.path.splitext(os.path.basename(args.ref_file))[0]
        audio_dir = os.path.join(DEFAULT_AUDIO_DIR, tag)
    out_manifest = args.out_manifest or os.path.join(audio_dir, "reference_manifest.jsonl")

    with open(args.ref_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(lines)} reference entries from {args.ref_file}")
    print(f"Output dir: {audio_dir}")

    os.makedirs(audio_dir, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"],
        )
        monitor = PlaywrightMonitor()
        page = await make_page(browser, monitor)

        with open(out_manifest, "w", encoding="utf-8") as manifest_f:
            for idx, data in enumerate(tqdm(lines, desc="reference audio")):
                code = extract_ref_code(data)
                out_wav = os.path.join(audio_dir, f"entry_{idx}.wav")

                result = await capture_entry_audio(page, code, monitor, out_wav)

                if result["status"] == "TIMEOUT":
                    try:
                        await page.close()
                    except Exception:
                        pass
                    page = await make_page(browser, monitor)

                manifest_f.write(json.dumps({
                    "source_file": "reference",
                    "entry_index": idx,
                    "ref_function": data.get("ref_function"),
                    **result,
                }, ensure_ascii=False) + "\n")
                manifest_f.flush()

        await browser.close()

    print(f"\nDone. Reference manifest written to {out_manifest}")
    print("Use this as --ref_manifest in mel_similarity_score.py once candidate captures are ready.")


if __name__ == "__main__":
    asyncio.run(main())
