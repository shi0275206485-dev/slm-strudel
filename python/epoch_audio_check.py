"""
epoch_audio_check.py

Thin CLI wrapper around t9_audio_capture.py's capture_entry_audio /
make_page (same safety patterns: unsafe-code pre-screen, cps-based
duration alignment, pre-roll/trailing padding, page-recovery after
TIMEOUT), for a single input completions file. Used by
training_callback.py to capture audio for one epoch's 150 generated
completions.

Usage:
    python epoch_audio_check.py --input epoch_N_completions.jsonl --out_dir epoch_N_audio --out_manifest epoch_N_audio_manifest.jsonl
"""

import argparse
import asyncio
import json
import os

from playwright.async_api import async_playwright

from t9_audio_capture import make_page, capture_entry_audio, PlaywrightMonitor


async def run(input_file, out_dir, out_manifest):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    os.makedirs(out_dir, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"],
        )
        monitor = PlaywrightMonitor()
        page = await make_page(browser, monitor)

        with open(out_manifest, "w", encoding="utf-8") as out_f:
            for idx, data in enumerate(lines):
                code = data["messages"][-1]["content"]
                out_wav = os.path.join(out_dir, f"entry_{idx}.wav")
                result = await capture_entry_audio(page, code, monitor, out_wav)

                if result["status"] == "TIMEOUT":
                    try:
                        await page.close()
                    except Exception:
                        pass
                    page = await make_page(browser, monitor)

                out_f.write(json.dumps({"entry_index": idx, **result}, ensure_ascii=False) + "\n")

        await browser.close()

    print(f"AUDIO_CAPTURE_DONE={len(lines)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_manifest", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.input, args.out_dir, args.out_manifest))
