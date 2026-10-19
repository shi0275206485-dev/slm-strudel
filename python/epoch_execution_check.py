"""
epoch_execution_check.py

Execution-only check (no audio recording) -- adapted directly from t9.py's
validation logic (same unsafe-code pre-screen, same per-entry timeout, same
page-recovery after TIMEOUT), stripped down to a single input/output pair
driven by CLI args, meant to be invoked once per epoch via subprocess from
the training callback. Prints EXECUTION_PASS_RATE=<float> at the end for
the calling process to grab from stdout.

Usage:
    python epoch_execution_check.py --input epoch_N_completions.jsonl --output_manifest epoch_N_execution_manifest.jsonl
"""

import argparse
import asyncio
import json
import re

from playwright.async_api import async_playwright

STRUDEL_LOCAL_URL = "http://localhost:4321"
ENTRY_TIMEOUT_S = 15
OP_TIMEOUT_MS = 5000

_UNSAFE_PATTERNS = [
    (r'\bwhile\s*\(\s*true\s*\)',        'while(true)'),
    (r'\bwhile\s*\(\s*1\s*\)',           'while(1)'),
    (r'\bfor\s*\(\s*;\s*;\s*\)',         'for(;;)'),
    (r'\bfor\s*\(\s*;\s*true\s*;\s*\)',  'for(;true;)'),
    (r'\bdo\s*\{[\s\S]*?\}\s*while\s*\(\s*(?:true|1)\s*\)', 'do...while(true)'),
]


def screen_unsafe_code(code):
    for pat, label in _UNSAFE_PATTERNS:
        if re.search(pat, code):
            return label
    return None


class PlaywrightMonitor:
    def __init__(self):
        self.syntax_errors = []
        self.audio_errors = []

    def reset(self):
        self.syntax_errors.clear()
        self.audio_errors.clear()

    def handle_console(self, msg):
        if msg.type != "error":
            return
        text = msg.text
        if "Label '$'" in text:
            return
        if "SyntaxError" in text or "TypeError" in text or "ReferenceError" in text or "Error" in text:
            self.syntax_errors.append(text)
        elif "AudioContext" in text or "node" in text.lower():
            self.audio_errors.append(text)

    def handle_pageerror(self, err):
        text = err.message
        if "Label '$'" not in text:
            self.syntax_errors.append(text)


async def _validate_inner(page, code_snippet, monitor):
    await page.click('.cm-content', timeout=OP_TIMEOUT_MS)
    await page.keyboard.press('Control+A')
    await page.keyboard.press('Backspace')
    await page.keyboard.insert_text(code_snippet)
    await page.keyboard.press('Control+Enter')
    await page.wait_for_timeout(500)
    await page.keyboard.press('Control+.')

    if len(monitor.syntax_errors) > 0:
        return {"status": "SYNTAX_ERROR", "message": " | ".join(monitor.syntax_errors)}
    elif len(monitor.audio_errors) > 0:
        return {"status": "AUDIO_RUNTIME_ERROR", "message": " | ".join(monitor.audio_errors)}
    else:
        return {"status": "SUCCESS"}


async def validate_code(page, code_snippet, monitor):
    monitor.reset()
    unsafe = screen_unsafe_code(code_snippet)
    if unsafe:
        return {"status": "UNSAFE_CODE", "message": f"non-terminating loop: {unsafe}"}
    try:
        return await asyncio.wait_for(_validate_inner(page, code_snippet, monitor), timeout=ENTRY_TIMEOUT_S)
    except asyncio.TimeoutError:
        try:
            await page.keyboard.press('Escape')
            await page.click('.cm-content', timeout=2000)
            await page.keyboard.press('Control+A')
            await page.keyboard.press('Backspace')
        except Exception:
            pass
        return {"status": "TIMEOUT", "message": f"entry exceeded {ENTRY_TIMEOUT_S}s"}
    except Exception as e:
        return {"status": "PLAYWRIGHT_DOM_ERROR", "message": str(e)}


async def make_page(browser, monitor):
    page = await browser.new_page()
    page.on("console", monitor.handle_console)
    page.on("pageerror", monitor.handle_pageerror)
    # "networkidle" is flaky against Vite dev servers -- Vite keeps an HMR
    # websocket open, which can prevent the page from ever looking "fully
    # idle," and the very first hit after starting/restarting the dev server
    # can also be slow (on-demand dependency pre-bundling). "load" is more
    # deterministic. Retry a few times regardless, since a single transient
    # timeout here shouldn't be allowed to silently zero out an entire
    # epoch's score.
    last_err = None
    for attempt in range(3):
        try:
            await page.goto(STRUDEL_LOCAL_URL, wait_until="load", timeout=30000)
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(f"[epoch_execution_check] page.goto attempt {attempt+1}/3 failed: {e}")
            await page.wait_for_timeout(2000)
    if last_err is not None:
        raise last_err
    # Wait for the actual thing we need (the code editor), not a guessed
    # fixed delay -- "load" fires earlier than "networkidle" did, so Strudel
    # may still be async-loading sample banks/chunks at that point. Waiting
    # for the editor element to actually be attached is a more reliable
    # readiness signal than sleeping a fixed amount and hoping.
    await page.wait_for_selector('.cm-content', timeout=15000)
    await page.mouse.click(100, 100)
    await page.wait_for_timeout(1000)
    return page


async def run(input_file, output_manifest):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f if l.strip()]

    success = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"],
        )
        monitor = PlaywrightMonitor()
        page = await make_page(browser, monitor)

        with open(output_manifest, 'w', encoding='utf-8') as out_f:
            for idx, data in enumerate(lines):
                code = data["messages"][-1]["content"]
                result = await validate_code(page, code, monitor)

                if result["status"] == "TIMEOUT":
                    try:
                        await page.close()
                    except Exception:
                        pass
                    page = await make_page(browser, monitor)

                if result["status"] == "SUCCESS":
                    success += 1

                out_f.write(json.dumps({"entry_index": idx, **result}, ensure_ascii=False) + "\n")

        await browser.close()

    rate = success / len(lines) if lines else 0.0
    print(f"EXECUTION_PASS_RATE={rate}")
    print(f"EXECUTION_SUCCESS={success}")
    print(f"EXECUTION_TOTAL={len(lines)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_manifest", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output_manifest))
