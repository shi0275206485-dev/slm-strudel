import asyncio
import json
import os
import glob
import re
from collections import defaultdict
from playwright.async_api import async_playwright
from tqdm import tqdm

STRUDEL_LOCAL_URL = "http://localhost:4321"
INPUT_DIR = "./baseline_shot"
OUTPUT_DIR = "./baseline_shot/1/outputs"


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


# Per-entry hard cap (seconds). If a single snippet stalls the editor (e.g. an
# autocomplete popup swallows keystrokes, or the page enters a bad state after a
# prior eval), the whole entry is abandoned and classified TIMEOUT so the batch
# never hangs. 15s comfortably exceeds the ~0.5-1s a normal entry takes.
ENTRY_TIMEOUT_S = 15
OP_TIMEOUT_MS = 5000  # cap on each individual Playwright DOM/keyboard op


# --- static pre-screen -----------------------------------------------------
# Some generations fall back to general-purpose JS and emit non-terminating
# loops (e.g. `while (true) { ... }`). These are syntactically valid, so the
# editor's Ctrl+Enter actually RUNS them, and an infinite loop pins the page's
# JS thread forever — Playwright cannot recover it because the render thread is
# saturated. We therefore detect these BEFORE injecting the code and classify
# them as UNSAFE_CODE without evaluating. (They are never valid Strudel anyway:
# Strudel is declarative pattern code and does not use raw while/for loops.)
import re as _re_unsafe

_UNSAFE_PATTERNS = [
    (r'\bwhile\s*\(\s*true\s*\)',        'while(true)'),
    (r'\bwhile\s*\(\s*1\s*\)',           'while(1)'),
    (r'\bfor\s*\(\s*;\s*;\s*\)',         'for(;;)'),
    (r'\bfor\s*\(\s*;\s*true\s*;\s*\)',  'for(;true;)'),
    (r'\bdo\s*\{[\s\S]*?\}\s*while\s*\(\s*(?:true|1)\s*\)', 'do...while(true)'),
]

def screen_unsafe_code(code):
    """Return a short reason string if the code contains a non-terminating loop,
    else None."""
    for pat, label in _UNSAFE_PATTERNS:
        if _re_unsafe.search(pat, code):
            return label
    return None


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


async def validate_code_with_audio(page, code_snippet, monitor):
    monitor.reset()

    # Pre-screen FIRST: never inject a known infinite loop into the live editor.
    unsafe = screen_unsafe_code(code_snippet)
    if unsafe:
        return {"status": "UNSAFE_CODE", "message": f"non-terminating loop: {unsafe}"}

    try:
        # Outer hard cap: even if a Playwright op ignores its own timeout, this
        # guarantees the entry returns within ENTRY_TIMEOUT_S.
        return await asyncio.wait_for(
            _validate_inner(page, code_snippet, monitor),
            timeout=ENTRY_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        # Try to reset the editor so the NEXT entry starts clean.
        try:
            await page.keyboard.press('Escape')          # dismiss any popup
            await page.click('.cm-content', timeout=2000)
            await page.keyboard.press('Control+A')
            await page.keyboard.press('Backspace')
        except Exception:
            pass
        return {"status": "TIMEOUT", "message": f"entry exceeded {ENTRY_TIMEOUT_S}s"}
    except Exception as e:
        return {"status": "PLAYWRIGHT_DOM_ERROR", "message": str(e)}


async def make_page(browser, monitor):
    """Create a fresh page, wire the monitor, load Strudel, and prime audio.
    Used for initial setup AND for recovery after a hang (a page whose JS thread
    was pinned by an infinite loop is a zombie and must be replaced, not reused)."""
    page = await browser.new_page()
    page.on("console", monitor.handle_console)
    page.on("pageerror", monitor.handle_pageerror)
    await page.goto(STRUDEL_LOCAL_URL, wait_until="networkidle")
    await page.mouse.click(100, 100)
    await page.wait_for_timeout(2000)
    return page


async def process_file(browser, page, monitor, input_file):
    fname = os.path.basename(input_file)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_success = os.path.join(OUTPUT_DIR, f"verified_{fname}")
    output_failed = os.path.join(OUTPUT_DIR, f"bad_cases_{fname}")

    success_count = 0
    fail_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_success, 'w', encoding='utf-8') as f_ok, \
         open(output_failed, 'w', encoding='utf-8') as f_bad:

        lines = f_in.readlines()
        print(f"\n[{fname}] {len(lines)} entries...")

        for line in tqdm(lines, desc=fname):
            if not line.strip():
                continue
            data = json.loads(line)
            code_snippet = data["messages"][-1]["content"]
            result = await validate_code_with_audio(page, code_snippet, monitor)

            # If an entry hung the page despite pre-screening, the page may be a
            # zombie (JS thread saturated). Replace it with a fresh page so the
            # remaining entries run on a clean editor.
            if result["status"] == "TIMEOUT":
                try:
                    await page.close()
                except Exception:
                    pass
                page = await make_page(browser, monitor)

            if result["status"] == "SUCCESS":
                f_ok.write(json.dumps(data, ensure_ascii=False) + "\n")
                success_count += 1
            else:
                data["validation_error"] = result
                f_bad.write(json.dumps(data, ensure_ascii=False) + "\n")
                fail_count += 1

    total = success_count + fail_count
    rate = success_count / total if total > 0 else 0.0
    print(f"[{fname}] success: {success_count}, fail: {fail_count}, rate: {rate:.4f}")
    return success_count, fail_count, page


def extract_model_name(filename):
    # baseline_{model}_1shot_seed{N}.jsonl  ->  {model}
    # ablation_{model}_1shot_seed{N}.jsonl  ->  {model}
    # finetuned_baseline_{model}_0shot_seed{N}.jsonl  ->  {model}

    m = re.match(r"baseline_(.+)_1shot_seed\d+\.jsonl", os.path.basename(filename))
    return m.group(1) if m else filename

    # m = re.match(r"finetuned_baseline_(.+)_0shot_seed\d+\.jsonl", os.path.basename(filename))
    # return m.group(1) if m else filename


async def main():
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "baseline_*_1shot_seed*.jsonl")))
    # input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "finetuned_baseline_*_0shot_seed*.jsonl")))
    if not input_files:
        print(f"No files found in {INPUT_DIR}")
        return

    print(f"Found {len(input_files)} files across seeds.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"]
        )
        monitor = PlaywrightMonitor()
        print(f"Loading Strudel: {STRUDEL_LOCAL_URL}")
        page = await make_page(browser, monitor)

        # model -> list of per-seed success rates
        model_rates = defaultdict(list)

        for input_file in input_files:
            model = extract_model_name(input_file)
            s, f, page = await process_file(browser, page, monitor, input_file)
            total = s + f
            rate = s / total if total > 0 else 0.0
            model_rates[model].append(rate)

        await browser.close()

    print("\n" + "=" * 60)
    print(f"{'Model':<45} {'Avg Success Rate':>16}")
    print("=" * 60)
    for model, rates in sorted(model_rates.items()):
        avg = sum(rates) / len(rates)
        seed_info = "  [" + ", ".join(f"{r:.4f}" for r in rates) + "]"
        print(f"{model:<45} {avg:>16.4f}{seed_info}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
