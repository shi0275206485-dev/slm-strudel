"""
t9_audio_capture.py

Extends t9.py: same execution-validation pipeline (unsafe-code pre-screen,
per-entry timeout, page-recovery after TIMEOUT), PLUS records the audio
output of each successfully-evaluated snippet, aligned to N_CYCLES cycles.

HOW THE AUDIO TAP WORKS
------------------------
page.add_init_script() runs BEFORE any page script on every navigation, so we
use it to monkey-patch window.AudioContext before Strudel constructs its own.
The patch:
  1. Wraps the (single) AudioContext Strudel creates, attaching a
     MediaStreamAudioDestinationNode + MediaRecorder to it.
  2. Overrides AudioNode.prototype.connect so that any node connecting to
     ctx.destination (i.e. "the master output") is ALSO connected to our
     MediaStreamAudioDestinationNode. This taps the exact signal that would
     otherwise go to speakers, without touching Strudel's own code.
This is a generic Web Audio technique -- it does not depend on knowing any
Strudel-internal variable names, so it should be robust across versions.

CPS DETECTION (IMPORTANT CAVEAT -- please verify before trusting results)
---------------------------------------------------------------------------
To align recording length to N_CYCLES cycles we need the real cps (cycles
per second) the generated code will run at. This version detects cps by
regex-parsing the code text for the standard tempo setters:
    setcps(X)           -> cps = X                  (direct, high confidence)
    setcpm(X)            -> cps = X / 60              (ASSUMED -- please verify)
    setbpm(X)             -> cps = X / 240             (verified against the
                                                          actual krill.pegjs
                                                          grammar: setbpm
                                                          converts to setcps
                                                          via value/120/2)
    .cps(X) / .cpm(X)    -> same conversions as above, chained form
If none match, falls back to DEFAULT_CPS (0.5, Strudel's commonly documented
default -- please double check this against the version you have running
locally, defaults can change between releases).
This is a text-parse approximation, NOT a live introspection of Strudel's
actual scheduler state (we don't know what your local build exposes on
`window` for that -- if you can find a live accessor, e.g. by checking
`window` in devtools after evaluating a snippet, that would be more robust
and I can switch to it).

DEPENDENCIES
------------
- ffmpeg must be installed and on PATH (used to convert the recorded
  audio/webm blob to .wav for downstream librosa analysis).
- Same Playwright setup as t9.py.

RECOMMENDED FIRST STEP
-----------------------
Run this on a SMALL batch (10-20 known-good entries) first to confirm:
  (a) the captured .wav files actually contain audio (not silence),
  (b) per-entry timing, so we can estimate full-batch cost before committing
      to a scope for the full sweep.
"""

import asyncio
import base64
import json
import os
import re
import subprocess
import glob
from collections import defaultdict
from playwright.async_api import async_playwright
from tqdm import tqdm

STRUDEL_LOCAL_URL = "http://localhost:4321"
INPUT_DIR = "./finetuned-0-shot"
OUTPUT_DIR = "./finetuned-0-shot/outputs"
AUDIO_DIR = "./audio_capture"  # .webm + .wav land here

N_CYCLES = 4                 # cycles to capture per entry (confirmed default)
DEFAULT_CPS = 0.5            # fallback if no tempo setter found in code (VERIFY)
MIN_RECORD_S = 2.0           # safety clamp: never record less than this
MAX_RECORD_S = 20.0          # safety clamp: never record more than this
                              # (prevents one pathologically slow cps entry
                              # from blowing up the whole batch's runtime)

# MediaRecorder's encoder needs a moment to actually start capturing after
# .start(), and Strudel's own scheduler has some lookahead before the first
# event actually sounds -- without slack here the first beat gets clipped.
# Rather than guessing the exact latency, we over-record on both ends and
# trim leading/trailing near-silence in analysis (librosa.effects.trim),
# which is robust to latency variance instead of requiring exact timing.
PRE_ROLL_S = 0.3      # wait after recorder.start(), before triggering Ctrl+Enter
TRAILING_PAD_S = 1.0  # extra recording time appended after the computed duration

ENTRY_TIMEOUT_S = 30         # raised vs t9.py's 15s to accommodate recording
OP_TIMEOUT_MS = 5000

# If ffmpeg isn't reliably on PATH (e.g. VS Code's integrated terminal hasn't
# picked up a PATH change until VS Code itself is restarted), set this to the
# absolute path of ffmpeg.exe instead, e.g.:
#   FFMPEG_PATH = r"C:\Users\<you>\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
# Find the real path with (in a terminal where `ffmpeg -version` already works):
#   Get-Command ffmpeg | Select-Object -ExpandProperty Source
FFMPEG_PATH = "ffmpeg"  # falls back to relying on PATH


# --- static pre-screen (unchanged from t9.py) -------------------------------
_UNSAFE_PATTERNS = [
    (r'\bwhile\s*\(\s*true\s*\)',        'while(true)'),
    (r'\bwhile\s*\(\s*1\s*\)',           'while(1)'),
    (r'\bfor\s*\(\s*;\s*;\s*\)',         'for(;;)'),
    (r'\bfor\s*\(\s*;\s*true\s*;\s*\)',  'for(;true;)'),
    (r'\bdo\s*\{[\s\S]*?\}\s*while\s*\(\s*(?:true|1)\s*\)', 'do...while(true)'),
]

# Event-explosion guard. Syntactically valid, non-looping code that still
# hangs browser execution / audio rendering by instructing Strudel to pack
# an astronomically large number of events into the query window. Two cases:
#   (a) a SINGLE huge multiplier -- e.g. `hh*4096`, `.fast(2048)`, `.ply(500)`;
#   (b) MANY chained density calls whose PRODUCT explodes even though each
#       one alone is harmless -- e.g. `.fast(2).fast(4)...fast(2048)` (the
#       real Gemma3-1B few-shot output that hung the pipeline, product=2^66).
# slow/div shrink density, so they are deliberately excluded.
_SINGLE_FACTOR_LIMIT = 256
_CUMULATIVE_FACTOR_LIMIT = 512


def screen_unsafe_code(code):
    for pat, label in _UNSAFE_PATTERNS:
        if re.search(pat, code):
            return label

    # (a) any single oversized density factor
    for m in re.finditer(r'(?:\bfast|\bply)\s*\(\s*(\d+(?:\.\d+)?)\s*\)', code):
        if float(m.group(1)) > _SINGLE_FACTOR_LIMIT:
            return f"oversized density factor: {m.group(0)}"
    for m in re.finditer(r'\*\s*(\d+)', code):
        if int(m.group(1)) > _SINGLE_FACTOR_LIMIT:
            return f"oversized repetition factor: *{m.group(1)}"

    # (b) cumulative product of chained fast/ply factors
    product = 1.0
    for m in re.finditer(r'(?:\bfast|\bply)\s*\(\s*(\d+(?:\.\d+)?)\s*\)', code):
        product *= float(m.group(1))
        if product > _CUMULATIVE_FACTOR_LIMIT:
            return f"cumulative density explosion (product>{_CUMULATIVE_FACTOR_LIMIT})"

    return None


# --- cps detection (see module docstring re: confidence level) -------------
_CPS_PATTERNS = [
    (r'setcps\(\s*([\d.]+)\s*\)', lambda v: v),
    (r'\.cps\(\s*([\d.]+)\s*\)', lambda v: v),
    (r'setcpm\(\s*([\d.]+)\s*\)', lambda v: v / 60.0),
    (r'\.cpm\(\s*([\d.]+)\s*\)', lambda v: v / 60.0),
    (r'setbpm\(\s*([\d.]+)\s*\)', lambda v: v / 240.0),  # verified via krill.pegjs
    (r'\.bpm\(\s*([\d.]+)\s*\)', lambda v: v / 240.0),
]


def detect_cps_from_code(code, default_cps=DEFAULT_CPS):
    for pat, conv in _CPS_PATTERNS:
        m = re.search(pat, code)
        if m:
            try:
                return conv(float(m.group(1))), "parsed"
            except (ValueError, ZeroDivisionError):
                pass
    return default_cps, "default"


def compute_record_duration(cps, n_cycles=N_CYCLES):
    dur = n_cycles / cps if cps > 0 else n_cycles / DEFAULT_CPS
    return max(MIN_RECORD_S, min(MAX_RECORD_S, dur))


# --- audio tap init script (runs before Strudel's own JS on every load) ----
AUDIO_TAP_INIT_SCRIPT = """
(function () {
  const NativeAC = window.AudioContext || window.webkitAudioContext;
  if (!NativeAC) return;

  function wrap(ctx) {
    if (ctx.__tapped) return;
    ctx.__tapped = true;
    try {
      const streamDest = ctx.createMediaStreamDestination();
      ctx.__tapDestination = streamDest;
      let recorder;
      try {
        recorder = new MediaRecorder(streamDest.stream, { mimeType: 'audio/webm;codecs=opus' });
      } catch (e) {
        recorder = new MediaRecorder(streamDest.stream);
      }
      const chunks = [];
      recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
      ctx.__recorder = recorder;
      ctx.__chunks = chunks;
      window.__strudelCtx = ctx;
    } catch (e) {
      console.error('[audio-tap] wrap failed', e);
    }
  }

  class PatchedAudioContext extends NativeAC {
    constructor(...args) {
      super(...args);
      wrap(this);
    }
  }
  window.AudioContext = PatchedAudioContext;
  window.webkitAudioContext = PatchedAudioContext;

  const origConnect = AudioNode.prototype.connect;
  AudioNode.prototype.connect = function (dest, ...rest) {
    const result = origConnect.call(this, dest, ...rest);
    try {
      const ctx = this.context;
      if (ctx && ctx.__tapDestination && dest === ctx.destination) {
        origConnect.call(this, ctx.__tapDestination);
      }
    } catch (e) { /* ignore */ }
    return result;
  };

  window.__audioTapStart = function () {
    const ctx = window.__strudelCtx;
    if (!ctx || !ctx.__recorder) return false;
    ctx.__chunks.length = 0;
    if (ctx.state === 'suspended') { ctx.resume(); }
    ctx.__recorder.start();
    return true;
  };

  window.__audioTapStopAndGetBase64 = function () {
    return new Promise((resolve) => {
      const ctx = window.__strudelCtx;
      if (!ctx || !ctx.__recorder) { resolve(null); return; }
      if (ctx.__recorder.state !== 'recording') { resolve(null); return; }
      ctx.__recorder.onstop = async () => {
        try {
          const blob = new Blob(ctx.__chunks, { type: 'audio/webm' });
          const buf = await blob.arrayBuffer();
          const bytes = new Uint8Array(buf);
          let binary = '';
          const CHUNK = 0x8000;
          for (let i = 0; i < bytes.length; i += CHUNK) {
            binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
          }
          resolve(btoa(binary));
        } catch (e) {
          resolve(null);
        }
      };
      ctx.__recorder.stop();
    });
  };
})();
"""


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


async def make_page(browser, monitor):
    """Same as t9.py's make_page, plus the audio-tap init script is attached
    BEFORE navigation so it's active when Strudel's bundle constructs its
    AudioContext.

    "networkidle" is flaky against Vite dev servers (persistent HMR
    websocket, plus slow first-hit dependency pre-bundling), so we use
    "load" instead and retry a few times -- a single transient timeout here
    shouldn't be allowed to silently zero out an entire batch/epoch's score."""
    page = await browser.new_page()
    await page.add_init_script(AUDIO_TAP_INIT_SCRIPT)
    page.on("console", monitor.handle_console)
    page.on("pageerror", monitor.handle_pageerror)
    last_err = None
    for attempt in range(3):
        try:
            await page.goto(STRUDEL_LOCAL_URL, wait_until="load", timeout=30000)
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(f"[t9_audio_capture] page.goto attempt {attempt+1}/3 failed: {e}")
            await page.wait_for_timeout(2000)
    if last_err is not None:
        raise last_err
    # Wait for the actual editor element, not a guessed fixed delay -- see
    # epoch_execution_check.py's make_page for why ("load" fires earlier
    # than "networkidle" did, so Strudel may still be async-loading sample
    # banks/chunks when it fires).
    await page.wait_for_selector('.cm-content', timeout=15000)
    await page.mouse.click(100, 100)
    await page.wait_for_timeout(2000)
    return page


async def _capture_inner(page, code_snippet, monitor, record_s):
    await page.click('.cm-content', timeout=OP_TIMEOUT_MS)
    await page.keyboard.press('Control+A')
    await page.keyboard.press('Backspace')
    await page.keyboard.insert_text(code_snippet)

    started = await page.evaluate("window.__audioTapStart()")
    # Pre-roll: give the MediaRecorder encoder a moment to actually start
    # capturing, and give Strudel's scheduler lookahead some slack, before
    # we trigger playback. Without this the first beat can be clipped.
    await page.wait_for_timeout(int(PRE_ROLL_S * 1000))
    await page.keyboard.press('Control+Enter')
    await page.wait_for_timeout(500)  # let evaluation settle before we trust error state

    if len(monitor.syntax_errors) > 0:
        # still stop the (empty/garbage) recording so state is clean for next entry
        await page.evaluate("window.__audioTapStopAndGetBase64()")
        await page.keyboard.press('Control+.')
        return {"status": "SYNTAX_ERROR", "message": " | ".join(monitor.syntax_errors)}, None

    # wait out the recording window (minus the 500ms already elapsed), plus
    # trailing padding so latency variance never clips the tail either --
    # silence at both edges gets trimmed later during analysis.
    remaining_ms = max(0, int((record_s + TRAILING_PAD_S) * 1000) - 500)
    await page.wait_for_timeout(remaining_ms)

    if len(monitor.audio_errors) > 0:
        await page.evaluate("window.__audioTapStopAndGetBase64()")
        await page.keyboard.press('Control+.')
        return {"status": "AUDIO_RUNTIME_ERROR", "message": " | ".join(monitor.audio_errors)}, None

    b64 = None
    if started:
        b64 = await page.evaluate("window.__audioTapStopAndGetBase64()")
    await page.keyboard.press('Control+.')  # hush

    if not started or not b64:
        return {"status": "AUDIO_CAPTURE_FAILED", "message": "recorder did not start or returned no data"}, None

    return {"status": "SUCCESS"}, b64


async def capture_entry_audio(page, code_snippet, monitor, out_wav_path, n_cycles=N_CYCLES):
    monitor.reset()

    unsafe = screen_unsafe_code(code_snippet)
    if unsafe:
        return {"status": "UNSAFE_CODE", "message": f"non-terminating loop: {unsafe}", "cps": None, "cps_source": None}

    cps, cps_source = detect_cps_from_code(code_snippet)
    record_s = compute_record_duration(cps, n_cycles)

    try:
        result, b64 = await asyncio.wait_for(
            _capture_inner(page, code_snippet, monitor, record_s),
            timeout=ENTRY_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        try:
            await page.keyboard.press('Escape')
            await page.click('.cm-content', timeout=2000)
            await page.keyboard.press('Control+A')
            await page.keyboard.press('Backspace')
        except Exception:
            pass
        return {"status": "TIMEOUT", "message": f"entry exceeded {ENTRY_TIMEOUT_S}s", "cps": cps, "cps_source": cps_source}
    except Exception as e:
        return {"status": "PLAYWRIGHT_DOM_ERROR", "message": str(e), "cps": cps, "cps_source": cps_source}

    result["cps"] = cps
    result["cps_source"] = cps_source
    result["record_s"] = record_s

    if result["status"] == "SUCCESS" and b64:
        os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
        webm_path = out_wav_path.replace(".wav", ".webm")
        with open(webm_path, "wb") as f:
            f.write(base64.b64decode(b64))
        ok = _convert_to_wav(webm_path, out_wav_path)
        if not ok:
            result["status"] = "FFMPEG_CONVERT_FAILED"
        else:
            os.remove(webm_path)
            result["wav_path"] = out_wav_path

    return result


def _convert_to_wav(webm_path, wav_path):
    try:
        subprocess.run(
            [FFMPEG_PATH, "-y", "-loglevel", "error", "-i", webm_path, "-ar", "22050", "-ac", "1", wav_path],
            check=True,
        )
        return os.path.exists(wav_path) and os.path.getsize(wav_path) > 0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[ffmpeg convert failed] {webm_path}: {e}")
        return False


# --- batch driver ------------------------------------------------------------
# Mirrors t9.py's process_file/main structure, but calls capture_entry_audio
# instead of validate_code_with_audio, and writes a manifest recording every
# entry's status + wav path + detected cps, for the similarity scorer to
# consume next.

def extract_model_name(filename):
    m = re.match(r"finetuned_baseline_(.+)_0shot_seed\d+\.jsonl", os.path.basename(filename))
    return m.group(1) if m else filename


async def process_file(browser, page, monitor, input_file, manifest_f):
    fname = os.path.basename(input_file)
    with open(input_file, "r", encoding="utf-8") as f_in:
        lines = [l for l in f_in.readlines() if l.strip()]

    print(f"\n[{fname}] {len(lines)} entries...")
    for idx, line in enumerate(tqdm(lines, desc=fname)):
        data = json.loads(line)
        code_snippet = data["messages"][-1]["content"]
        ref_function = data.get("ref_function")
        out_wav = os.path.join(AUDIO_DIR, os.path.splitext(fname)[0], f"entry_{idx}.wav")

        result = await capture_entry_audio(page, code_snippet, monitor, out_wav)

        if result["status"] == "TIMEOUT":
            try:
                await page.close()
            except Exception:
                pass
            page = await make_page(browser, monitor)

        manifest_f.write(json.dumps({
            "source_file": fname,
            "entry_index": idx,
            "ref_function": ref_function,
            **result,
        }, ensure_ascii=False) + "\n")
        manifest_f.flush()

    return page


async def main():
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "finetuned_baseline_*_0shot_seed*.jsonl")))
    if not input_files:
        print(f"No files found in {INPUT_DIR}")
        return

    print(f"Found {len(input_files)} files.")
    os.makedirs(AUDIO_DIR, exist_ok=True)
    manifest_path = os.path.join(AUDIO_DIR, "capture_manifest.jsonl")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"],
        )
        monitor = PlaywrightMonitor()
        print(f"Loading Strudel: {STRUDEL_LOCAL_URL}")
        page = await make_page(browser, monitor)

        with open(manifest_path, "w", encoding="utf-8") as manifest_f:
            for input_file in input_files:
                page = await process_file(browser, page, monitor, input_file, manifest_f)

        await browser.close()

    print(f"\nDone. Manifest written to {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
