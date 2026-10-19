"""
training_callback.py

CompositeEvalCallback: hooks into SFTTrainer's on_evaluate (fires once per
epoch given eval_strategy="epoch"). At each epoch it:
  1. Generates completions for the validation set with the CURRENT model
     (greedy decoding, for a fair apples-to-apples comparison across epochs).
  2. Writes those completions to disk as JSONL.
  3. Calls epoch_syntax_check.mjs (Node, @strudel/transpiler) via subprocess.
  4. Calls epoch_execution_check.py (Playwright, no audio) via subprocess.
  5. Calls epoch_hap_check.mjs (Node, queries Haps directly -- NO audio
     rendering, ~2ms/item) via subprocess, scored against reference Haps
     computed ONCE at __init__ time from val_raw's own reference code
     (no separate reference file needed).
  6. Calls epoch_audio_check.py (Playwright, WITH audio -- reuses
     t9_audio_capture.py's capture pipeline) via subprocess, then scores the
     captured audio against a pre-loaded, cached reference manifest using
     the SAME Pearson-correlation + SSIM + Onset-F1 functions as
     mel_similarity_score.py.
  7. Computes text similarity in-process (difflib.SequenceMatcher).
  8. Combines all of it into "full_composite" using the 4-category equal
     weighting you specified (25% each): Syntax, Execution, Text similarity,
     and Similarity -- where Similarity = mean(Mel Pearson, Mel SSIM,
     Onset F1, Hap F1), since those four are sub-views of "does this sound/
     structurally resemble the reference" rather than 4 independent
     top-level signals. Flag me if you'd rather treat any of them as its
     own separately-weighted category instead of folding into this one slot.
  9. Injects `full_composite` into the metrics dict Trainer already tracks,
     so metric_for_best_model="full_composite" + greater_is_better=True lets
     Trainer's own best-checkpoint bookkeeping (and EarlyStoppingCallback)
     use it directly.
  10. Deletes that epoch's captured .wav files after scoring (keeps all
     .jsonl manifests/logs) to avoid accumulating gigabytes of audio across
     many epochs. Set cleanup_audio=False in the constructor if you want to
     keep a specific run's audio around for spot-checking.
  11. Appends a row to epoch_ranking_log.jsonl with all the sub-scores.

** TIMING: THIS IS THE EXPENSIVE VERSION -- but the Hap step adds almost
nothing to it (~2ms x 150 items ~ 0.3s), the audio step is still what
dominates. **
You confirmed you want audio evaluation on every epoch despite the cost --
each epoch now takes roughly training-time (~1.3 min for qwen05 per your
logs) + syntax/execution/hap checks (a few minutes combined) + ~22-25 min
for audio capture over 150 items. Budget accordingly: with EPOCHS=20 (or
however many early stopping lets run), total wall-clock time could run into
several hours. Strongly recommend running 1-2 epochs first (see wiring note
below) before committing to the full run.

WIRING INTO YOUR NOTEBOOK (see cell 6, "Build trainer"):

    from training_callback import CompositeEvalCallback

    composite_cb = CompositeEvalCallback(
        val_raw=val_ds_raw,          # RAW (pre-fmt) val dataset -- see note below
        tokenizer=tokenizer,
        model_name=cfg["name"],
        outdir=OUTDIR,
        ref_manifest_path="audio_capture/reference/test_set_150_finetuned/reference_manifest.jsonl",
    )

    trainer = SFTTrainer(
        ...,
        args=SFTConfig(
            ...,
            metric_for_best_model="full_composite",   # was "eval_loss"
            greater_is_better=True,                     # was False
            load_best_model_at_end=True,                 # new
            save_total_limit=None,                        # keep all epoch
                                                           # checkpoints -- LoRA
                                                           # adapters are tiny
        ),
        callbacks=[
            composite_cb,                                # MUST come first -- it injects
                                                           # eval_full_composite into the
                                                           # metrics dict; EarlyStoppingCallback
                                                           # only sees keys added by callbacks
                                                           # that ran before it in this list.
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1e-4),
        ],
    )

    # FOR A QUICK 1-2 EPOCH DRY RUN FIRST (strongly recommended):
    # temporarily set EPOCHS = 2 in the config cell, rerun trainer.train(),
    # check the printed per-epoch breakdown makes sense, THEN set EPOCHS
    # back to 20 and rerun for the full training run.

IMPORTANT NOTES:
- val_raw must be the dataset BEFORE `.map(fmt)` is applied (raw `messages`),
  same as before -- keep a separate reference to it.
- ref_manifest_path should point at the FINE-TUNE val set's reference audio
  manifest (test_set_150_finetuned), i.e. the one you rendered with
  render_reference_audio.py earlier -- NOT the few-shot/test_set_150_new one.
- The reference spectrograms are loaded/cached ONCE (via mel_similarity_score's
  own module-level cache) on the first epoch and reused every subsequent
  epoch, so only the CANDIDATE audio capture cost repeats each epoch.
  Reference Haps are similarly computed ONCE, in __init__, from val_raw
  directly (no separate file needed for this one).
- Requires the Strudel dev server (localhost:4321) to stay running for the
  ENTIRE training run now, not just for a one-off validation step.
- epoch_hap_check.mjs AND hap_query_bundle.mjs (built from hap_query_entry.mjs,
  see that file's docstring -- needs @strudel/tonal installed too) must both
  be present in NODE_PROJECT_DIR alongside epoch_syntax_check.mjs.

** BEFORE wiring this into the notebook, fill in the PATH CONFIG block above
(SCRIPTS_DIR default should be fine, but NODE_PROJECT_DIR you MUST set), then
verify each subprocess script runs standalone from a plain terminal --
don't debug this for the first time inside a 20-epoch training run:

    # 1. put epoch_syntax_check.mjs and epoch_hap_check.mjs (+ hap_query_bundle.mjs)
    #    INSIDE your existing Strudel/node project folder (wherever
    #    n-shot-avg.mjs already runs successfully), NOT next to the other .py files.
    cd <that node project folder>
    node epoch_syntax_check.mjs some_small_test.jsonl out_manifest.jsonl
    node epoch_hap_check.mjs some_small_test.jsonl out_hap_manifest.jsonl

    # 2. put t9_audio_capture.py, epoch_execution_check.py, epoch_audio_check.py,
    #    mel_similarity_score.py, and training_callback.py together in one
    #    folder (SCRIPTS_DIR), then from THAT folder:
    python epoch_execution_check.py --input some_small_test.jsonl --output_manifest out.jsonl
    python epoch_audio_check.py --input some_small_test.jsonl --out_dir out_audio --out_manifest out_audio.jsonl

If all of those run clean on their own, the callback's subprocess calls
(which now pass explicit cwd= for each one) should work the same way from
inside Jupyter. If `node` isn't found, set NODE_BIN to an absolute path
(same fix as FFMPEG_PATH in t9_audio_capture.py -- run `Get-Command node`
in a terminal where `node --version` already works, in PowerShell).
"""

import json
import os
import shutil
import subprocess
import sys
from difflib import SequenceMatcher

import torch
from transformers import TrainerCallback

# Explicitly ensure this file's own directory is importable BEFORE importing
# mel_similarity_score -- don't rely on the notebook's cwd/sys.path already
# including it, which is exactly what broke on a fresh Jupyter kernel.
_SCRIPTS_DIR_FOR_IMPORT = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR_FOR_IMPORT)

from mel_similarity_score import (
    load_manifest, get_resized_spec, cosine_sim, ssim_sim, is_degenerate, onset_f1, hap_f1,
)

# ================= PATH CONFIG -- fill these in, don't assume =================
# All the PYTHON helper scripts (this file, epoch_execution_check.py,
# epoch_audio_check.py, mel_similarity_score.py, t9_audio_capture.py) need to
# live TOGETHER in one folder, because epoch_audio_check.py does
# `from t9_audio_capture import ...` -- that import only resolves if the
# subprocess's cwd is this folder. Defaults to wherever THIS file lives.

# The .mjs script is DIFFERENT: Node resolves `@strudel/transpiler` by walking
# up from the .mjs file's own location looking for node_modules, NOT from any
# Python cwd. It must live in (or under) whatever project directory already
# has `@strudel/transpiler` installed -- i.e. wherever n-shot-avg.mjs already
# runs successfully today. Point this at that directory; do NOT assume it's
# the same as SCRIPTS_DIR or your notebook's folder.
SCRIPTS_DIR = _SCRIPTS_DIR_FOR_IMPORT
NODE_PROJECT_DIR = os.environ.get("STRUDEL_NODE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "node"))  # override with env STRUDEL_NODE_DIR

NODE_BIN = "node"  # if `node` isn't reliably on PATH from Jupyter's kernel
                    # process, replace with an absolute path to node.exe,
                    # same fix as FFMPEG_PATH in t9_audio_capture.py --
                    # find it via `Get-Command node` in a terminal where
                    # `node --version` already works.

NODE_SCRIPT_PATH = os.path.join(NODE_PROJECT_DIR, "epoch_syntax_check.mjs")
HAP_SCRIPT_PATH = os.path.join(NODE_PROJECT_DIR, "epoch_hap_check.mjs")  # also needs
                                                                          # hap_query_bundle.mjs
                                                                          # sitting next to it
EXEC_SCRIPT_PATH = os.path.join(SCRIPTS_DIR, "epoch_execution_check.py")
AUDIO_SCRIPT_PATH = os.path.join(SCRIPTS_DIR, "epoch_audio_check.py")
# ================================================================================

MAX_NEW_TOKENS = 200


def _build_prompt_messages(messages, model_name):
    """Mirrors the gemma system-merge logic in the notebook's fmt(), so
    generation prompts match how the model was actually trained."""
    prompt_msgs = [m for m in messages if m["role"] != "assistant"]
    if "gemma-2" in model_name.lower():
        sys_c = ""
        for m in prompt_msgs:
            if m["role"] == "system":
                sys_c = m["content"]
                break
        merged = []
        first_user_done = False
        for m in prompt_msgs:
            if m["role"] == "system":
                continue
            if m["role"] == "user" and not first_user_done:
                content = (sys_c + "\n\n" + m["content"]) if sys_c else m["content"]
                merged.append({"role": "user", "content": content})
                first_user_done = True
            else:
                merged.append(m)
        prompt_msgs = merged
    return prompt_msgs


class CompositeEvalCallback(TrainerCallback):
    def __init__(self, val_raw, tokenizer, model_name, outdir, ref_manifest_path,
                 max_new_tokens=MAX_NEW_TOKENS, cleanup_audio=True):
        self.val_raw = val_raw
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.outdir = os.path.abspath(outdir)  # absolute -- avoids notebook-kernel-cwd
                                                # vs subprocess-cwd path mismatches
        self.max_new_tokens = max_new_tokens
        self.cleanup_audio = cleanup_audio  # delete each epoch's .wav files after
                                             # scoring (keeps all .jsonl logs/manifests) --
                                             # set False if you want to spot-check a
                                             # specific epoch's audio afterward.
        self.eval_dir = os.path.join(self.outdir, "epoch_eval")
        os.makedirs(self.eval_dir, exist_ok=True)
        self.ranking_log_path = os.path.join(self.eval_dir, "epoch_ranking_log.jsonl")
        self._warned_inference_toggle = False

        ref_entries = load_manifest(ref_manifest_path)
        self.ref_by_index = {
            e["entry_index"]: e["wav_path"] for e in ref_entries
            if e["status"] == "SUCCESS" and e.get("wav_path")
        }
        print(f"[CompositeEvalCallback] loaded {len(self.ref_by_index)} reference audios from {ref_manifest_path}")

        self.ref_haps_by_index = self._build_reference_haps()
        print(f"[CompositeEvalCallback] built {len(self.ref_haps_by_index)} reference Hap lists from val_raw "
              f"(cheap, ~2ms/item -- no audio rendering involved)")

    def _build_reference_haps(self):
        """val_raw already carries the reference code at messages[-1] for
        every item (that's what _generate_completions compares generations
        against), so we reuse it directly instead of needing a separate
        reference file -- Hap queries are cheap enough (~2ms/item) to just
        compute once here rather than pre-rendering anything."""
        ref_completions_path = os.path.join(self.eval_dir, "_reference_for_hap.jsonl")
        with open(ref_completions_path, "w", encoding="utf-8") as f:
            for ex in self.val_raw:
                f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

        ref_manifest_path = os.path.join(self.eval_dir, "_reference_hap_manifest.jsonl")
        result = subprocess.run(
            [NODE_BIN, HAP_SCRIPT_PATH, ref_completions_path, ref_manifest_path],
            capture_output=True, text=True, cwd=NODE_PROJECT_DIR,
        )
        if "HAP_QUERY_DONE=" not in result.stdout:
            print(f"[CompositeEvalCallback] WARNING: reference hap query failed.\n"
                  f"stdout: {result.stdout}\nstderr: {result.stderr}")
            return {}

        ref_hap_entries = load_manifest(ref_manifest_path)
        return {
            e["entry_index"]: e["haps"] for e in ref_hap_entries
            if e["status"] == "SUCCESS" and e.get("haps")
        }

        # Fail loudly HERE, before training starts, rather than hours into a
        # run -- if these paths are wrong (e.g. relative paths that resolved
        # fine wherever render_reference_audio.py was originally run, but
        # don't resolve the same way from this notebook's cwd), you want to
        # know now, not after epoch 1's ~25 minutes of audio capture.
        missing = [(idx, p) for idx, p in self.ref_by_index.items() if not os.path.isfile(p)]
        if missing:
            sample = missing[:3]
            raise FileNotFoundError(
                f"[CompositeEvalCallback] {len(missing)}/{len(self.ref_by_index)} reference wav paths "
                f"from {ref_manifest_path} don't exist from this notebook's cwd ({os.getcwd()}). "
                f"Example missing paths: {sample}. "
                f"These were likely written as relative paths by render_reference_audio.py when run "
                f"from a different working directory. Easiest fix: pass an ABSOLUTE ref_manifest_path, "
                f"and if that manifest's wav_path entries are themselves relative, either re-run "
                f"render_reference_audio.py from this same notebook's cwd, or tell me the exact "
                f"original wav_path format and I'll add a proper resolution rule instead of guessing."
            )

    def _generate_completions(self, model):
        # The model's own generation_config.json often ships with a default
        # max_length (e.g. 32768) that conflicts with the max_new_tokens we
        # pass below -- harmless (max_new_tokens wins either way) but noisy,
        # since it'd otherwise print once per item (150x per epoch). Clearing
        # it here removes the conflict at the source instead of blanket-
        # suppressing all transformers warnings.
        model.generation_config.max_length = None

        used_toggle = False
        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
            used_toggle = True
        except Exception as e:
            if not self._warned_inference_toggle:
                print(f"[CompositeEvalCallback] note: FastLanguageModel.for_inference() unavailable/failed ({e}); "
                      f"generating without it.")
                self._warned_inference_toggle = True

        model.eval()
        completions = []
        with torch.no_grad():
            for ex in self.val_raw:
                messages = ex["messages"]
                reference_code = messages[-1]["content"]
                prompt_msgs = _build_prompt_messages(messages, self.model_name)
                text = self.tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                gen_text = self.tokenizer.decode(
                    out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                completions.append({
                    "messages": [messages[0], messages[1], {"role": "assistant", "content": gen_text}],
                    "reference_code": reference_code,
                })

        if used_toggle:
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_training(model)
            except Exception:
                pass
        model.train()
        return completions

    def _text_similarity(self, completions):
        scores = [
            SequenceMatcher(None, c["messages"][-1]["content"], c["reference_code"]).ratio()
            for c in completions
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def _run_syntax_check(self, completions_path, manifest_path):
        result = subprocess.run(
            [NODE_BIN, NODE_SCRIPT_PATH, completions_path, manifest_path],
            capture_output=True, text=True, cwd=NODE_PROJECT_DIR,
        )
        for line in result.stdout.splitlines():
            if line.startswith("SYNTAX_PASS_RATE="):
                return float(line.split("=", 1)[1])
        print(f"[CompositeEvalCallback] WARNING: syntax check produced no parseable rate.\n"
              f"stdout: {result.stdout}\nstderr: {result.stderr}")
        return 0.0

    def _run_execution_check(self, completions_path, manifest_path):
        result = subprocess.run(
            [sys.executable, EXEC_SCRIPT_PATH, "--input", completions_path, "--output_manifest", manifest_path],
            capture_output=True, text=True, cwd=SCRIPTS_DIR,
        )
        for line in result.stdout.splitlines():
            if line.startswith("EXECUTION_PASS_RATE="):
                return float(line.split("=", 1)[1])
        print(f"[CompositeEvalCallback] WARNING: execution check produced no parseable rate.\n"
              f"stdout: {result.stdout}\nstderr: {result.stderr}")
        return 0.0

    def _run_audio_check(self, completions_path, audio_dir, manifest_path):
        result = subprocess.run(
            [sys.executable, AUDIO_SCRIPT_PATH, "--input", completions_path,
             "--out_dir", audio_dir, "--out_manifest", manifest_path],
            capture_output=True, text=True, cwd=SCRIPTS_DIR,
        )
        if "AUDIO_CAPTURE_DONE=" not in result.stdout:
            print(f"[CompositeEvalCallback] WARNING: audio check did not finish cleanly.\n"
                  f"stdout: {result.stdout}\nstderr: {result.stderr}")
        return manifest_path

    def _score_audio(self, audio_manifest_path):
        """Returns (pearson_avg, ssim_avg, onset_avg), all on a 0-1 scale,
        averaged over all 150 entries (failures/degenerate candidates score
        0 for all three, same convention as mel_similarity_score.py)."""
        entries = load_manifest(audio_manifest_path)
        pearson_scores, ssim_scores, onset_scores = [], [], []
        for e in entries:
            idx = e["entry_index"]
            if e["status"] != "SUCCESS" or not e.get("wav_path") or idx not in self.ref_by_index:
                pearson_scores.append(0.0)
                ssim_scores.append(0.0)
                onset_scores.append(0.0)
                continue
            try:
                cand_spec = get_resized_spec(e["wav_path"])
                ref_spec = get_resized_spec(self.ref_by_index[idx])
                if is_degenerate(cand_spec) or is_degenerate(ref_spec):
                    pearson_scores.append(0.0)
                    ssim_scores.append(0.0)
                    onset_scores.append(0.0)
                else:
                    pearson_scores.append(max(0.0, cosine_sim(cand_spec, ref_spec)))
                    ssim_scores.append(ssim_sim(cand_spec, ref_spec))
                    onset_scores.append(onset_f1(e["wav_path"], self.ref_by_index[idx]))
            except Exception:
                pearson_scores.append(0.0)
                ssim_scores.append(0.0)
                onset_scores.append(0.0)

        pearson_avg = sum(pearson_scores) / len(pearson_scores) if pearson_scores else 0.0
        ssim_avg = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0.0
        onset_avg = sum(onset_scores) / len(onset_scores) if onset_scores else 0.0
        return pearson_avg, ssim_avg, onset_avg

    def _run_hap_check(self, completions_path, manifest_path):
        """Queries Haps directly from the code (no audio rendering) --
        ~2ms/item, so this adds negligible time to each epoch compared to
        the audio step."""
        result = subprocess.run(
            [NODE_BIN, HAP_SCRIPT_PATH, completions_path, manifest_path],
            capture_output=True, text=True, cwd=NODE_PROJECT_DIR,
        )
        if "HAP_QUERY_DONE=" not in result.stdout:
            print(f"[CompositeEvalCallback] WARNING: hap check did not finish cleanly.\n"
                  f"stdout: {result.stdout}\nstderr: {result.stderr}")
        return manifest_path

    def _score_hap(self, hap_manifest_path):
        entries = load_manifest(hap_manifest_path)
        scores = []
        for e in entries:
            idx = e["entry_index"]
            if e["status"] != "SUCCESS" or not e.get("haps") or idx not in self.ref_haps_by_index:
                scores.append(0.0)
                continue
            try:
                scores.append(hap_f1(e["haps"], self.ref_haps_by_index[idx]))
            except Exception:
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs["model"]
        epoch = state.epoch

        epoch_dir = os.path.join(self.eval_dir, f"epoch_{epoch:.2f}")
        os.makedirs(epoch_dir, exist_ok=True)
        completions_path = os.path.join(epoch_dir, "completions.jsonl")
        syntax_manifest_path = os.path.join(epoch_dir, "syntax_manifest.jsonl")
        exec_manifest_path = os.path.join(epoch_dir, "execution_manifest.jsonl")
        hap_manifest_path = os.path.join(epoch_dir, "hap_manifest.jsonl")
        audio_dir = os.path.join(epoch_dir, "audio")
        audio_manifest_path = os.path.join(epoch_dir, "audio_manifest.jsonl")

        print(f"\n[CompositeEvalCallback] epoch {epoch:.2f}: generating {len(self.val_raw)} completions...")
        completions = self._generate_completions(model)
        with open(completions_path, "w", encoding="utf-8") as f:
            for c in completions:
                f.write(json.dumps({"messages": c["messages"]}, ensure_ascii=False) + "\n")

        text_sim = self._text_similarity(completions)

        print(f"[CompositeEvalCallback] epoch {epoch:.2f}: running syntax check (node)...")
        syntax_rate = self._run_syntax_check(completions_path, syntax_manifest_path)

        print(f"[CompositeEvalCallback] epoch {epoch:.2f}: running execution check (playwright)...")
        exec_rate = self._run_execution_check(completions_path, exec_manifest_path)

        print(f"[CompositeEvalCallback] epoch {epoch:.2f}: running hap query (node, no audio -- fast)...")
        self._run_hap_check(completions_path, hap_manifest_path)
        hap_avg = self._score_hap(hap_manifest_path)

        print(f"[CompositeEvalCallback] epoch {epoch:.2f}: running audio capture + scoring "
              f"(playwright, ~20+ min for 150 items)...")
        self._run_audio_check(completions_path, audio_dir, audio_manifest_path)
        pearson_avg, ssim_avg, onset_avg = self._score_audio(audio_manifest_path)
        similarity_avg = (0.1*pearson_avg + 0.1*ssim_avg + 0.1*onset_avg + 0.7*hap_avg)

        if self.cleanup_audio:
            try:
                shutil.rmtree(audio_dir, ignore_errors=False)
                print(f"[CompositeEvalCallback] epoch {epoch:.2f}: deleted {audio_dir} "
                      f"(wav files) to save disk space -- manifests/logs kept.")
            except Exception as e:
                # Don't let cleanup failure (e.g. a file still locked by an
                # antivirus scan on Windows) kill the training run.
                print(f"[CompositeEvalCallback] WARNING: could not delete {audio_dir}: {e}")

        full_composite = (0.1*syntax_rate + 0.4*exec_rate + 0.1*text_sim + 0.4*similarity_avg)

        print(f"[CompositeEvalCallback] epoch {epoch:.2f}: "
              f"syntax={syntax_rate:.4f} execution={exec_rate:.4f} text_sim={text_sim:.4f} "
              f"mel_pearson={pearson_avg:.4f} mel_ssim={ssim_avg:.4f} onset_f1={onset_avg:.4f} "
              f"hap_f1={hap_avg:.4f} similarity_avg={similarity_avg:.4f} -> full_composite={full_composite:.4f}")

        if metrics is not None:
            metrics["eval_full_composite"] = full_composite
            metrics["eval_syntax_pass_rate"] = syntax_rate
            metrics["eval_execution_pass_rate"] = exec_rate
            metrics["eval_text_similarity"] = text_sim
            metrics["eval_mel_pearson"] = pearson_avg
            metrics["eval_mel_ssim"] = ssim_avg
            metrics["eval_onset_f1"] = onset_avg
            metrics["eval_hap_f1"] = hap_avg

        with open(self.ranking_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "global_step": state.global_step,
                "syntax_pass_rate": syntax_rate,
                "execution_pass_rate": exec_rate,
                "text_similarity": text_sim,
                "mel_pearson": pearson_avg,
                "mel_ssim": ssim_avg,
                "onset_f1": onset_avg,
                "hap_f1": hap_avg,
                "full_composite": full_composite,
                "completions_path": completions_path,
            }, ensure_ascii=False) + "\n")

        return control
