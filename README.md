# Small Language Models for Domain-Specific Code Generation: Strudel

Code and data for the COMPX520 dissertation *Small Language Models for
Domain-Specific Language Code Generation: A Case Study on Strudel*
(Jinchun Shi, University of Waikato, 2026).

The project fine-tunes small language models (Qwen2.5-Coder 0.5B and
Gemma 3 1B) with LoRA so that they generate code in
[Strudel](https://strudel.cc), a browser-based language for live coding music.
It also evaluates the generated code on four axes: syntactic validity,
execution, text similarity and sound similarity. Sound similarity includes
**Hap F1**, which compares the musical events two programs produce by querying
Strudel's pattern engine directly, without rendering audio.

> This is an independent research project. It is not an official Strudel
> project and is not affiliated with the Strudel authors. Strudel is free
> software by the Strudel contributors, licensed under AGPL-3.0; source code:
> <https://codeberg.org/uzu/strudel>.

## Repository layout

```
data/
  cleaned/      corpus after cleaning: 906 train / 150 validation / 150 test, plus few-shot pool
  original/     corpus as first constructed (1,376 + 150); used only by the baseline survey
  README.md     data card: files, schema, provenance, known issues
node/           Node.js tools built on the Strudel packages
                (syntax check, Hap extraction, hit-rate analysis, corpus validation)
python/         generation notebooks, LoRA training, browser execution,
                audio capture, scoring and aggregation
```

`python/` and `node/` mirror the two working folders used during the project.
Python scripts call the Node tools through `subprocess`. They find `node/`
through the environment variable `STRUDEL_NODE_DIR`, which defaults to
`../node` relative to the script.

## Requirements

* **GPU.** All experiments ran on one laptop GPU with 16 GB of memory
  (RTX 5080 Laptop, Windows).
* **Python.** Install with `pip install -r requirements.txt`, then
  `playwright install chromium`. The training run logged these versions:
  torch 2.10.0+cu130, Unsloth 2026.6.9, Transformers 5.5.0, Triton 3.6.0.
  Other package versions were not recorded.
* **ffmpeg** on `PATH`, for audio capture (`t9_audio_capture.py`).
* **Node.js** with npm.
* **A running Strudel development server at `http://localhost:4321`.** Clone
  the Strudel repository and start its website in development mode. Execution
  and audio checks open each program in this page through Playwright.
* **Hugging Face access** to the gated Gemma models. Log in with
  `huggingface-cli login`, or run `notebook_login()` in the notebooks. Never
  write a token into a notebook.

## Setup

### Node tools and the Hap bundle

```bash
cd node
npm ci                                   # exact versions from package-lock.json (@strudel/* 1.2.6)
npx esbuild hap_query_entry.mjs --bundle --format=esm --platform=browser --outfile=hap_query_bundle.mjs
python patch_hap_bundle.py hap_query_bundle.mjs
```

`hap_query_bundle.mjs` is a build product and is not committed. With the
lockfile, these three commands reproduce the bundle used in the dissertation
byte for byte, apart from line endings.

`patch_hap_bundle.py` fixes multi-track (`$:`) programs. Without it, only the
last track is queried. The script is idempotent and keeps a `.bak` file.

### Reference audio

Run once per evaluation set, from `python/`:

```bash
python render_reference_audio.py --ref_file ../data/cleaned/test_set_150_new.jsonl
python render_reference_audio.py --ref_file ../data/cleaned/test_set_150_finetuned.jsonl
```

This writes `audio_capture/reference/<set>/reference_manifest.jsonl`. Both
the training callback and `run_batch_metrics.py` use it.

## Pipeline

All Python commands run from `python/`.

| Step | Script | Notes |
|---|---|---|
| Corpus generation and verification | `node/strudel_pipeline.mjs`, `node/strudel_headless_validate.mjs` | See `node/CORPUS_PIPELINE.md`. |
| Cleaning | — | Done interactively, not scripted: leakage removal, `voicing()` repair, duplicate replacement. Results are in `data/cleaned/`; `data/README.md` describes them. |
| Train / validation split | `split_data.py` | Splits the 1,056-item pool into 906 training and 150 validation items. |
| Base-model generation | `N-Few-shot.ipynb` | Shot counts $K \in \{0,1,3,5,10\}$; the baseline survey also ran $K=11$. 8-way sampling: $K$ demonstrations from each of the eight task dimensions. For the survey, point the two data constants at `data/original/`. |
| Hit-rate ablation (no *Reference function* line) | `N-Few-shot-ab.ipynb`, `node/score_ablation.mjs` | Baseline survey only. |
| LoRA fine-tuning | `train_finetune.ipynb`, `training_callback.py` | The callback scores the validation set every epoch and selects the checkpoint by the composite score. |
| Fine-tuned generation, $K \in \{0,1,3,5,10\}$ | `N-Few-shot-finetuned.ipynb` | Loops over rank × training seed × $K$ × generation seed. |
| Scoring | `run_batch_metrics.py` | Syntax, execution, text similarity, Mel Pearson, Mel SSIM, Onset F1 and Hap F1. Writes `per_file_scores.jsonl` and `summary.csv`. |
| Aggregation over seeds | `aggregate_seed_stability.py` | Stage 1 averages over five generation seeds. Stage 2 gives the mean ± sample SD over three training seeds. |
| Rank ANOVA | `anova_rank.py` | One-way ANOVA across ranks; each trained adapter is one replicate. |
| Baseline-survey measures | `node/batch_all.mjs`, `t9.py`, `evaluate_similarity_v2.py`, `node/final_analysis.mjs` | The four measures of the original framework, including the reference-function funnel. |

Composite weights (W2):

```
sound_sim   = 0.2·MelPearson + 0.1·MelSSIM + 0.2·OnsetF1 + 0.5·HapF1
final_score = 0.1·syntax + 0.5·execution + 0.1·text_sim + 0.3·sound_sim
```

**Seeds.**

* Training seeds: 42, 123 and 2024. Each one is passed to both the trainer
  and the LoRA initialisation.
* Generation seeds: 2, 4, 6, 8 and 10.
* At $K=0$, completions are sampled with temperature 0.7 and top-p 0.9.
* At $K \geq 1$, decoding is greedy, and the generation seed selects which
  demonstrations are drawn.

Several notebooks and scripts keep configuration constants at the top: model
key, ranks, input and output folders. Set these before running.

## Safety

Every evaluation step executes model-generated JavaScript. The execution and
audio checks run it in a browser page; Hap extraction runs it in Node with
`Function()`. A static pre-screen rejects known non-terminating loops and
event explosions (`fast`/`ply` factors above 256, or a cumulative product
above 512), but generated code is otherwise not sandboxed. Run the pipeline
in an isolated environment.

## Licence

This repository is licensed under the GNU Affero General Public License v3.0
(see `LICENSE`). It depends on, and the Hap bundle incorporates, Strudel,
which is AGPL-3.0.

## Citation

```bibtex
@mastersthesis{shi2026strudel,
  author = {Shi, Jinchun},
  title  = {Small Language Models for Domain-Specific Language Code Generation: A Case Study on {Strudel}},
  school = {University of Waikato},
  year   = {2026}
}
```
