# Strudel train/valid split + headless validation pipeline

This toolkit does four things in one flow:

1. Reads an input JSONL dataset
2. Splits it into `train` and `valid` with a deterministic, dimension-aware split
3. Executes each code snippet with headless Strudel validation
4. Writes cleaned train/valid JSONL files plus detailed reports

It works with both of these dataset styles:

- generic instruction JSONL where code is in `output`
- OpenAI chat JSONL where code is the final assistant message in `messages`

## Install

```bash
npm install
```

## 1) Validate an existing dataset without splitting

```bash
node strudel_headless_validate.mjs input.jsonl reports/validation.json
```

## 2) Split only

```bash
node strudel_split_jsonl.mjs input.jsonl out/split --train-ratio 0.9 --seed 42
```

Outputs:

- `train_raw.jsonl`
- `valid_raw.jsonl`
- `split_stats.json`
- `split_stats.csv`

## 3) Full pipeline: split + validate + clean

```bash
node strudel_pipeline.mjs input.jsonl out/run1 --train-ratio 0.9 --seed 42
```

If you want the clean splits to exclude warning-producing samples too:

```bash
node strudel_pipeline.mjs input.jsonl out/run1 --train-ratio 0.9 --seed 42 --drop-warnings
```

## Output structure

```text
out/run1/
  train_raw.jsonl
  valid_raw.jsonl
  train_clean.jsonl
  valid_clean.jsonl
  train_rejected_fail.jsonl
  valid_rejected_fail.jsonl
  train_rejected_warn.jsonl
  valid_rejected_warn.jsonl
  reports/
    split_stats.json
    split_stats.csv
    train_validation.json
    train_validation.csv
    train_failures.csv
    train_warnings.csv
    valid_validation.json
    valid_validation.csv
    valid_failures.csv
    valid_warnings.csv
    summary.json
```

## What counts as a passing sample

Each snippet is:

1. evaluated with Strudel
2. checked to ensure it returns a `Pattern`
3. queried with:
   - `pattern.queryArc(0, 1)`
   - `pattern.queryArc(0, 2)`

A record is marked as failed if evaluation throws, the result is not a pattern, or `queryArc` throws.

Warnings are captured from `console.warn` and `console.error` during evaluation. By default, warning samples remain in `*_clean.jsonl`. Use `--drop-warnings` to exclude them.

## Notes on splitting

The split is deterministic and grouped by `Dimension` where available.

Dimension is detected in this order:

1. `record.dimension`
2. `Dimension:` line inside `record.input`
3. `Dimension:` line inside the user message for chat-format datasets

## OpenAI fine-tuning note

If you use the OpenAI chat-format dataset, keep `train` and `valid` as separate JSONL files. OpenAI's SFT docs specify JSONL with one complete JSON object per line, chat-format messages, and separate train/validation files when you want validation metrics.

