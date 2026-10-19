import json
import re
import glob
import os
from collections import defaultdict
from difflib import SequenceMatcher

GROUND_TRUTH_FILE = "test_set_150.jsonl"
INPUT_DIR = "./finetuned-0-shot"

def extract_functions(code_str):
    return set(re.findall(r'([a-zA-Z0-9_]+)\(', code_str))

def calculate_jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# The Task line is the matching key (the ablation removes the Reference function line from the prompt,
# so the whole prompt no longer matches the ground truth, but the Task line is identical in groups A and B).
def get_task(content):
    for ln in content.split('\n'):
        m = re.match(r'\s*Task:\s*(.*?)\s*$', ln)
        if m:
            return m.group(1).strip()
    return None

def evaluate_file(generated_file, ground_truth_map):
    total_samples = 0
    hit_count = 0
    total_jaccard = 0.0
    total_str_sim = 0.0

    with open(generated_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            prompt = data['messages'][1]['content']
            gen_code = data['messages'][2]['content']

            # Look up the ground truth by the Task line (instead of matching the whole prompt)
            task = get_task(prompt)
            if task not in ground_truth_map:
                continue

            total_samples += 1
            gt_data = ground_truth_map[task]
            gt_code = gt_data["code"]
            # Take F from the ref_function field of the ablation output if present, otherwise from the ground truth
            ref_func = data.get("ref_function") or gt_data["ref_func"]

            if ref_func in gen_code:
                hit_count += 1

            gt_funcs = extract_functions(gt_code)
            gen_funcs = extract_functions(gen_code)
            total_jaccard += calculate_jaccard(gt_funcs, gen_funcs)
            total_str_sim += string_similarity(gt_code, gen_code)

    return total_samples, hit_count, total_jaccard, total_str_sim

def extract_model_name(filename):
    m = re.match(r"(?:finetuned_baseline|ablation)_(.+)_0shot_seed\d+\.jsonl", os.path.basename(filename))
    return m.group(1) if m else filename


def main():
    ground_truth_map = {}
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            prompt = data['messages'][1]['content']
            gt_code = data['messages'][2]['content']
            ref_match = re.search(r'Reference function:\s*([^\n]+)', prompt)
            ref_func = ref_match.group(1).strip() if ref_match else ""
            task = get_task(prompt)
            ground_truth_map[task] = {"code": gt_code, "ref_func": ref_func}

    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_0shot_seed*.jsonl")))
    if not input_files:
        print(f"No files found in {INPUT_DIR}")
        return

    # model -> list of per-seed metric dicts
    model_metrics = defaultdict(list)

    for generated_file in input_files:
        fname = os.path.basename(generated_file)
        model = extract_model_name(generated_file)
        total_samples, hit_count, total_jaccard, total_str_sim = evaluate_file(generated_file, ground_truth_map)

        if total_samples == 0:
            print(f"[{fname}] warning: no matching prompts found, skipped.")
            continue

        model_metrics[model].append({
            "hit_rate": hit_count / total_samples,
            "str_sim": total_str_sim / total_samples,
        })

    print("\n" + "=" * 75)
    print(f"{'Model':<45} {'Hit Rate (avg)':>14} {'Str Sim (avg)':>13}")
    print("=" * 75)
    for model, metrics in sorted(model_metrics.items()):
        avg_hit = sum(m["hit_rate"] for m in metrics) / len(metrics)
        avg_sim = sum(m["str_sim"] for m in metrics) / len(metrics)
        seeds_hit = "[" + ", ".join(f"{m['hit_rate']:.4f}" for m in metrics) + "]"
        seeds_sim = "[" + ", ".join(f"{m['str_sim']:.4f}" for m in metrics) + "]"
        print(f"{model:<45} {avg_hit:>13.4f}  {avg_sim:>12.4f}")
        print(f"  hit/seed : {seeds_hit}")
        print(f"  sim/seed : {seeds_sim}")
    print("=" * 75)

if __name__ == "__main__":
    main()
