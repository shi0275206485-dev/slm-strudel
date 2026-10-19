import json
import random
from collections import defaultdict
import re

# ================= Configuration =================
INPUT_DATASET = "../data/cleaned/train_only.jsonl"
TEST_SET_OUTPUT = "../data/cleaned/test_set_150_finetuned.jsonl"
TRAIN_SET_OUTPUT = "../data/cleaned/train_set_finetuned.jsonl"

def main():
    all_data = []
    with open(INPUT_DATASET, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

    # 1. Extract dimensions and merge labels (the generator produced fragmented label variants)
    grouped_data = defaultdict(list)
    for row in all_data:
        user_content = row['messages'][1]['content']
        dim_match = re.search(r'Dimension:\s*(.+)', user_content)
        dim = dim_match.group(1).strip() if dim_match else "Unknown"
        
        # Normalise and merge near-identical dimension labels
        if "Melody" in dim or "Harmony" in dim:
            dim = "Melody and Harmony"
        elif "Sound Shaping" in dim or "DSP" in dim:
            dim = "Sound Shaping / DSP"
        elif "Sampling" in dim or "Synthesis" in dim:
            dim = "Sampling and Synthesis"
        elif "Time" in dim:
            dim = "Time Transformation"
        elif "Structure" in dim or "Multitrack" in dim or "Advanced" in dim:
            dim = "Structure & Multitrack"
        elif "Rhythm" in dim:
            dim = "Basic Rhythm"
            
        grouped_data[dim].append(row)

    # 2. Stratified random sampling
    target_total = 150
    categories = list(grouped_data.keys())
    target_per_cat = target_total // len(categories)

    test_set = []
    train_set = []
    allocation = {k: 0 for k in categories}
    remaining = target_total

    # Round 1: base quota per dimension
    for k in categories:
        take = min(target_per_cat, len(grouped_data[k]))
        allocation[k] = take
        remaining -= take

    # Round 2: redistribute the remainder (if a dimension has too few items, its unused quota goes to the others)
    while remaining > 0:
        for k in categories:
            if allocation[k] < len(grouped_data[k]) and remaining > 0:
                allocation[k] += 1
                remaining -= 1

    # 3. Shuffle and split
    random.seed(42)  # fixed seed: every run produces exactly the same split
    for k in categories:
        random.shuffle(grouped_data[k])
        test_set.extend(grouped_data[k][:allocation[k]])
        train_set.extend(grouped_data[k][allocation[k]:])

    # Shuffle the held-out set once more so its order carries no dimension blocks
    random.shuffle(test_set)

    # 4. Write the two disjoint files
    with open(TEST_SET_OUTPUT, 'w', encoding='utf-8') as f:
        for row in test_set:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    with open(TRAIN_SET_OUTPUT, 'w', encoding='utf-8') as f:
        for row in train_set:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f"Split complete.")
    print(f"Held-out set: {len(test_set)} items -> {TEST_SET_OUTPUT}")
    print(f"Training set: {len(train_set)} items -> {TRAIN_SET_OUTPUT}")

if __name__ == "__main__":
    main()