
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("data/paired")


def load_split(split):
    path = ROOT / f"{split}.jsonl"

    with path.open() as f:
        return [json.loads(line) for line in f]


train = load_split("train")
test = load_split("test")

positions = defaultdict(list)

for sample in train:
    positions[sample["task"]].append(sample["target_xy"])

task_means = {
    task: np.mean(coords, axis=0)
    for task, coords in positions.items()
}

errors = defaultdict(list)

for sample in test:
    prediction = task_means[sample["task"]]
    target = np.array(sample["target_xy"])

    error = np.linalg.norm(prediction - target)
    errors[sample["task"]].append(error)

print("Task-only baseline results:\n")

for task, values in sorted(errors.items()):
    print(f"{task}: mean error = {np.mean(values):.4f}")

all_errors = [e for values in errors.values() for e in values]

print(f"\nOverall mean error: {np.mean(all_errors):.4f}")
print(f"Overall median error: {np.median(all_errors):.4f}")
