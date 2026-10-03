#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parents[1]
path = HERE / "statistics" / "ordinary10_clean_5seed_seed_level.csv"
with path.open(newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 5
hard = np.array([float(r["hard_top1"]) for r in rows])
random = np.array([float(r["random_top1"]) for r in rows])
delta = hard - random
ci = stats.t.interval(0.95, len(delta)-1, loc=float(delta.mean()), scale=stats.sem(delta))
out = {
    "n": 5,
    "hard_mean": float(hard.mean()),
    "random_mean": float(random.mean()),
    "delta_mean": float(delta.mean()),
    "delta_values": [float(x) for x in delta],
    "ci95": [float(ci[0]), float(ci[1])],
    "p_value": float(stats.ttest_1samp(delta, 0.0).pvalue),
    "all_five_deltas_negative": bool(np.all(delta < 0)),
}
assert round(out["delta_mean"], 2) == -14.87
assert out["all_five_deltas_negative"]
print(json.dumps(out, indent=2))
