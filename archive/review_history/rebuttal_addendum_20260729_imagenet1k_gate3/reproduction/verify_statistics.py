#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]


def calc(hard, random):
    deltas = np.asarray(hard, dtype=float) - np.asarray(random, dtype=float)
    mean = float(deltas.mean())
    se = float(stats.sem(deltas))
    ci = stats.t.interval(0.95, len(deltas) - 1, loc=mean, scale=se)
    _, p = stats.ttest_1samp(deltas, 0.0)
    return mean, [float(ci[0]), float(ci[1])], float(p), deltas


rows = list(csv.DictReader((ROOT / "SEED_LEVEL_RESULTS.csv").open()))
assert [int(row["seed"]) for row in rows] == list(range(5))
hard = [float(row["proxy_hard_top1_best"]) for row in rows]
random = [float(row["random_top1_best"]) for row in rows]
mean, ci, p, deltas = calc(hard, random)
expected = json.loads((ROOT / "FINAL_STATISTICS.json").read_text())
reported = expected["best_checkpoint"]["top1"]
assert np.isclose(mean, reported["paired_delta_mean"], atol=1e-10)
assert np.allclose(ci, reported["paired_95_ci"], atol=1e-10)
assert np.isclose(p, reported["paired_t_p_value"], atol=1e-12)
assert np.all(deltas < 0)
print("CPU_STATISTICAL_VERIFICATION=PASS")
print(f"PAIRED_DELTA_TOP1_PP={mean:.10f}")
print(f"PAIRED_95_CI=[{ci[0]:.10f},{ci[1]:.10f}]")
print(f"PAIRED_P_VALUE={p:.12g}")
