#!/usr/bin/env python3
"""Reproduce DINOv2 multi-budget paired statistics from sanitized values."""

from pathlib import Path
import csv
import json

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent


def paired_deltas(rows, experiment, budget):
    values = [
        float(row["paired_delta"])
        for row in rows
        if row["experiment"] == experiment
        and float(row["budget"]) == budget
        and row["condition"] == "hard"
        and row["paired_delta"].strip()
    ]
    return np.asarray(values, dtype=float)


def summarize(values):
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=stats.sem(values))
    pvalue = float(stats.ttest_1samp(values, 0.0).pvalue)
    return {
        "n": int(values.size),
        "mean_delta_pp": mean,
        "sample_sd_pp": sd,
        "ci95_pp": [float(ci[0]), float(ci[1])],
        "paired_t_pvalue": pvalue,
        "deltas_pp": [float(x) for x in values],
    }


def main():
    with (HERE / "per_seed_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    result = {"ordinary": {}, "class_balanced": {}}
    for budget in (0.10, 0.30, 0.50):
        result["ordinary"][f"{budget:.2f}"] = summarize(
            paired_deltas(rows, "ordinary", budget)
        )
    result["class_balanced"]["0.10"] = summarize(
        paired_deltas(rows, "class_balanced", 0.10)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
