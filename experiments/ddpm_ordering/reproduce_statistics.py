#!/usr/bin/env python3
"""Reproduce the eight-seed DDPM ordering statistics."""

from pathlib import Path
import csv
import json

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent


def summarize(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = float(stats.sem(values))
    ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=se)
    test = stats.ttest_1samp(values, 0.0)
    return {
        "n": int(values.size),
        "mean_pp": mean,
        "sample_sd_pp": sd,
        "standard_error_pp": se,
        "ci95_pp": [float(ci[0]), float(ci[1])],
        "paired_t_statistic": float(test.statistic),
        "paired_t_pvalue": float(test.pvalue),
        "values_pp": [float(x) for x in values],
    }


def main():
    with (HERE / "per_seed_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("per_seed_results.csv is empty")

    computed = []
    for row_number, row in enumerate(rows, start=2):
        random_top1 = float(row["random_top1"])
        easy_top1 = float(row["easy_top1"])
        shuffled_top1 = float(row["shuffled_top1"])
        expected = {
            "easy_minus_random_pp": easy_top1 - random_top1,
            "shuffled_minus_random_pp": shuffled_top1 - random_top1,
            "easy_minus_shuffled_pp": easy_top1 - shuffled_top1,
        }
        for column, value in expected.items():
            stored = float(row[column])
            if not np.isclose(stored, value, rtol=0.0, atol=1e-9):
                raise SystemExit(
                    f"delta mismatch at CSV row {row_number}, {column}: "
                    f"stored={stored}, computed={value}"
                )
        computed.append(expected)

    result = {
        "easy_minus_random": summarize(
            [item["easy_minus_random_pp"] for item in computed]
        ),
        "shuffled_minus_random": summarize(
            [item["shuffled_minus_random_pp"] for item in computed]
        ),
        "easy_minus_shuffled": summarize(
            [item["easy_minus_shuffled_pp"] for item in computed]
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
