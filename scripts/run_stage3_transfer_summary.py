#!/usr/bin/env python3
"""Stage 3: summarize method-vs-Random transfer deltas by budget."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    by_key = defaultdict(list)
    with open(args.transfer, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["method"], row["budget"])
            by_key[key].append(float(row["accuracy"]))

    random_means = {
        budget: sum(vals) / len(vals)
        for (method, budget), vals in by_key.items()
        if method == "Random"
    }

    rows = []
    for (method, budget), vals in sorted(by_key.items()):
        mean = sum(vals) / len(vals)
        baseline = random_means.get(budget)
        delta = "" if baseline is None else mean - baseline
        rows.append(
            {
                "method": method,
                "budget": budget,
                "n": len(vals),
                "mean_accuracy": mean,
                "delta_vs_random": delta,
            }
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "budget", "n", "mean_accuracy", "delta_vs_random"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
