#!/usr/bin/env python3
"""Stage 2: one-dimensional held-out linear prediction summary."""

import argparse
import csv
from pathlib import Path


def read_scores(path, value_col):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["sample_id"]] = float(row[value_col])
    return rows


def read_splits(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["sample_id"]] = row["split"].strip().lower()
    return rows


def fit_line(x, y):
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    denom = sum((v - mx) ** 2 for v in x)
    slope = 0.0 if denom == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / denom
    intercept = my - slope * mx
    return intercept, slope


def r2_score(y_true, y_pred):
    mean = sum(y_true) / len(y_true)
    ss_tot = sum((v - mean) ** 2 for v in y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    proxy = read_scores(args.proxy, "proxy_score")
    target = read_scores(args.target, "target_score")
    splits = read_splits(args.splits)
    ids = sorted(set(proxy) & set(target) & set(splits))
    train = [i for i in ids if splits[i] == "train"]
    test = [i for i in ids if splits[i] == "test"]
    if len(train) < 2 or len(test) < 1:
        raise SystemExit("Need at least two train samples and one test sample.")

    x_train = [proxy[i] for i in train]
    y_train = [target[i] for i in train]
    x_test = [proxy[i] for i in test]
    y_test = [target[i] for i in test]
    intercept, slope = fit_line(x_train, y_train)
    pred = [intercept + slope * x for x in x_test]
    result = {
        "n_train": len(train),
        "n_test": len(test),
        "model": "one_dimensional_linear_regression",
        "heldout_r2": r2_score(y_test, pred),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)


if __name__ == "__main__":
    main()
