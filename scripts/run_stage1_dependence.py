#!/usr/bin/env python3
"""Stage 1: dependence summary for a sample-indexed proxy and target CSV."""

import argparse
import csv
import math
from pathlib import Path


def read_scores(path, value_col):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["sample_id"]] = float(row[value_col])
    return rows


def pearson(x, y):
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx == 0 or vy == 0:
        return float("nan")
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    proxy = read_scores(args.proxy, "proxy_score")
    target = read_scores(args.target, "target_score")
    ids = sorted(set(proxy) & set(target))
    x = [proxy[i] for i in ids]
    y = [target[i] for i in ids]
    result = {
        "n": len(ids),
        "pearson_r": pearson(x, y),
        "spearman_rho": pearson(ranks(x), ranks(y)),
        "nonlinear_check": "not_run",
        "note": "HSIC or nonlinear dependence checks are optional when feasible.",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)


if __name__ == "__main__":
    main()
