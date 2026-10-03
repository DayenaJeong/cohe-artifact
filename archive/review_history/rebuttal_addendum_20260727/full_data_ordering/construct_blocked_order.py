#!/usr/bin/env python3
"""Reconstruct and verify the released ten-bin DDPM ordering."""

from pathlib import Path
import argparse
import csv
import gzip

import numpy as np


HERE = Path(__file__).resolve().parent
SCORES = HERE.parent / "cifar100_proxy_arrays" / "arrays" / "ddpm_scores.npy"
MEMBERSHIP = HERE / "ordering_metadata" / "bin_membership.csv.gz"


def build_membership(scores):
    sample_ids = np.arange(scores.size, dtype=np.int64)
    order = np.lexsort((sample_ids, scores))
    rows = []
    for rank, sample_id in enumerate(order):
        rows.append((int(sample_id), int(rank // 5000), int(rank)))
    return rows


def verify():
    scores = np.load(SCORES, allow_pickle=False)
    if scores.shape != (50000,) or scores.dtype != np.float32:
        raise SystemExit(f"unexpected score array: shape={scores.shape}, dtype={scores.dtype}")
    if not np.isfinite(scores).all():
        raise SystemExit("score array contains non-finite values")
    expected = build_membership(scores)
    with gzip.open(MEMBERSHIP, "rt", newline="") as handle:
        actual = [
            (int(row["sample_index"]), int(row["bin_index"]), int(row["rank_in_sorted_order"]))
            for row in csv.DictReader(handle)
        ]
    if actual != expected:
        raise SystemExit("released bin membership does not match deterministic reconstruction")
    bins = [row[1] for row in actual]
    if sorted(bins) != [bin_index for bin_index in range(10) for _ in range(5000)]:
        raise SystemExit("bin cardinalities are not ten bins of 5000")
    print("ORDERING_VERIFICATION=PASS")
    print(f"SAMPLE_COUNT={len(actual)}")
    print("BIN_COUNT=10")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="verify the released manifest")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        print("Use --verify to check the released membership manifest.")


if __name__ == "__main__":
    main()
