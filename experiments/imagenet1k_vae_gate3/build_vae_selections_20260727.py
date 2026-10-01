#!/usr/bin/env python3
"""Build locked class-balanced Proxy-Hard and Random selections for seeds 0..4."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


OUT = Path(os.environ.get("COHE_IMAGENET_WORKDIR", "outputs/imagenet1k_vae_gate3"))
SCORE_MANIFEST = OUT / "scores/retry_20260727/imagenet1k_train_manifest.csv"
DEST = OUT / "selections/retry_20260727"
BUDGET = 0.30
SEEDS = [0, 1, 2, 3, 4]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_selection(path: Path, rows: list[dict[str, object]], policy: str, seed: int | str) -> None:
    fields = ["global_index", "relative_path", "class_name", "class_index", "file_size", "proxy_score", "policy", "selection_seed"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "global_index": row["global_index"],
                "relative_path": row["relative_path"],
                "class_name": row["class_name"],
                "class_index": row["class_index"],
                "file_size": row["file_size"],
                "proxy_score": row["proxy_score"],
                "policy": policy,
                "selection_seed": seed,
            })


def main() -> None:
    start = time.time()
    DEST.mkdir(parents=True, exist_ok=True)
    by_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    with SCORE_MANIFEST.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["global_index"] = int(row["global_index"])
            row["class_index"] = int(row["class_index"])
            row["proxy_score"] = float(row["proxy_score"])
            by_class[row["class_index"]].append(row)
    if len(by_class) != 1000 or sum(len(rows) for rows in by_class.values()) != 1_281_167:
        raise RuntimeError("score manifest coverage mismatch")

    class_counts = []
    hard_rows: list[dict[str, object]] = []
    selections: dict[str, list[dict[str, object]]] = {}
    for class_index in sorted(by_class):
        rows = by_class[class_index]
        k = int(math.floor(BUDGET * len(rows)))
        hard = sorted(rows, key=lambda r: (-float(r["proxy_score"]), str(r["relative_path"])))[:k]
        hard_rows.extend(hard)
        class_counts.append({"class_index": class_index, "class_name": rows[0]["class_name"], "source_count": len(rows), "selected_count": k})
    hard_rows.sort(key=lambda r: int(r["global_index"]))
    hard_path = DEST / "proxy_hard_30_seed_independent.csv"
    write_selection(hard_path, hard_rows, "Proxy-Hard", "seed-independent")
    selections["proxy_hard_30_seed_independent"] = hard_rows

    for seed in SEEDS:
        selected: list[dict[str, object]] = []
        for class_index in sorted(by_class):
            rows = by_class[class_index]
            k = int(math.floor(BUDGET * len(rows)))
            rng = np.random.default_rng(np.random.SeedSequence([seed, class_index]))
            chosen = rng.choice(len(rows), size=k, replace=False)
            selected.extend(rows[int(i)] for i in chosen)
        selected.sort(key=lambda r: int(r["global_index"]))
        path = DEST / f"random_30_seed{seed}.csv"
        write_selection(path, selected, "Random", seed)
        selections[f"random_30_seed{seed}"] = selected

    compare_path = DEST / "class_count_comparison.csv"
    with compare_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["class_index", "class_name", "source_count", "proxy_hard_count"] + [f"random_seed{seed}_count" for seed in SEEDS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in class_counts:
            ci = c["class_index"]
            row = {"class_index": ci, "class_name": c["class_name"], "source_count": c["source_count"], "proxy_hard_count": c["selected_count"]}
            for seed in SEEDS:
                row[f"random_seed{seed}_count"] = sum(int(r["class_index"]) == ci for r in selections[f"random_30_seed{seed}"])
            writer.writerow(row)

    totals = {name: len(rows) for name, rows in selections.items()}
    counts_match = all(all(sum(int(r["class_index"]) == ci for r in rows) == c["selected_count"] for c in class_counts for ci in [c["class_index"]]) for rows in selections.values())
    audit = {
        "status": "PASS" if counts_match and len(hard_rows) == sum(c["selected_count"] for c in class_counts) else "FAIL",
        "budget": BUDGET,
        "rounding_rule": "floor(class_count * budget)",
        "proxy_name": "VAE reconstruction error",
        "score_direction": "higher reconstruction MSE = harder",
        "hard_tie_break": "relative_path lexical ascending after descending score",
        "random_seed_rule": "np.random.SeedSequence([training_seed, class_index]) without replacement",
        "seeds": SEEDS,
        "class_coverage": len(by_class),
        "source_sample_count": 1_281_167,
        "selected_counts": totals,
        "class_count_match_across_policies": counts_match,
        "train_validation_leakage": "PASS based on verified dataset audit; selections contain train relative paths only",
        "score_manifest": str(SCORE_MANIFEST),
        "output_dir": str(DEST),
        "elapsed_seconds": time.time() - start,
    }
    (DEST / "selection_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks = []
    for path in sorted(DEST.glob("*.csv")):
        checks.append(f"{sha256_file(path)}  {path.name}")
    checks.append(f"{sha256_file(DEST / 'selection_audit.json')}  selection_audit.json")
    (DEST / "checksums.sha256").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise RuntimeError(audit)


if __name__ == "__main__":
    main()
