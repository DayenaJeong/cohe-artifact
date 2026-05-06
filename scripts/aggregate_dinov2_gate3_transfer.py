#!/usr/bin/env python3
"""Aggregate DINOv2 Gate 3 transfer runs against seed-matched Random."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


METHOD_ORDER = ["DINOv2-Isolation-Hard", "DINOv2-Isolation-Easy", "FirstLearning-Hard"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_proxy_csv(path: Path, score_col: str) -> np.ndarray:
    rows: list[tuple[int, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["sample_id"]), float(row[score_col])))
    rows.sort(key=lambda item: item[0])
    return np.array([score for _, score in rows], dtype=np.float64)


def load_labels(data_root: Path) -> np.ndarray:
    import torchvision

    dataset = torchvision.datasets.CIFAR100(root=str(data_root), train=True, download=False, transform=None)
    return np.array(dataset.targets, dtype=np.int64)


def normalized_entropy(labels: np.ndarray, num_classes: int = 100) -> float:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    probs = counts[counts > 0] / max(1.0, counts.sum())
    if len(probs) <= 1:
        return 0.0
    return float(-(probs * np.log(probs)).sum() / np.log(num_classes))


def selection_indices(method: str, scores: np.ndarray, first_learning: np.ndarray | None, budget: float, seed: int) -> np.ndarray:
    k = max(1, min(len(scores), int(round(budget * len(scores)))))
    if method == "Random":
        return np.random.RandomState(seed).choice(np.arange(len(scores)), size=k, replace=False)
    if method == "DINOv2-Isolation-Hard":
        return np.argsort(scores)[-k:]
    if method == "DINOv2-Isolation-Easy":
        return np.argsort(scores)[:k]
    if method == "FirstLearning-Hard":
        if first_learning is None:
            raise ValueError("FirstLearning-Hard requires --first-learning")
        return np.argsort(first_learning)[-k:]
    raise ValueError(f"Unknown method: {method}")


def diagnostics_for(
    method: str,
    budget: float,
    seed: int,
    scores: np.ndarray,
    labels: np.ndarray,
    first_learning: np.ndarray | None,
) -> dict[str, object]:
    idx = selection_indices(method, scores, first_learning, budget, seed)
    out: dict[str, object] = {
        "budget": f"{budget:.1f}",
        "seed": seed,
        "method": method,
        "classes_present": int(len(np.unique(labels[idx]))),
        "class_entropy_norm": f"{normalized_entropy(labels[idx]):.6f}",
        "mean_dinov2_isolation": f"{float(np.mean(scores[idx])):.6f}",
        "delta_vs_full_dinov2_isolation": f"{float(np.mean(scores[idx]) - np.mean(scores)):.6f}",
    }
    if first_learning is not None:
        threshold = float(np.percentile(first_learning, 90))
        out.update(
            {
                "mean_first_learning": f"{float(np.mean(first_learning[idx])):.6f}",
                "delta_vs_full_first_learning": f"{float(np.mean(first_learning[idx]) - np.mean(first_learning)):.6f}",
                "frac_top10_first_learning": f"{float(np.mean(first_learning[idx] >= threshold)):.6f}",
            }
        )
    else:
        out.update(
            {
                "mean_first_learning": "",
                "delta_vs_full_first_learning": "",
                "frac_top10_first_learning": "",
            }
        )
    return out


def load_run_rows(result_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((result_root / "raw").glob("*.json")):
        payload = json.loads(path.read_text())
        rows.append(
            {
                "dataset": payload.get("dataset", "CIFAR-100"),
                "model_name": payload.get("model_name", "resnet18"),
                "budget": float(payload["budget"]),
                "K": int(payload["K"]),
                "seed": int(payload["seed"]),
                "method": payload["method"],
                "acc": float(payload["acc"]),
                "source": path.name,
            }
        )
    return rows


def load_random_rows(path: Path, budgets: set[float], seeds: set[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(path):
        if row.get("method") != "Random":
            continue
        budget = float(row["budget"])
        seed = int(row["seed"])
        if budget in budgets and seed in seeds:
            rows.append(
                {
                    "dataset": "CIFAR-100",
                    "model_name": "resnet18",
                    "budget": budget,
                    "K": int(row["K"]),
                    "seed": seed,
                    "method": "Random",
                    "acc": float(row["acc"]),
                    "source": "existing_cifar100_random_baseline",
                }
            )
    return rows


def summarize_accuracy(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    keys = sorted({(float(r["budget"]), str(r["method"])) for r in rows})
    for budget, method in keys:
        vals = [float(r["acc"]) for r in rows if float(r["budget"]) == budget and str(r["method"]) == method]
        seeds = sorted(int(r["seed"]) for r in rows if float(r["budget"]) == budget and str(r["method"]) == method)
        arr = np.array(vals, dtype=float)
        out.append(
            {
                "budget": f"{budget:.1f}",
                "method": method,
                "mean_acc": f"{arr.mean():.4f}",
                "std_acc": f"{arr.std(ddof=1):.4f}" if len(arr) > 1 else "0.0000",
                "n": len(arr),
                "seeds": " ".join(str(s) for s in seeds),
            }
        )
    return out


def summarize_deltas(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_seed = {(float(r["budget"]), int(r["seed"]), str(r["method"])): float(r["acc"]) for r in rows}
    budgets = sorted({float(r["budget"]) for r in rows})
    delta_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for budget in budgets:
        random_seeds = {seed for (b, seed, method) in by_seed if b == budget and method == "Random"}
        methods = [m for m in METHOD_ORDER if any(b == budget and method == m for (b, _, method) in by_seed)]
        for method in methods:
            method_seeds = {seed for (b, seed, m) in by_seed if b == budget and m == method}
            paired = sorted(random_seeds & method_seeds)
            missing = sorted(random_seeds ^ method_seeds)
            if missing:
                missing_rows.append(
                    {
                        "budget": f"{budget:.1f}",
                        "method": method,
                        "random_seeds": " ".join(str(s) for s in sorted(random_seeds)),
                        "method_seeds": " ".join(str(s) for s in sorted(method_seeds)),
                        "unpaired_seeds": " ".join(str(s) for s in missing),
                    }
                )
            if not paired:
                continue
            deltas = np.array([by_seed[(budget, seed, method)] - by_seed[(budget, seed, "Random")] for seed in paired])
            mean = float(deltas.mean())
            if len(deltas) > 1:
                sem = float(stats.sem(deltas))
                tcrit = float(stats.t.ppf(0.975, len(deltas) - 1))
                lo = mean - tcrit * sem
                hi = mean + tcrit * sem
                p_value = float(stats.ttest_1samp(deltas, 0.0).pvalue)
                std = float(deltas.std(ddof=1))
            else:
                lo = hi = mean
                p_value = float("nan")
                std = 0.0
            delta_rows.append(
                {
                    "budget": f"{budget:.1f}",
                    "method": method,
                    "mean_delta_acc": f"{mean:.4f}",
                    "std_delta_acc": f"{std:.4f}",
                    "ci95_low": f"{lo:.4f}",
                    "ci95_high": f"{hi:.4f}",
                    "p_value": f"{p_value:.6g}" if np.isfinite(p_value) else "",
                    "n_pairs": len(paired),
                    "paired_seeds": " ".join(str(s) for s in paired),
                }
            )
    return delta_rows, missing_rows


def summarize_diagnostics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["budget"]), str(row["method"])), []).append(row)
    out = []
    metric_names = [
        "classes_present",
        "class_entropy_norm",
        "mean_dinov2_isolation",
        "delta_vs_full_dinov2_isolation",
        "mean_first_learning",
        "delta_vs_full_first_learning",
        "frac_top10_first_learning",
    ]
    for (budget, method), vals in sorted(grouped.items(), key=lambda kv: (float(kv[0][0]), kv[0][1])):
        row: dict[str, object] = {"budget": budget, "method": method, "n": len(vals)}
        for metric in metric_names:
            arr = np.array([float(v[metric]) for v in vals if v.get(metric) not in {"", None}], dtype=float)
            row[f"{metric}_mean"] = f"{float(arr.mean()):.6f}" if len(arr) else ""
            row[f"{metric}_std"] = f"{float(arr.std(ddof=1)):.6f}" if len(arr) > 1 else ("0.000000" if len(arr) else "")
        out.append(row)
    return out


def gate_status(delta_rows: list[dict[str, object]], declared_budgets: list[str]) -> dict[str, object]:
    method_status = {}
    for method in sorted({str(r["method"]) for r in delta_rows}):
        rows = [r for r in delta_rows if str(r["method"]) == method]
        by_budget = {str(r["budget"]): r for r in rows}
        present = all(b in by_budget for b in declared_budgets)
        pass_all = present and all(float(by_budget[b]["ci95_low"]) > 0.0 for b in declared_budgets)
        method_status[method] = {
            "declared_budgets": declared_budgets,
            "all_declared_budgets_present": present,
            "criterion": "Gate 3 passes only if every declared budget has paired CI95 lower bound > 0 vs Random.",
            "passes_gate3": bool(pass_all),
        }
    return {"stage": "DINOv2 Gate 3 transfer", "method_status": method_status}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--random-baseline", type=Path, required=True)
    parser.add_argument("--proxy-csv", type=Path, required=True)
    parser.add_argument("--score-col", type=str, default="dinov2_isolation_k20")
    parser.add_argument("--first-learning", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    run_rows = load_run_rows(args.result_root)
    budgets = {float(r["budget"]) for r in run_rows}
    seeds = {int(r["seed"]) for r in run_rows}
    random_rows = load_random_rows(args.random_baseline, budgets, seeds)
    rows = sorted(run_rows + random_rows, key=lambda r: (float(r["budget"]), str(r["method"]), int(r["seed"])))

    scores = load_proxy_csv(args.proxy_csv, args.score_col)
    first_learning = np.load(args.first_learning).astype(np.float64) if args.first_learning else None
    labels = load_labels(args.data_root)
    diag_rows = [
        diagnostics_for(str(r["method"]), float(r["budget"]), int(r["seed"]), scores, labels, first_learning)
        for r in rows
    ]
    # Add delta vs random first-learning means when matched diagnostics exist.
    random_diag = {(str(r["budget"]), int(r["seed"])): r for r in diag_rows if r["method"] == "Random"}
    for row in diag_rows:
        ref = random_diag.get((str(row["budget"]), int(row["seed"])))
        if ref and row.get("mean_first_learning") not in {"", None}:
            row["delta_vs_random_first_learning"] = f"{float(row['mean_first_learning']) - float(ref['mean_first_learning']):.6f}"
        else:
            row["delta_vs_random_first_learning"] = ""

    acc_rows = summarize_accuracy(rows)
    delta_rows, missing_rows = summarize_deltas(rows)
    diag_summary = summarize_diagnostics(diag_rows)
    declared_budgets = sorted({str(r["budget"]) for r in rows if r["method"] != "Random"}, key=float)
    status = gate_status(delta_rows, declared_budgets)

    out_dir = args.result_root / "aggregated"
    seed_fields = ["dataset", "model_name", "budget", "K", "seed", "method", "acc", "source"]
    write_csv(out_dir / "dinov2_gate3_seed_level_results.csv", seed_fields, rows)
    write_csv(out_dir / "dinov2_gate3_accuracy_summary.csv", ["budget", "method", "mean_acc", "std_acc", "n", "seeds"], acc_rows)
    write_csv(
        out_dir / "dinov2_gate3_transfer_delta_summary.csv",
        ["budget", "method", "mean_delta_acc", "std_delta_acc", "ci95_low", "ci95_high", "p_value", "n_pairs", "paired_seeds"],
        delta_rows,
    )
    write_csv(out_dir / "dinov2_gate3_missing_pairs.csv", ["budget", "method", "random_seeds", "method_seeds", "unpaired_seeds"], missing_rows)
    diag_fields = list(diag_rows[0].keys()) if diag_rows else []
    if diag_fields:
        write_csv(out_dir / "dinov2_gate3_diagnostics_seed_level.csv", diag_fields, diag_rows)
    summary_fields = list(diag_summary[0].keys()) if diag_summary else []
    if summary_fields:
        write_csv(out_dir / "dinov2_gate3_diagnostics_summary.csv", summary_fields, diag_summary)
    (out_dir / "dinov2_gate3_gate_status.json").write_text(json.dumps(status, indent=2) + "\n")

    print(f"Wrote {len(rows)} seed-level rows to {out_dir}")
    print(f"Wrote {len(delta_rows)} paired delta rows")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
