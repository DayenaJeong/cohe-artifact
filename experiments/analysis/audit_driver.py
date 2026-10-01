#!/usr/bin/env python3
"""Utility driver for ICML 2026 rebuttal audit and analyses.

This script is intentionally additive: it reads existing project assets and
writes all rebuttal outputs under ``rebuttal_2026/``.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


def summarize_npy(path: Path) -> dict[str, Any]:
    arr = np.load(path, allow_pickle=True)
    summary: dict[str, Any] = {
        "path": str(path.resolve()),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }
    if arr.ndim > 0 and np.issubdtype(arr.dtype, np.number):
        flat = np.asarray(arr).reshape(-1)
        finite = np.isfinite(flat)
        summary.update(
            {
                "numel": int(flat.size),
                "finite_fraction": float(finite.mean()) if flat.size else 1.0,
                "min": float(np.nanmin(flat)) if flat.size else None,
                "max": float(np.nanmax(flat)) if flat.size else None,
                "mean": float(np.nanmean(flat)) if flat.size else None,
                "std": float(np.nanstd(flat)) if flat.size else None,
            }
        )
    return summary


def summarize_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    summary: dict[str, Any] = {
        "path": str(path.resolve()),
        "type": type(payload).__name__,
    }
    if isinstance(payload, dict):
        summary["keys"] = sorted(payload.keys())
        if "results" in payload and isinstance(payload["results"], list):
            summary["num_results"] = len(payload["results"])
            summary["methods"] = [r.get("method", r.get("strategy")) for r in payload["results"]]
    elif isinstance(payload, list):
        summary["length"] = len(payload)
    return summary


def cmd_probe_npy(args: argparse.Namespace) -> None:
    print(json.dumps(summarize_npy(Path(args.path)), indent=2))


def cmd_probe_json(args: argparse.Namespace) -> None:
    print(json.dumps(summarize_json(Path(args.path)), indent=2))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def cmd_search_text(args: argparse.Namespace) -> None:
    root = Path(args.root)
    pattern = re.compile(args.pattern)
    matches: list[dict[str, Any]] = []
    allowed_suffixes = {".py", ".md", ".txt", ".tex", ".json", ".yaml", ".yml", ".sh"}
    skip_dirs = {
        ".git",
        "__pycache__",
        ".ipynb_checkpoints",
        "data",
        "checkpoints",
        "checkpoints_gradpurify",
        "models",
        "outputs",
        "outputs_alpha_mix",
        "outputs_clip_proxy",
        "outputs_gen",
        "results_icml_diffusion_grad",
        "results_icml_diffusion_grad_corrupt",
        "results_icml_diffusion_grad_fixed",
        "qualitative_samples",
        "png",
        "figures",
        "masks",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        lines = text.splitlines()
        for lineno, line in enumerate(lines, start=1):
            if pattern.search(line):
                matches.append(
                    {
                        "path": str(path.resolve()),
                        "line": lineno,
                        "text": line.strip(),
                    }
                )
                if len(matches) >= args.limit:
                    print(json.dumps(matches, indent=2))
                    return
    print(json.dumps(matches, indent=2))


def _paired_summary(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n = int(diff.size)
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    se_diff = float(sd_diff / math.sqrt(n)) if n > 1 else float("nan")

    ci_low = float("nan")
    ci_high = float("nan")
    t_p = float("nan")
    t_stat = float("nan")
    if n > 1:
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        ci_low = mean_diff - tcrit * se_diff
        ci_high = mean_diff + tcrit * se_diff
        t_res = stats.ttest_rel(a, b)
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)

    wilcoxon_p = float("nan")
    wilcoxon_stat = float("nan")
    if n > 1 and not np.allclose(diff, 0.0):
        try:
            w_res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="exact")
            wilcoxon_stat = float(w_res.statistic)
            wilcoxon_p = float(w_res.pvalue)
        except Exception:
            pass

    cohen_dz = float("nan")
    if n > 1 and sd_diff > 0:
        cohen_dz = mean_diff / sd_diff

    return {
        "n_pairs": n,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": mean_diff,
        "std_diff": sd_diff,
        "se_diff": se_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "paired_t_stat": t_stat,
        "paired_t_p": t_p,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "cohen_dz": cohen_dz,
    }


def cmd_selection_stats(args: argparse.Namespace) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    payload = load_json(Path(args.alpha_mix_json))
    rows = payload["results"]
    df = pd.DataFrame(rows)
    df = df.rename(columns={"strategy": "method"})
    df["budget"] = df["budget"].astype(float)
    df["seed"] = df["seed"].astype(int)
    df["acc"] = df["acc"].astype(float)

    methods = [m.strip() for m in args.compare_methods.split(",") if m.strip()]
    baseline = args.baseline

    paired_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    budgets = sorted(df["budget"].unique())
    for budget in budgets:
        for method in methods:
            sub = df[(df["budget"] == budget) & (df["method"].isin([baseline, method]))]
            if sub.empty:
                continue
            pivot = sub.pivot_table(index="seed", columns="method", values="acc", aggfunc="first")
            if baseline not in pivot.columns or method not in pivot.columns:
                continue
            pivot = pivot[[baseline, method]].dropna()
            if pivot.empty:
                continue

            a = pivot[method].to_numpy(dtype=float)
            b = pivot[baseline].to_numpy(dtype=float)
            stats_row = _paired_summary(a, b)
            stats_row.update(
                {
                    "budget": float(budget),
                    "method": method,
                    "baseline": baseline,
                }
            )
            summary_rows.append(stats_row)

            for seed, row in pivot.iterrows():
                paired_rows.append(
                    {
                        "budget": float(budget),
                        "method": method,
                        "baseline": baseline,
                        "seed": int(seed),
                        "acc_method": float(row[method]),
                        "acc_baseline": float(row[baseline]),
                        "diff": float(row[method] - row[baseline]),
                    }
                )

    paired_df = pd.DataFrame(paired_rows).sort_values(["budget", "method", "seed"])
    summary_df = pd.DataFrame(summary_rows).sort_values(["budget", "method"])

    paired_csv = out_dir / "paired_differences.csv"
    summary_csv = out_dir / "selection_stats_summary.csv"
    raw_csv = out_dir / "alpha_mix_long_results.csv"
    paired_df.to_csv(paired_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    df.sort_values(["budget", "method", "seed"]).to_csv(raw_csv, index=False)

    # Rebuttal-friendly markdown table.
    table_lines = [
        "| Budget | Comparison | Mean(Method) | Mean(Random) | Mean Diff | 95% CI | p (paired t) | dz |",
        "|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        ci = f"[{row['ci95_low']:.2f}, {row['ci95_high']:.2f}]" if np.isfinite(row["ci95_low"]) else "[NA, NA]"
        table_lines.append(
            f"| {row['budget']:.1f} | {row['method']} - {row['baseline']} | "
            f"{row['mean_a']:.2f} | {row['mean_b']:.2f} | {row['mean_diff']:.2f} | "
            f"{ci} | {row['paired_t_p']:.3f} | {row['cohen_dz']:.2f} |"
        )
    (out_dir / "selection_stats_table.md").write_text("\n".join(table_lines) + "\n")

    # Effect-size plot.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plot_df = summary_df.copy()
    plot_df["label"] = plot_df.apply(lambda r: f"{r['budget']:.1f}: {r['method']}", axis=1)
    x = np.arange(len(plot_df))
    y = plot_df["mean_diff"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            y - plot_df["ci95_low"].to_numpy(dtype=float),
            plot_df["ci95_high"].to_numpy(dtype=float) - y,
        ]
    )
    colors = {
        "OHS-XOR(p75)": "#1f77b4",
        "Gen-Hard": "#2ca02c",
        "Disc-Hard": "#d62728",
    }
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.bar(
        x,
        y,
        yerr=yerr,
        color=[colors.get(m, "#888888") for m in plot_df["method"]],
        alpha=0.85,
        capsize=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=30, ha="right")
    ax.set_ylabel("Paired Accuracy Difference vs Random")
    ax.set_title("Selection Gains Are Small and Uncertain Across 3 Seeds")
    fig.tight_layout()
    fig.savefig(out_dir / "selection_stats_effects.png", dpi=220)
    plt.close(fig)

    summary_payload = {
        "alpha_mix_json": str(Path(args.alpha_mix_json).resolve()),
        "baseline": baseline,
        "compare_methods": methods,
        "budgets": budgets,
        "outputs": {
            "raw_long_csv": str(raw_csv.resolve()),
            "paired_differences_csv": str(paired_csv.resolve()),
            "summary_csv": str(summary_csv.resolve()),
            "table_md": str((out_dir / "selection_stats_table.md").resolve()),
            "figure_png": str((out_dir / "selection_stats_effects.png").resolve()),
        },
    }
    (out_dir / "selection_stats_manifest.json").write_text(json.dumps(summary_payload, indent=2))
    print(json.dumps(summary_payload, indent=2))


def _load_xy(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(x_path).reshape(-1).astype(float)
    y = np.load(y_path).reshape(-1).astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def cmd_corr_report(args: argparse.Namespace) -> None:
    from scipy import stats

    x, y = _load_xy(Path(args.x_path), Path(args.y_path))
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    payload = {
        "x_path": str(Path(args.x_path).resolve()),
        "y_path": str(Path(args.y_path).resolve()),
        "n": int(x.size),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
        "r_squared_linear_identity": float(pr**2),
    }
    if args.out_json:
        out_path = Path(args.out_json)
        ensure_dir(out_path.parent)
        out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def _fit_models_for_direction(
    x: np.ndarray,
    y: np.ndarray,
    split_seed: int,
    max_train_samples: int,
) -> list[dict[str, Any]]:
    import warnings
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=split_seed, shuffle=True
    )

    if max_train_samples > 0 and len(x_train) > max_train_samples:
        rng = np.random.default_rng(split_seed)
        keep = rng.choice(len(x_train), size=max_train_samples, replace=False)
        x_train = x_train[keep]
        y_train = y_train[keep]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            random_state=split_seed,
            n_jobs=-1,
        ),
        "HistGBDT": HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            random_state=split_seed,
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 32),
                activation="relu",
                alpha=1e-3,
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=split_seed,
            ),
        ),
    }

    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append(
            {
                "model": name,
                "split_seed": int(split_seed),
                "train_n": int(len(x_train)),
                "test_n": int(len(x_test)),
                "r2": float(r2_score(y_test, pred)),
            }
        )
    rows.append(
        {
            "model": "MeanPredictor",
            "split_seed": int(split_seed),
            "train_n": int(len(x_train)),
            "test_n": int(len(x_test)),
            "r2": 0.0,
        }
    )
    return rows


def cmd_nonlinear_r2(args: argparse.Namespace) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    x, y = _load_xy(Path(args.x_path), Path(args.y_path))
    seeds = [int(s.strip()) for s in args.split_seeds.split(",") if s.strip()]

    all_rows: list[dict[str, Any]] = []
    directions = [("x_to_y", x, y, args.x_name, args.y_name)]
    if args.bidirectional:
        directions.append(("y_to_x", y, x, args.y_name, args.x_name))

    for direction, src, dst, src_name, dst_name in directions:
        for seed in seeds:
            rows = _fit_models_for_direction(src, dst, seed, args.max_train_samples)
            for row in rows:
                row.update(
                    {
                        "direction": direction,
                        "source_name": src_name,
                        "target_name": dst_name,
                    }
                )
            all_rows.extend(rows)

    per_split_df = pd.DataFrame(all_rows)
    summary_df = (
        per_split_df.groupby(["direction", "source_name", "target_name", "model"], as_index=False)
        .agg(mean_r2=("r2", "mean"), std_r2=("r2", "std"), min_r2=("r2", "min"), max_r2=("r2", "max"))
        .sort_values(["direction", "mean_r2"], ascending=[True, False])
    )

    per_split_csv = out_dir / "nonlinear_r2_per_split.csv"
    summary_csv = out_dir / "nonlinear_r2_summary.csv"
    per_split_df.to_csv(per_split_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(1, len(directions), figsize=(6.5 * len(directions), 4.8), squeeze=False)
    for ax, (direction, _, _, src_name, dst_name) in zip(axes[0], directions):
        sub = summary_df[summary_df["direction"] == direction].copy()
        ax.bar(
            np.arange(len(sub)),
            sub["mean_r2"].to_numpy(dtype=float),
            yerr=sub["std_r2"].fillna(0.0).to_numpy(dtype=float),
            color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#9c755f"][: len(sub)],
            alpha=0.9,
            capsize=4,
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["model"], rotation=30, ha="right")
        ax.set_ylabel("Held-out $R^2$")
        ax.set_title(f"{src_name} -> {dst_name}")
    fig.suptitle("Nonlinear Predictive Power Remains Near Zero", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "nonlinear_r2_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "x_path": str(Path(args.x_path).resolve()),
        "y_path": str(Path(args.y_path).resolve()),
        "x_name": args.x_name,
        "y_name": args.y_name,
        "split_seeds": seeds,
        "bidirectional": bool(args.bidirectional),
        "max_train_samples": int(args.max_train_samples),
        "outputs": {
            "per_split_csv": str(per_split_csv.resolve()),
            "summary_csv": str(summary_csv.resolve()),
            "figure_png": str((out_dir / "nonlinear_r2_summary.png").resolve()),
        },
    }
    (out_dir / "nonlinear_r2_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def _quantile_binned_trend(x: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.array([float(np.mean(x))]), np.array([float(np.mean(y))])

    x_centers: list[float] = []
    y_means: list[float] = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)
        if not np.any(mask):
            continue
        x_centers.append(float(np.mean(x[mask])))
        y_means.append(float(np.mean(y[mask])))
    return np.asarray(x_centers, dtype=float), np.asarray(y_means, dtype=float)


def cmd_training_dynamics_report(args: argparse.Namespace) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    dgen = np.load(args.dgen_path).reshape(-1).astype(float)
    forgetting = np.load(args.forgetting_path).reshape(-1).astype(float)
    first_learning_raw = np.load(args.first_learning_path).reshape(-1).astype(float)

    if not (len(dgen) == len(forgetting) == len(first_learning_raw)):
        raise ValueError("D_gen and dynamics arrays must have the same length.")

    never_value = float(args.epochs + 1) if args.never_value == "epochs_plus_one" else float(args.never_value)
    first_learning = first_learning_raw.copy()
    never_mask = first_learning < 0
    first_learning[never_mask] = never_value

    metric_payloads = [
        {
            "metric": "forgetting_count",
            "display_name": "Forgetting Count",
            "values": forgetting,
            "note": "Number of correct-to-incorrect flips across epochs.",
        },
        {
            "metric": "first_learning_epoch",
            "display_name": "First-Learning Epoch",
            "values": first_learning,
            "note": f"Samples never stably learned were mapped to epoch {never_value:.0f}.",
        },
    ]

    corr_rows: list[dict[str, Any]] = []
    nonlinear_rows: list[dict[str, Any]] = []

    for payload in metric_payloads:
        metric = payload["metric"]
        values = np.asarray(payload["values"], dtype=float)
        mask = np.isfinite(dgen) & np.isfinite(values)
        x = dgen[mask]
        y = values[mask]

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)

        seeds = [int(s.strip()) for s in args.split_seeds.split(",") if s.strip()]
        metric_model_rows: list[dict[str, Any]] = []
        for seed in seeds:
            rows = _fit_models_for_direction(x, y, seed, args.max_train_samples)
            for row in rows:
                row.update(
                    {
                        "metric": metric,
                        "display_name": payload["display_name"],
                    }
                )
            metric_model_rows.extend(rows)
        nonlinear_rows.extend(metric_model_rows)

        metric_df = pd.DataFrame(metric_model_rows)
        metric_summary = (
            metric_df.groupby(["metric", "display_name", "model"], as_index=False)
            .agg(mean_r2=("r2", "mean"), std_r2=("r2", "std"))
            .sort_values(["metric", "mean_r2"], ascending=[True, False])
        )
        best_row = metric_summary.iloc[0]

        corr_rows.append(
            {
                "metric": metric,
                "display_name": payload["display_name"],
                "n": int(mask.sum()),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_rho": float(spearman_rho),
                "spearman_p": float(spearman_p),
                "linear_r2_identity": float(pearson_r**2),
                "best_model": str(best_row["model"]),
                "best_mean_r2": float(best_row["mean_r2"]),
                "best_std_r2": float(best_row["std_r2"]) if pd.notna(best_row["std_r2"]) else 0.0,
                "metric_note": payload["note"],
            }
        )

    corr_df = pd.DataFrame(corr_rows)
    nonlinear_df = pd.DataFrame(nonlinear_rows)
    nonlinear_summary_df = (
        nonlinear_df.groupby(["metric", "display_name", "model"], as_index=False)
        .agg(mean_r2=("r2", "mean"), std_r2=("r2", "std"), min_r2=("r2", "min"), max_r2=("r2", "max"))
        .sort_values(["metric", "mean_r2"], ascending=[True, False])
    )

    corr_csv = out_dir / "training_dynamics_correlations.csv"
    nonlinear_per_split_csv = out_dir / "training_dynamics_nonlinear_r2_per_split.csv"
    nonlinear_summary_csv = out_dir / "training_dynamics_nonlinear_r2_summary.csv"
    corr_df.to_csv(corr_csv, index=False)
    nonlinear_df.to_csv(nonlinear_per_split_csv, index=False)
    nonlinear_summary_df.to_csv(nonlinear_summary_csv, index=False)

    rng = np.random.default_rng(args.plot_seed)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), squeeze=False)
    for ax, payload in zip(axes[0], metric_payloads):
        metric = payload["metric"]
        values = np.asarray(payload["values"], dtype=float)
        mask = np.isfinite(dgen) & np.isfinite(values)
        x = dgen[mask]
        y = values[mask]
        show_n = min(args.scatter_points, len(x))
        show_idx = rng.choice(len(x), size=show_n, replace=False) if show_n < len(x) else np.arange(len(x))
        ax.scatter(x[show_idx], y[show_idx], s=5, alpha=0.12, color="#4c78a8")
        bx, by = _quantile_binned_trend(x, y, args.n_bins)
        ax.plot(bx, by, color="#e45756", linewidth=2.2)
        row = corr_df[corr_df["metric"] == metric].iloc[0]
        ax.set_xlabel(args.dgen_name)
        ax.set_ylabel(payload["display_name"])
        ax.set_title(
            f"{payload['display_name']}\n"
            f"r={row['pearson_r']:.3f}, rho={row['spearman_rho']:.3f}, best R^2={row['best_mean_r2']:.4f}"
        )
        if metric == "first_learning_epoch":
            ax.set_ylim(-0.5, max(float(args.epochs) + 2.0, np.nanmax(y) + 1.0))
    fig.suptitle("Training-Dynamics Hardness Shows Weak Association with D_gen", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "training_dynamics_plot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    forgetting_row = corr_df[corr_df["metric"] == "forgetting_count"].iloc[0]
    first_row = corr_df[corr_df["metric"] == "first_learning_epoch"].iloc[0]
    never_count = int(never_mask.sum())
    n_total = int(len(first_learning_raw))

    def _interpret_row(row: pd.Series) -> str:
        abs_corr = max(abs(float(row["pearson_r"])), abs(float(row["spearman_rho"])))
        best_r2 = float(row["best_mean_r2"])
        if abs_corr < 0.1 and best_r2 <= 0.01:
            return "weak predictive signal"
        if abs_corr < 0.2 and best_r2 <= 0.03:
            return "small but still limited signal"
        return "non-negligible signal that should be described cautiously"

    summary_lines = [
        "# Training-Dynamics Summary",
        "",
        "Setup:",
        f"- Generative proxy: `{Path(args.dgen_path).resolve()}`",
        f"- Forgetting counts: `{Path(args.forgetting_path).resolve()}`",
        f"- First-learning epochs: `{Path(args.first_learning_path).resolve()}`",
        f"- Non-learned handling: mapped `{never_count}` / `{n_total}` samples to epoch `{never_value:.0f}`",
        "",
        "Results:",
        (
            f"- Forgetting count vs D_gen: `r = {forgetting_row['pearson_r']:.4f}`, "
            f"`rho = {forgetting_row['spearman_rho']:.4f}`, "
            f"best held-out `R^2 = {forgetting_row['best_mean_r2']:.4f}` "
            f"({forgetting_row['best_model']}); interpretation: {_interpret_row(forgetting_row)}."
        ),
        (
            f"- First-learning epoch vs D_gen: `r = {first_row['pearson_r']:.4f}`, "
            f"`rho = {first_row['spearman_rho']:.4f}`, "
            f"best held-out `R^2 = {first_row['best_mean_r2']:.4f}` "
            f"({first_row['best_model']}); interpretation: {_interpret_row(first_row)}."
        ),
        "",
        "Takeaway:",
        (
            "These dynamics-based discriminative hardness measures should be framed as evidence about "
            "empirical decoupling, not independence. If the reported correlations remain small and the "
            "best held-out nonlinear `R^2` stays near zero, then D_gen still has weak predictive power "
            "even when discriminative hardness is defined through training dynamics rather than static loss."
        ),
    ]
    summary_md = out_dir / "training_dynamics_summary.md"
    summary_md.write_text("\n".join(summary_lines) + "\n")

    manifest = {
        "dgen_path": str(Path(args.dgen_path).resolve()),
        "forgetting_path": str(Path(args.forgetting_path).resolve()),
        "first_learning_path": str(Path(args.first_learning_path).resolve()),
        "dgen_name": args.dgen_name,
        "epochs": int(args.epochs),
        "never_value": never_value,
        "split_seeds": [int(s.strip()) for s in args.split_seeds.split(",") if s.strip()],
        "outputs": {
            "correlations_csv": str(corr_csv.resolve()),
            "nonlinear_per_split_csv": str(nonlinear_per_split_csv.resolve()),
            "nonlinear_summary_csv": str(nonlinear_summary_csv.resolve()),
            "plot_png": str((out_dir / "training_dynamics_plot.png").resolve()),
            "summary_md": str(summary_md.resolve()),
        },
    }
    (out_dir / "training_dynamics_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def _topk_indices(x: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(x)[-k:]


def _random_indices(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice(np.arange(n), size=k, replace=False)


def _load_cifar100_train_labels(cifar_root: Path) -> np.ndarray:
    train_path = cifar_root / "train"
    with train_path.open("rb") as f:
        payload = pickle.load(f, encoding="bytes")
    return np.array(payload[b"fine_labels"], dtype=int)


def _load_cifar100_train_images(cifar_root: Path) -> np.ndarray:
    train_path = cifar_root / "train"
    with train_path.open("rb") as f:
        payload = pickle.load(f, encoding="bytes")
    data = payload[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data.astype(np.uint8)


def _subset_metrics(
    indices: np.ndarray,
    labels: np.ndarray,
    dgen: np.ndarray,
    ddisc: np.ndarray,
    q90_gen: float,
    q90_disc: float,
    q75_gen: float,
    q75_disc: float,
    med_gen: float,
    med_disc: float,
) -> dict[str, Any]:
    counts = np.bincount(labels[indices], minlength=100).astype(float)
    probs = counts / max(counts.sum(), 1.0)
    nonzero = probs[probs > 0]
    class_entropy = float(-(nonzero * np.log(nonzero)).sum())
    class_entropy_norm = float(class_entropy / np.log(len(probs)))

    sel_gen = dgen[indices]
    sel_disc = ddisc[indices]
    q1 = ((sel_gen < med_gen) & (sel_disc < med_disc)).mean()
    q2 = ((sel_gen >= med_gen) & (sel_disc < med_disc)).mean()
    q3 = ((sel_gen < med_gen) & (sel_disc >= med_disc)).mean()
    q4 = ((sel_gen >= med_gen) & (sel_disc >= med_disc)).mean()

    return {
        "n_selected": int(len(indices)),
        "classes_present": int((counts > 0).sum()),
        "class_entropy_norm": class_entropy_norm,
        "max_class_share": float(probs.max()),
        "l1_to_uniform": float(np.abs(probs - (1.0 / len(probs))).sum()),
        "avg_disc_ce": float(sel_disc.mean()),
        "avg_gen": float(sel_gen.mean()),
        "frac_top10_disc": float((sel_disc >= q90_disc).mean()),
        "frac_top10_gen": float((sel_gen >= q90_gen).mean()),
        "frac_double_hard_q75": float(((sel_gen >= q75_gen) & (sel_disc >= q75_disc)).mean()),
        "quad_easy_easy": float(q1),
        "quad_hard_gen": float(q2),
        "quad_hard_disc": float(q3),
        "quad_double_hard": float(q4),
    }


def _save_image_grid(
    images: np.ndarray,
    out_path: Path,
    title: str,
    row_labels: list[str],
    row_indices: list[np.ndarray],
    n_cols: int = 10,
) -> None:
    from PIL import Image, ImageDraw

    cell = 48
    pad = 8
    title_h = 30
    label_w = 90
    n_rows = len(row_indices)
    width = label_w + n_cols * cell + (n_cols + 1) * pad
    height = title_h + n_rows * cell + (n_rows + 1) * pad
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 6), title, fill=(0, 0, 0))
    for r, (label, idxs) in enumerate(zip(row_labels, row_indices)):
        y = title_h + pad + r * cell
        draw.text((pad, y + cell // 3), label, fill=(0, 0, 0))
        for c in range(n_cols):
            x = label_w + pad + c * cell
            if c < len(idxs):
                tile = Image.fromarray(images[int(idxs[c])]).resize((cell - 2, cell - 2), Image.Resampling.NEAREST)
                canvas.paste(tile, (x, y))
            else:
                draw.rectangle([x, y, x + cell - 2, y + cell - 2], outline=(180, 180, 180), width=1)
    canvas.save(out_path)


def cmd_disc_hard_failure(args: argparse.Namespace) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    dgen = np.load(args.dgen_path).reshape(-1).astype(float)
    ddisc = np.load(args.ddisc_path).reshape(-1).astype(float)
    labels = _load_cifar100_train_labels(Path(args.cifar_root))
    images = _load_cifar100_train_images(Path(args.cifar_root))

    budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    n = len(dgen)
    q90_gen = float(np.quantile(dgen, 0.9))
    q90_disc = float(np.quantile(ddisc, 0.9))
    q75_gen = float(np.quantile(dgen, 0.75))
    q75_disc = float(np.quantile(ddisc, 0.75))
    med_gen = float(np.median(dgen))
    med_disc = float(np.median(ddisc))

    per_seed_rows: list[dict[str, Any]] = []
    agg_rows: list[dict[str, Any]] = []
    chosen_for_grid: dict[tuple[float, str], np.ndarray] = {}

    for budget in budgets:
        k = int(round(n * budget))
        disc_idx = _topk_indices(ddisc, k)
        chosen_for_grid[(budget, "Disc-Hard")] = disc_idx[: args.grid_cols]

        disc_metrics = _subset_metrics(disc_idx, labels, dgen, ddisc, q90_gen, q90_disc, q75_gen, q75_disc, med_gen, med_disc)
        disc_metrics.update({"budget": budget, "strategy": "Disc-Hard", "seed": -1})
        per_seed_rows.append(disc_metrics)

        random_metric_rows = []
        for seed in seeds:
            rand_idx = _random_indices(n, k, seed)
            if seed == seeds[0]:
                chosen_for_grid[(budget, "Random")] = rand_idx[: args.grid_cols]
            metrics = _subset_metrics(rand_idx, labels, dgen, ddisc, q90_gen, q90_disc, q75_gen, q75_disc, med_gen, med_disc)
            metrics.update({"budget": budget, "strategy": "Random", "seed": seed})
            per_seed_rows.append(metrics)
            random_metric_rows.append(metrics)

        for strategy, rows in [("Random", random_metric_rows), ("Disc-Hard", [disc_metrics])]:
            row_df = pd.DataFrame(rows)
            agg = {"budget": budget, "strategy": strategy}
            for col in row_df.columns:
                if col in {"budget", "strategy", "seed"}:
                    continue
                agg[f"{col}_mean"] = float(row_df[col].mean())
                agg[f"{col}_std"] = float(row_df[col].std(ddof=1)) if len(row_df) > 1 else 0.0
            agg_rows.append(agg)

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["budget", "strategy", "seed"])
    agg_df = pd.DataFrame(agg_rows).sort_values(["budget", "strategy"])
    per_seed_df.to_csv(out_dir / "disc_hard_failure_per_seed.csv", index=False)
    agg_df.to_csv(out_dir / "disc_hard_failure_summary.csv", index=False)

    # Diagnostic markdown table with selected columns.
    cols = [
        "classes_present_mean",
        "class_entropy_norm_mean",
        "avg_disc_ce_mean",
        "frac_top10_disc_mean",
        "frac_double_hard_q75_mean",
        "quad_hard_disc_mean",
        "quad_double_hard_mean",
    ]
    header = "| Budget | Strategy | Classes Present | Class Entropy | Avg CE | Top-10% Disc | Double-Hard (Q75) | Hard-Disc Quad | Double-Hard Quad |"
    sep = "|---:|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for _, row in agg_df.iterrows():
        lines.append(
            f"| {row['budget']:.1f} | {row['strategy']} | "
            f"{row['classes_present_mean']:.1f} | {row['class_entropy_norm_mean']:.3f} | "
            f"{row['avg_disc_ce_mean']:.3f} | {row['frac_top10_disc_mean']:.3f} | "
            f"{row['frac_double_hard_q75_mean']:.3f} | {row['quad_hard_disc_mean']:.3f} | "
            f"{row['quad_double_hard_mean']:.3f} |"
        )
    (out_dir / "disc_hard_failure_table.md").write_text("\n".join(lines) + "\n")

    # Occupancy plot.
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(n, size=min(args.background_points, n), replace=False)
    fig, axes = plt.subplots(len(budgets), 2, figsize=(10.5, 4.2 * len(budgets)), squeeze=False)
    for r, budget in enumerate(budgets):
        k = int(round(n * budget))
        panels = [
            ("Random", _random_indices(n, k, seeds[0]), "#4c78a8"),
            ("Disc-Hard", _topk_indices(ddisc, k), "#e45756"),
        ]
        for c, (name, idx, color) in enumerate(panels):
            ax = axes[r, c]
            ax.scatter(dgen[bg_idx], ddisc[bg_idx], s=4, alpha=0.08, color="gray")
            show_idx = idx if len(idx) <= args.overlay_points else rng.choice(idx, size=args.overlay_points, replace=False)
            ax.scatter(dgen[show_idx], ddisc[show_idx], s=7, alpha=0.45, color=color)
            ax.axvline(med_gen, color="black", linestyle="--", linewidth=1)
            ax.axhline(med_disc, color="black", linestyle="--", linewidth=1)
            ax.set_xlabel("Generative Hardness")
            ax.set_ylabel("Discriminative Hardness")
            ax.set_title(f"Budget {budget:.1f} | {name}")
    fig.suptitle("Disc-Hard Concentrates in the High-Ddisc / Low-Coverage Region", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "disc_hard_subset_occupancy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Small qualitative grid for budget 0.1 if available.
    if budgets:
        budget0 = budgets[0]
        _save_image_grid(
            images,
            out_dir / "disc_hard_examples_budget0.1.png",
            title="Budget 0.1 | Random vs Disc-Hard (seed 0 for Random)",
            row_labels=["Random", "Disc-Hard"],
            row_indices=[
                chosen_for_grid[(budget0, "Random")],
                chosen_for_grid[(budget0, "Disc-Hard")],
            ],
            n_cols=args.grid_cols,
        )

    manifest = {
        "dgen_path": str(Path(args.dgen_path).resolve()),
        "ddisc_path": str(Path(args.ddisc_path).resolve()),
        "cifar_root": str(Path(args.cifar_root).resolve()),
        "budgets": budgets,
        "seeds": seeds,
        "outputs": {
            "per_seed_csv": str((out_dir / "disc_hard_failure_per_seed.csv").resolve()),
            "summary_csv": str((out_dir / "disc_hard_failure_summary.csv").resolve()),
            "table_md": str((out_dir / "disc_hard_failure_table.md").resolve()),
            "occupancy_png": str((out_dir / "disc_hard_subset_occupancy.png").resolve()),
            "grid_png": str((out_dir / "disc_hard_examples_budget0.1.png").resolve()),
        },
        "note": "Logit-margin diagnostics were not available in cached artifacts and require a working PyTorch inference environment.",
    }
    (out_dir / "disc_hard_failure_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def cmd_env_info(args: argparse.Namespace) -> None:
    import importlib
    import platform

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "modules": {},
    }
    modules = ["numpy", "scipy", "sklearn", "pandas", "matplotlib", "torch", "seaborn", "statsmodels"]
    for name in modules:
        try:
            mod = importlib.import_module(name)
            info["modules"][name] = {"available": True, "version": getattr(mod, "__version__", None)}
        except Exception as exc:
            info["modules"][name] = {"available": False, "error": type(exc).__name__}
    print(json.dumps(info, indent=2))


def cmd_pdf_excerpt(args: argparse.Namespace) -> None:
    raise RuntimeError("Use pdftotext from the shell for PDF excerpts in this environment.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    ap_npy = sub.add_parser("probe-npy")
    ap_npy.add_argument("path")
    ap_npy.set_defaults(func=cmd_probe_npy)

    ap_json = sub.add_parser("probe-json")
    ap_json.add_argument("path")
    ap_json.set_defaults(func=cmd_probe_json)

    ap_search = sub.add_parser("search-text")
    ap_search.add_argument("root")
    ap_search.add_argument("pattern")
    ap_search.add_argument("--limit", type=int, default=100)
    ap_search.set_defaults(func=cmd_search_text)

    ap_sel = sub.add_parser("selection-stats")
    ap_sel.add_argument("--alpha-mix-json", required=True)
    ap_sel.add_argument("--out-dir", required=True)
    ap_sel.add_argument("--baseline", default="Random")
    ap_sel.add_argument("--compare-methods", default="OHS-XOR(p75),Gen-Hard,Disc-Hard")
    ap_sel.set_defaults(func=cmd_selection_stats)

    ap_corr = sub.add_parser("corr-report")
    ap_corr.add_argument("--x-path", required=True)
    ap_corr.add_argument("--y-path", required=True)
    ap_corr.add_argument("--out-json", default="")
    ap_corr.set_defaults(func=cmd_corr_report)

    ap_nlr = sub.add_parser("nonlinear-r2")
    ap_nlr.add_argument("--x-path", required=True)
    ap_nlr.add_argument("--y-path", required=True)
    ap_nlr.add_argument("--x-name", default="D_gen")
    ap_nlr.add_argument("--y-name", default="D_disc")
    ap_nlr.add_argument("--out-dir", required=True)
    ap_nlr.add_argument("--split-seeds", default="0,1,2")
    ap_nlr.add_argument("--max-train-samples", type=int, default=20000)
    ap_nlr.add_argument("--bidirectional", action="store_true")
    ap_nlr.set_defaults(func=cmd_nonlinear_r2)

    ap_disc = sub.add_parser("disc-hard-failure")
    ap_disc.add_argument("--dgen-path", required=True)
    ap_disc.add_argument("--ddisc-path", required=True)
    ap_disc.add_argument("--cifar-root", required=True)
    ap_disc.add_argument("--out-dir", required=True)
    ap_disc.add_argument("--budgets", default="0.1,0.3")
    ap_disc.add_argument("--seeds", default="0,1,2")
    ap_disc.add_argument("--background-points", type=int, default=8000)
    ap_disc.add_argument("--overlay-points", type=int, default=3000)
    ap_disc.add_argument("--grid-cols", type=int, default=10)
    ap_disc.set_defaults(func=cmd_disc_hard_failure)

    ap_td = sub.add_parser("training-dynamics-report")
    ap_td.add_argument("--dgen-path", required=True)
    ap_td.add_argument("--forgetting-path", required=True)
    ap_td.add_argument("--first-learning-path", required=True)
    ap_td.add_argument("--out-dir", required=True)
    ap_td.add_argument("--dgen-name", default="D_gen")
    ap_td.add_argument("--epochs", type=int, default=50)
    ap_td.add_argument("--never-value", default="epochs_plus_one")
    ap_td.add_argument("--split-seeds", default="0,1,2")
    ap_td.add_argument("--max-train-samples", type=int, default=20000)
    ap_td.add_argument("--scatter-points", type=int, default=8000)
    ap_td.add_argument("--n-bins", type=int, default=20)
    ap_td.add_argument("--plot-seed", type=int, default=0)
    ap_td.set_defaults(func=cmd_training_dynamics_report)

    ap_env = sub.add_parser("env-info")
    ap_env.set_defaults(func=cmd_env_info)

    ap_pdf = sub.add_parser("pdf-excerpt")
    ap_pdf.add_argument("path")
    ap_pdf.add_argument("--start-page", type=int, default=1)
    ap_pdf.add_argument("--end-page", type=int, default=1)
    ap_pdf.add_argument("--max-lines", type=int, default=80)
    ap_pdf.set_defaults(func=cmd_pdf_excerpt)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
