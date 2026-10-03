#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def ci95(vals: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    if vals.size <= 1:
        x = float(vals.mean()) if vals.size else float("nan")
        return x, x
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1))
    half = 1.96 * sd / math.sqrt(vals.size)
    return mean - half, mean + half


def load_named_array(path_spec: str) -> tuple[str, np.ndarray, str]:
    name, path = path_spec.split("=", 1)
    arr = np.load(path).reshape(-1).astype(float)
    return name, arr, str(Path(path).resolve())


def fit_linear(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return float(r2_score(y_test, pred))


def fit_histgbdt(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, seed: int) -> float:
    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return float(r2_score(y_test, pred))


def maybe_subsample(X_train: np.ndarray, y_train: np.ndarray, max_train_samples: int, seed: int):
    if max_train_samples > 0 and len(X_train) > max_train_samples:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(X_train), size=max_train_samples, replace=False)
        X_train = X_train[keep]
        y_train = y_train[keep]
    return X_train, y_train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", action="append", default=[], help="name=user_inputs/path.npy")
    ap.add_argument("--target", action="append", default=[], help="name=user_inputs/path.npy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split-seeds", default="0,1,2,3,4")
    ap.add_argument("--max-train-samples", type=int, default=40000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proxy_items = [load_named_array(s) for s in args.proxy]
    target_items = [load_named_array(s) for s in args.target]
    seeds = [int(s) for s in args.split_seeds.split(",") if s.strip()]

    proxies = {name: arr for name, arr, _ in proxy_items}
    proxy_paths = {name: path for name, _, path in proxy_items}
    targets = {name: arr for name, arr, _ in target_items}
    target_paths = {name: path for name, _, path in target_items}

    proxy_names = list(proxies.keys())
    X_all = np.column_stack([proxies[name] for name in proxy_names]).astype(float)

    per_split_rows = []
    summary_rows = []

    for target_name, y_all in targets.items():
        valid_mask = np.isfinite(y_all)
        for name in proxy_names:
            valid_mask &= np.isfinite(proxies[name])

        X = X_all[valid_mask]
        y = y_all[valid_mask]

        # Single-proxy linear and nonlinear.
        single_rows = []
        for proxy_idx, proxy_name in enumerate(proxy_names):
            x = X[:, [proxy_idx]]
            for seed in seeds:
                X_train, X_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.2, random_state=seed, shuffle=True
                )
                X_train, y_train = maybe_subsample(X_train, y_train, args.max_train_samples, seed)
                lin_r2 = fit_linear(X_train, y_train, X_test, y_test)
                gbdt_r2 = fit_histgbdt(X_train, y_train, X_test, y_test, seed)
                rows = [
                    {"model": "LinearRegression", "r2": lin_r2},
                    {"model": "HistGBDT", "r2": gbdt_r2},
                    {"model": "MeanPredictor", "r2": 0.0},
                ]
                for row in rows:
                    payload = {
                        "target": target_name,
                        "feature_set": proxy_name,
                        "feature_type": "single",
                        "model": row["model"],
                        "split_seed": seed,
                        "r2": row["r2"],
                    }
                    single_rows.append(payload)
                    per_split_rows.append(payload)

        single_df = pd.DataFrame(single_rows)
        linear_summary = (
            single_df[single_df["model"] == "LinearRegression"]
            .groupby(["feature_set", "model"], as_index=False)
            .agg(mean_r2=("r2", "mean"), std_r2=("r2", "std"))
            .sort_values("mean_r2", ascending=False)
        )
        nonlinear_summary = (
            single_df[single_df["model"] == "HistGBDT"]
            .groupby(["feature_set", "model"], as_index=False)
            .agg(mean_r2=("r2", "mean"), std_r2=("r2", "std"))
            .sort_values("mean_r2", ascending=False)
        )
        best_single_linear = linear_summary.iloc[0]
        best_single_nonlinear = nonlinear_summary.iloc[0]

        # Multi-proxy nonlinear.
        multi_rows = []
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed, shuffle=True
            )
            X_train, y_train = maybe_subsample(X_train, y_train, args.max_train_samples, seed)
            lin_r2 = fit_linear(X_train, y_train, X_test, y_test)
            gbdt_r2 = fit_histgbdt(X_train, y_train, X_test, y_test, seed)
            rows = [
                {"model": "LinearRegression", "r2": lin_r2},
                {"model": "HistGBDT", "r2": gbdt_r2},
                {"model": "MeanPredictor", "r2": 0.0},
            ]
            for row in rows:
                payload = {
                    "target": target_name,
                    "feature_set": "+".join(proxy_names),
                    "feature_type": "multi",
                    "model": row["model"],
                    "split_seed": seed,
                    "r2": row["r2"],
                }
                multi_rows.append(payload)
                per_split_rows.append(payload)

        multi_df = pd.DataFrame(multi_rows)
        multi_summary = (
            multi_df[multi_df["model"] == "HistGBDT"]
            .groupby(["feature_set", "model"], as_index=False)
            .agg(mean_r2=("r2", "mean"), std_r2=("r2", "std"))
            .sort_values("mean_r2", ascending=False)
        )
        best_multi_nonlinear = multi_summary.iloc[0]

        def pick_vals(df: pd.DataFrame, feature_set: str, model: str) -> np.ndarray:
            return df[(df["feature_set"] == feature_set) & (df["model"] == model)]["r2"].to_numpy(dtype=float)

        lin_vals = pick_vals(single_df, best_single_linear["feature_set"], "LinearRegression")
        nonlin_vals = pick_vals(single_df, best_single_nonlinear["feature_set"], "HistGBDT")
        multi_vals = pick_vals(multi_df, best_multi_nonlinear["feature_set"], "HistGBDT")
        lin_lo, lin_hi = ci95(lin_vals)
        nonlin_lo, nonlin_hi = ci95(nonlin_vals)
        multi_lo, multi_hi = ci95(multi_vals)

        summary_rows.append(
            {
                "target": target_name,
                "n": int(len(y)),
                "best_single_linear_proxy": best_single_linear["feature_set"],
                "best_single_linear_r2": float(best_single_linear["mean_r2"]),
                "best_single_linear_ci95_low": lin_lo,
                "best_single_linear_ci95_high": lin_hi,
                "best_single_nonlinear_proxy": best_single_nonlinear["feature_set"],
                "best_single_nonlinear_model": "HistGBDT",
                "best_single_nonlinear_r2": float(best_single_nonlinear["mean_r2"]),
                "best_single_nonlinear_ci95_low": nonlin_lo,
                "best_single_nonlinear_ci95_high": nonlin_hi,
                "multi_proxy_model": "HistGBDT",
                "multi_proxy_nonlinear_r2": float(best_multi_nonlinear["mean_r2"]),
                "multi_proxy_ci95_low": multi_lo,
                "multi_proxy_ci95_high": multi_hi,
            }
        )

    per_split_df = pd.DataFrame(per_split_rows)
    summary_df = pd.DataFrame(summary_rows)
    per_split_df.to_csv(out_dir / "predictive_power_per_split.csv", index=False)
    summary_df.to_csv(out_dir / "predictive_power_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(summary_df))
    w = 0.26
    ax.bar(x - w, summary_df["best_single_linear_r2"], width=w, label="Best single linear")
    ax.bar(x, summary_df["best_single_nonlinear_r2"], width=w, label="Best single nonlinear")
    ax.bar(x + w, summary_df["multi_proxy_nonlinear_r2"], width=w, label="Multi-proxy nonlinear")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["target"], rotation=25, ha="right")
    ax.set_ylabel("Held-out $R^2$")
    ax.set_title("Held-out Predictive Power of Generative Proxies (Fast Sweep)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "predictive_power_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "proxies": proxy_paths,
        "targets": target_paths,
        "split_seeds": seeds,
        "max_train_samples": int(args.max_train_samples),
        "models": ["LinearRegression", "HistGBDT", "MeanPredictor"],
        "outputs": {
            "per_split_csv": str((out_dir / "predictive_power_per_split.csv").resolve()),
            "summary_csv": str((out_dir / "predictive_power_summary.csv").resolve()),
            "figure_png": str((out_dir / "predictive_power_summary.png").resolve()),
        },
    }
    (out_dir / "predictive_power_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
