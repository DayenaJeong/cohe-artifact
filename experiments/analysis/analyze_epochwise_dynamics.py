#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_named_array(spec: str) -> tuple[str, np.ndarray]:
    name, path = spec.split("=", 1)
    arr = np.load(path).reshape(-1).astype(float)
    return name, arr


def corr_payload(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan"), float("nan")
    return float(stats.pearsonr(x, y).statistic), float(stats.spearmanr(x, y).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--correctness-path", required=True)
    ap.add_argument("--proxy", action="append", default=[], help="name=user_inputs/path.npy")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    correctness = np.load(args.correctness_path).astype(np.int8)
    if correctness.ndim != 2:
        raise ValueError("correctness array must be 2D [n, epochs]")
    n, num_epochs = correctness.shape

    transitions = (correctness[:, :-1] == 1) & (correctness[:, 1:] == 0)
    cumulative_forgetting = np.concatenate(
        [np.zeros((n, 1), dtype=np.int32), np.cumsum(transitions.astype(np.int32), axis=1)],
        axis=1,
    )

    rows = []
    proxies = dict(load_named_array(s) for s in args.proxy)
    for proxy_name, proxy_arr in proxies.items():
        if proxy_arr.shape[0] != n:
            raise ValueError(f"Proxy {proxy_name} length {proxy_arr.shape[0]} != correctness n {n}")
        for epoch_idx in range(num_epochs):
            p_corr, s_corr = corr_payload(proxy_arr, cumulative_forgetting[:, epoch_idx].astype(float))
            p_cur, s_cur = corr_payload(proxy_arr, correctness[:, epoch_idx].astype(float))
            rows.append(
                {
                    "proxy": proxy_name,
                    "epoch": epoch_idx + 1,
                    "metric": "cumulative_forgetting",
                    "pearson_r": p_corr,
                    "spearman_rho": s_corr,
                }
            )
            rows.append(
                {
                    "proxy": proxy_name,
                    "epoch": epoch_idx + 1,
                    "metric": "current_correctness",
                    "pearson_r": p_cur,
                    "spearman_rho": s_cur,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "epochwise_dynamics_correlations.csv"
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, metric, title in zip(
        axes,
        ["cumulative_forgetting", "current_correctness"],
        ["Cumulative Forgetting", "Current Correctness"],
    ):
        sub = df[df["metric"] == metric]
        for proxy_name in proxies.keys():
            one = sub[sub["proxy"] == proxy_name]
            ax.plot(one["epoch"], one["spearman_rho"], marker="o", markersize=2.5, linewidth=1.4, label=proxy_name)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Spearman rho")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Generative Proxies vs Training Dynamics Across Epochs", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "epochwise_dynamics_correlations.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "correctness_path": str(Path(args.correctness_path).resolve()),
        "n": int(n),
        "num_epochs": int(num_epochs),
        "proxies": list(proxies.keys()),
        "outputs": {
            "csv": str(csv_path.resolve()),
            "figure_png": str((out_dir / "epochwise_dynamics_correlations.png").resolve()),
        },
    }
    (out_dir / "epochwise_dynamics_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
