#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSIC permutation test between D_gen and D_disc.

Example:
  python hsic_test.py \
    --gen user_inputs/ICML/train_d_gen.npy \
    --disc user_inputs/ICML/train_d_disc.npy \
    --subsample 5000 \
    --n_perm 200 \
    --seed 0 \
    --out user_inputs/ICML/hsic_results.json
"""

import argparse
import json
import math
from typing import Dict, Tuple

import numpy as np

# Optional correlations (safe fallback if scipy isn't available)
try:
    from scipy.stats import pearsonr, spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _median_heuristic_sigma(x: np.ndarray) -> float:
    """
    Median heuristic for RBF bandwidth sigma.
    x: (n, 1)
    Returns sigma > 0
    """
    x = x.reshape(-1, 1).astype(np.float64)
    n = x.shape[0]
    if n < 2:
        return 1.0
    # Compute pairwise distances efficiently with broadcasting for 1D
    # dist_ij = |x_i - x_j|
    diffs = np.abs(x - x.T)
    # Take upper triangle excluding diagonal
    iu = np.triu_indices(n, k=1)
    d = diffs[iu]
    med = np.median(d)
    if not np.isfinite(med) or med <= 1e-12:
        return 1.0
    return float(med)


def _rbf_kernel_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    RBF kernel for 1D vectors.
    x: (n, 1)
    sigma: bandwidth > 0
    returns K: (n, n)
    """
    x = x.reshape(-1, 1).astype(np.float64)
    sigma = float(max(sigma, 1e-12))
    # squared distances for 1D: (x_i - x_j)^2
    sq = (x - x.T) ** 2
    K = np.exp(-sq / (2.0 * sigma * sigma))
    return K


def _center_gram(K: np.ndarray) -> np.ndarray:
    """
    Center a Gram matrix: Kc = H K H where H = I - 1/n 11^T
    """
    n = K.shape[0]
    one_n = np.ones((n, n), dtype=np.float64) / n
    Kc = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return Kc


def hsic_unbiased_rbf(x: np.ndarray, y: np.ndarray, sigma_x: float = None, sigma_y: float = None) -> float:
    """
    HSIC (biased estimator) with RBF kernels, Gram-centered.
    For permutation test this is totally fine (consistent comparison vs null).

    x, y: (n,) arrays
    sigma_x, sigma_y: optional bandwidths; if None, uses median heuristic.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    assert x.shape[0] == y.shape[0], "x and y must have same length"
    n = x.shape[0]
    if n < 3:
        return 0.0

    x1 = x.reshape(-1, 1)
    y1 = y.reshape(-1, 1)

    if sigma_x is None:
        sigma_x = _median_heuristic_sigma(x1)
    if sigma_y is None:
        sigma_y = _median_heuristic_sigma(y1)

    K = _rbf_kernel_1d(x1, sigma_x)
    L = _rbf_kernel_1d(y1, sigma_y)

    Kc = _center_gram(K)
    Lc = _center_gram(L)

    # HSIC biased estimator: (1/n^2) trace(Kc Lc)
    hsic = np.trace(Kc @ Lc) / (n * n)
    return float(hsic)


def hsic_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 200,
    subsample: int = 5000,
    seed: int = 0,
    bandwidth: str = "median",
) -> Dict:
    """
    Permutation test for HSIC (RBF kernel).
    Returns dict with observed_hsic, p_value, null_mean, null_std, n_used, sigmas.

    - subsample: if len(x) > subsample, random subset is used to keep O(n^2) feasible.
    - bandwidth: 'median' currently.
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    assert x.shape[0] == y.shape[0], "x and y must have same length"

    n_all = x.shape[0]
    if subsample is not None and n_all > subsample:
        idx = rng.choice(n_all, size=subsample, replace=False)
        x = x[idx]
        y = y[idx]

    n = x.shape[0]
    if n < 3:
        return {
            "n_used": int(n),
            "observed_hsic": 0.0,
            "p_value": 1.0,
            "null_mean": 0.0,
            "null_std": 0.0,
            "sigma_x": None,
            "sigma_y": None,
            "n_perm": int(n_perm),
            "seed": int(seed),
        }

    # fix bandwidths from original to keep fair under permutations
    sigma_x = _median_heuristic_sigma(x.reshape(-1, 1))
    sigma_y = _median_heuristic_sigma(y.reshape(-1, 1))

    obs = hsic_unbiased_rbf(x, y, sigma_x=sigma_x, sigma_y=sigma_y)

    null_vals = np.empty(n_perm, dtype=np.float64)
    for b in range(n_perm):
        yp = rng.permutation(y)
        null_vals[b] = hsic_unbiased_rbf(x, yp, sigma_x=sigma_x, sigma_y=sigma_y)

    # p-value with +1 smoothing
    p = (1.0 + np.sum(null_vals >= obs)) / (n_perm + 1.0)

    return {
        "n_used": int(n),
        "observed_hsic": float(obs),
        "p_value": float(p),
        "null_mean": float(null_vals.mean()),
        "null_std": float(null_vals.std(ddof=1) if n_perm > 1 else 0.0),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "n_perm": int(n_perm),
        "seed": int(seed),
    }


def _zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    mu = a.mean()
    sd = a.std()
    return (a - mu) / (sd + 1e-12)


def _corrs(x: np.ndarray, y: np.ndarray) -> Dict:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 3:
        return {"pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None}

    if _HAS_SCIPY:
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        return {
            "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp),
        }

    # fallback (no p-values)
    pr = np.corrcoef(x, y)[0, 1]
    # spearman fallback: rank then pearson
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    sr = np.corrcoef(rx, ry)[0, 1]
    return {
        "pearson_r": float(pr), "pearson_p": None,
        "spearman_rho": float(sr), "spearman_p": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=str, required=True, help="Path to D_gen .npy")
    ap.add_argument("--disc", type=str, required=True, help="Path to D_disc .npy")
    ap.add_argument("--subsample", type=int, default=5000, help="Subsample size for HSIC (O(n^2)).")
    ap.add_argument("--n_perm", type=int, default=200, help="Number of permutations.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--out", type=str, default="", help="Optional output JSON path.")
    args = ap.parse_args()

    D_gen = np.load(args.gen)
    D_disc = np.load(args.disc)

    # Flatten typical (N,1) too
    D_gen = np.asarray(D_gen).reshape(-1)
    D_disc = np.asarray(D_disc).reshape(-1)

    if D_gen.shape != D_disc.shape:
        raise ValueError(f"Shape mismatch: gen {D_gen.shape} vs disc {D_disc.shape}")

    # clean
    mask = np.isfinite(D_gen) & np.isfinite(D_disc)
    D_gen = D_gen[mask]
    D_disc = D_disc[mask]

    # normalize for stability
    D_gen_z = _zscore(D_gen)
    D_disc_z = _zscore(D_disc)

    corrs = _corrs(D_gen_z, D_disc_z)

    hsic_res = hsic_permutation_test(
        D_gen_z, D_disc_z,
        n_perm=args.n_perm,
        subsample=args.subsample,
        seed=args.seed
    )

    result = {
        "paths": {"gen": args.gen, "disc": args.disc},
        "n_total_after_clean": int(D_gen_z.shape[0]),
        "corrs_on_zscore": corrs,
        "hsic_rbf_permutation_on_zscore": hsic_res,
        "notes": {
            "hsic_estimator": "biased HSIC with centered Gram matrices; permutation test for dependence",
            "bandwidth": "median heuristic (fixed from unpermuted data)",
            "subsample": args.subsample,
        }
    }

    # pretty print
    print(json.dumps(result, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[Saved] {args.out}")


if __name__ == "__main__":
    main()
