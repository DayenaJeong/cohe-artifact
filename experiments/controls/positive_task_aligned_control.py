#!/usr/bin/env python3
"""Task-aligned positive calibration control for COHE.

This script is intentionally narrow: it either scores a torchvision ImageNet
classifier with deterministic evaluation transforms, or computes Gate 1/2/3
metrics from matched per-sample CE arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def build_model(name: str):
    import torchvision

    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model for deterministic scoring: {name}")
    return model, weights


def score_imagenet(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    import torchvision
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    class IndexedDataset(Dataset):
        def __init__(self, base: Dataset) -> None:
            self.base = base

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, i: int):
            x, y = self.base[i]
            return x, y, i

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, weights = build_model(args.model)
    model = model.to(device)
    model.eval()
    ds = torchvision.datasets.ImageNet(
        root=args.imagenet_root,
        split=args.split,
        transform=weights.transforms(),
    )
    loader = DataLoader(
        IndexedDataset(ds),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )

    all_idx: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_ce: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, idx in tqdm(loader, desc=f"score {args.model}", ncols=100):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            ce = F.cross_entropy(logits, y, reduction="none")
            all_idx.append(idx.numpy())
            all_targets.append(y.cpu().numpy())
            all_ce.append(ce.cpu().numpy())

    np.savez_compressed(
        out_path,
        indices=np.concatenate(all_idx).astype(np.int64),
        targets=np.concatenate(all_targets).astype(np.int64),
        ce_loss=np.concatenate(all_ce).astype(np.float32),
    )
    meta = {
        "model": args.model,
        "root": "<imagenet-root>",
        "split": args.split,
        "output_npz": str(out_path),
        "n_samples": int(sum(len(x) for x in all_idx)),
        "transform": "torchvision pretrained weights deterministic eval transform",
        "score": "per-sample cross-entropy; higher means harder",
    }
    out_path.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def load_ce(path: Path, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        data = np.load(path)
        ce = data["ce_loss"].astype(float)
        idx = data["indices"].astype(np.int64)
        if n is None:
            n = int(idx.max()) + 1
        out = np.full(n, np.nan, dtype=float)
        out[idx] = ce
        if np.isnan(out).any():
            raise RuntimeError(f"{path} does not cover all {n} samples")
        return np.arange(n, dtype=np.int64), out
    arr = np.load(path).astype(float).reshape(-1)
    return np.arange(arr.shape[0], dtype=np.int64), arr


def fit_linear_r2(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    x_mean = float(x_train.mean())
    y_mean = float(y_train.mean())
    var = float(((x_train - x_mean) ** 2).sum())
    if var <= 0:
        pred = np.full_like(y_test, y_mean, dtype=float)
    else:
        slope = float(((x_train - x_mean) * (y_train - y_mean)).sum() / var)
        intercept = y_mean - slope * x_mean
        pred = intercept + slope * x_test
    ss_res = float(((y_test - pred) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    return 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def ci95(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    mean = float(values.mean())
    half = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean - half, mean + half


def recall_at_q(proxy: np.ndarray, target: np.ndarray, q: float) -> float:
    n = len(proxy)
    k = int(round(q * n))
    proxy_top = np.argpartition(-proxy, k - 1)[:k]
    target_top = np.argpartition(-target, k - 1)[:k]
    return float(np.intersect1d(proxy_top, target_top, assume_unique=False).size / k)


def compute_metrics(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_id_proxy, proxy = load_ce(Path(args.proxy_array), args.n)
    sample_id_target, target = load_ce(Path(args.target_array), len(proxy))
    if len(proxy) != len(target):
        raise RuntimeError(f"Length mismatch: proxy={len(proxy)}, target={len(target)}")
    if not np.array_equal(sample_id_proxy, sample_id_target):
        raise RuntimeError("Sample IDs do not match")

    matched_csv = out_dir / "matched_proxy_target_ce.csv"
    with matched_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "proxy_ce", "target_ce"])
        for i, p, t in zip(sample_id_proxy, proxy, target):
            writer.writerow([int(i), float(p), float(t)])

    pearson = float(pearsonr(proxy, target).statistic)
    spearman = float(spearmanr(proxy, target).statistic)

    split_r2 = []
    rng_master = np.random.default_rng(args.seed)
    for seed in range(args.split_seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(proxy))
        n_test = int(round(args.test_fraction * len(proxy)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        split_r2.append(
            fit_linear_r2(proxy[train_idx], target[train_idx], proxy[test_idx], target[test_idx])
        )
    split_r2_arr = np.asarray(split_r2, dtype=float)
    r2_ci = ci95(split_r2_arr)

    null_r2 = []
    for _ in range(args.null_permutations):
        shuffled = proxy[rng_master.permutation(len(proxy))]
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        perm = rng.permutation(len(proxy))
        n_test = int(round(args.test_fraction * len(proxy)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        null_r2.append(
            fit_linear_r2(shuffled[train_idx], target[train_idx], shuffled[test_idx], target[test_idx])
        )
    null_r2_arr = np.asarray(null_r2, dtype=float)

    gate3_rows = []
    for q in args.q:
        observed = recall_at_q(proxy, target, q)
        null_recalls = []
        for _ in range(args.null_permutations):
            shuffled = proxy[rng_master.permutation(len(proxy))]
            null_recalls.append(recall_at_q(shuffled, target, q))
        null_arr = np.asarray(null_recalls, dtype=float)
        gate3_rows.append(
            {
                "q": q,
                "recall": observed,
                "random_expected_recall": q,
                "delta_vs_random": observed - q,
                "enrichment": observed / q,
                "null_mean": float(null_arr.mean()),
                "null_std": float(null_arr.std(ddof=1)),
                "null_p2_5": float(np.percentile(null_arr, 2.5)),
                "null_p97_5": float(np.percentile(null_arr, 97.5)),
            }
        )

    summary = {
        "proxy": args.proxy_name,
        "target": args.target_name,
        "n": int(len(proxy)),
        "matched_csv": str(matched_csv),
        "gate1": {
            "pearson_r": pearson,
            "spearman_rho": spearman,
        },
        "gate2": {
            "model": "univariate linear regression",
            "split_seeds": int(args.split_seeds),
            "test_fraction": float(args.test_fraction),
            "r2_mean": float(split_r2_arr.mean()),
            "r2_std": float(split_r2_arr.std(ddof=1)),
            "r2_ci95_low": r2_ci[0],
            "r2_ci95_high": r2_ci[1],
            "shuffled_null_mean": float(null_r2_arr.mean()),
            "shuffled_null_std": float(null_r2_arr.std(ddof=1)),
            "shuffled_null_p97_5": float(np.percentile(null_r2_arr, 97.5)),
        },
        "gate3": gate3_rows,
        "claim_boundary": "retrieval/triage only; not subset-retraining actionability",
    }
    (out_dir / "positive_task_aligned_control_summary.json").write_text(json.dumps(summary, indent=2))

    with (out_dir / "positive_task_aligned_control_summary.csv").open("w", newline="") as f:
        fieldnames = [
            "proxy",
            "target",
            "n",
            "pearson_r",
            "spearman_rho",
            "r2_mean",
            "r2_ci95_low",
            "r2_ci95_high",
            "r2_null_mean",
            "q",
            "recall",
            "delta_vs_random",
            "enrichment",
            "null_mean",
            "random_expected_recall",
            "null_std",
            "null_p2_5",
            "null_p97_5",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in gate3_rows:
            writer.writerow(
                {
                    "proxy": args.proxy_name,
                    "target": args.target_name,
                    "n": int(len(proxy)),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "r2_mean": float(split_r2_arr.mean()),
                    "r2_ci95_low": r2_ci[0],
                    "r2_ci95_high": r2_ci[1],
                    "r2_null_mean": float(null_r2_arr.mean()),
                    **row,
                }
            )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    score = sub.add_parser("score-imagenet")
    score.add_argument("--imagenet-root", required=True)
    score.add_argument("--split", default="val")
    score.add_argument("--model", required=True, choices=["resnet18", "resnet50"])
    score.add_argument("--out-npz", required=True)
    score.add_argument("--batch-size", type=int, default=256)
    score.add_argument("--num-workers", type=int, default=8)
    score.add_argument("--device", default="cuda")

    metrics = sub.add_parser("compute")
    metrics.add_argument("--proxy-array", required=True)
    metrics.add_argument("--target-array", required=True)
    metrics.add_argument("--proxy-name", required=True)
    metrics.add_argument("--target-name", required=True)
    metrics.add_argument("--out-dir", required=True)
    metrics.add_argument("--n", type=int, default=None)
    metrics.add_argument("--split-seeds", type=int, default=5)
    metrics.add_argument("--test-fraction", type=float, default=0.2)
    metrics.add_argument("--null-permutations", type=int, default=100)
    metrics.add_argument("--q", type=float, nargs="+", default=[0.1, 0.3])
    metrics.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.cmd == "score-imagenet":
        score_imagenet(args)
    elif args.cmd == "compute":
        compute_metrics(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
