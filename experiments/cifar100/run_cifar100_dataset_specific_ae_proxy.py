#!/usr/bin/env python3
"""Train a CIFAR-100-specific autoencoder proxy and audit it against targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


@dataclass
class AEConfig:
    seed: int = 0
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 4
    data_root: str = "./data"


class ConvAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ci95(vals: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    if vals.size <= 1:
        v = float(vals.mean()) if vals.size else float("nan")
        return v, v
    half = 1.96 * float(vals.std(ddof=1)) / math.sqrt(vals.size)
    mean = float(vals.mean())
    return mean - half, mean + half


def fit_r2_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGBDT": HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            random_state=seed,
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
                random_state=seed,
            ),
        ),
    }
    rows: list[dict[str, object]] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append({"model": name, "r2": float(r2_score(y_test, pred))})
    rows.append({"model": "MeanPredictor", "r2": 0.0})
    return rows


def train_autoencoder(cfg: AEConfig, out_dir: Path, device: torch.device) -> tuple[ConvAutoencoder, list[dict[str, object]]]:
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR100(root=cfg.data_root, train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    model = ConvAutoencoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()
    rows: list[dict[str, object]] = []
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for inputs, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon = model(inputs)
            loss = criterion(recon, inputs)
            loss.backward()
            optimizer.step()
            bs = int(inputs.shape[0])
            total_loss += float(loss.item()) * bs
            total_seen += bs
        mean_loss = total_loss / max(1, total_seen)
        rows.append({"epoch": epoch + 1, "train_mse": mean_loss})
        if (epoch + 1) % max(1, cfg.epochs // 10) == 0:
            print(f"AE epoch {epoch + 1:03d}/{cfg.epochs} mse={mean_loss:.6f}", flush=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, ckpt_dir / "cifar100_conv_ae_seed0.pt")
    return model, rows


def score_autoencoder(model: nn.Module, cfg: AEConfig, out_dir: Path, device: torch.device) -> np.ndarray:
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR100(root=cfg.data_root, train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            recon = model(inputs)
            mse = ((recon - inputs) ** 2).flatten(1).mean(dim=1)
            scores.append(mse.detach().cpu().numpy())
    arr = np.concatenate(scores).astype(np.float64)
    arrays_dir = out_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.save(arrays_dir / "cifar100_conv_ae_recon_mse.npy", arr)
    return arr


def audit_proxy(
    proxy: np.ndarray,
    targets: dict[str, np.ndarray],
    out_dir: Path,
    split_seeds: list[int],
    max_train_samples: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    corr_rows: list[dict[str, object]] = []
    per_split_rows: list[dict[str, object]] = []
    r2_summary_rows: list[dict[str, object]] = []
    x_all = proxy.reshape(-1, 1).astype(float)

    for target_name, y_raw in targets.items():
        y_all = y_raw.reshape(-1).astype(float)
        valid = np.isfinite(proxy) & np.isfinite(y_all)
        x = x_all[valid]
        y = y_all[valid]
        pearson_r, pearson_p = stats.pearsonr(x[:, 0], y)
        spearman_rho, spearman_p = stats.spearmanr(x[:, 0], y)
        corr_rows.append(
            {
                "proxy": "CIFAR-100 ConvAE reconstruction",
                "target": target_name,
                "n": int(len(y)),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_rho": float(spearman_rho),
                "spearman_p": float(spearman_p),
            }
        )

        for split_seed in split_seeds:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=split_seed, shuffle=True)
            if max_train_samples > 0 and len(x_train) > max_train_samples:
                rng = np.random.default_rng(split_seed)
                keep = rng.choice(len(x_train), size=max_train_samples, replace=False)
                x_train = x_train[keep]
                y_train = y_train[keep]
            for row in fit_r2_models(x_train, y_train, x_test, y_test, split_seed):
                per_split_rows.append(
                    {
                        "proxy": "CIFAR-100 ConvAE reconstruction",
                        "target": target_name,
                        "split_seed": split_seed,
                        "model": row["model"],
                        "r2": float(row["r2"]),
                    }
                )

        target_rows = [row for row in per_split_rows if row["target"] == target_name]
        model_names = sorted({str(row["model"]) for row in target_rows})
        linear_vals = np.array([float(row["r2"]) for row in target_rows if row["model"] == "LinearRegression"], dtype=float)
        nonlinear_candidates = [m for m in model_names if m not in {"LinearRegression", "MeanPredictor"}]
        nonlinear_means = {
            model_name: np.array([float(row["r2"]) for row in target_rows if row["model"] == model_name], dtype=float)
            for model_name in nonlinear_candidates
        }
        best_nonlinear_model = max(nonlinear_means, key=lambda name: float(nonlinear_means[name].mean()))
        best_nonlinear_vals = nonlinear_means[best_nonlinear_model]
        lin_lo, lin_hi = ci95(linear_vals)
        nonlin_lo, nonlin_hi = ci95(best_nonlinear_vals)
        r2_summary_rows.append(
            {
                "proxy": "CIFAR-100 ConvAE reconstruction",
                "target": target_name,
                "linear_r2_mean": float(linear_vals.mean()),
                "linear_r2_ci95_low": lin_lo,
                "linear_r2_ci95_high": lin_hi,
                "best_nonlinear_model": best_nonlinear_model,
                "best_nonlinear_r2_mean": float(best_nonlinear_vals.mean()),
                "best_nonlinear_r2_ci95_low": nonlin_lo,
                "best_nonlinear_r2_ci95_high": nonlin_hi,
                "split_seeds": " ".join(str(s) for s in split_seeds),
            }
        )
    return corr_rows, per_split_rows, r2_summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ce-path", type=Path, required=True)
    parser.add_argument("--margin-path", type=Path, required=True)
    parser.add_argument("--gradnorm-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--max-train-samples", type=int, default=40000)
    parser.add_argument("--reuse-score", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = AEConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        data_root=str(args.data_root),
    )
    set_seed(cfg.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if args.reuse_score is not None:
        proxy = np.load(args.reuse_score).astype(np.float64)
        train_rows: list[dict[str, object]] = []
    else:
        model, train_rows = train_autoencoder(cfg, args.out_dir, device)
        proxy = score_autoencoder(model, cfg, args.out_dir, device)

    targets = {
        "CE loss": np.load(args.ce_path).astype(np.float64),
        "Margin hardness": np.load(args.margin_path).astype(np.float64),
        "GradNorm": np.load(args.gradnorm_path).astype(np.float64),
    }
    for target_name, arr in targets.items():
        if len(arr) != len(proxy):
            raise ValueError(f"{target_name} length mismatch: {len(arr)} vs {len(proxy)}")

    split_seeds = [int(s.strip()) for s in args.split_seeds.split(",") if s.strip()]
    corr_rows, per_split_rows, r2_summary_rows = audit_proxy(proxy, targets, args.out_dir, split_seeds, args.max_train_samples)
    write_csv(args.out_dir / "training_curve.csv", train_rows)
    write_csv(args.out_dir / "dataset_specific_ae_correlation_summary.csv", corr_rows)
    write_csv(args.out_dir / "dataset_specific_ae_predictive_r2_per_split.csv", per_split_rows)
    write_csv(args.out_dir / "dataset_specific_ae_predictive_r2_summary.csv", r2_summary_rows)

    manifest = {
        "proxy": "CIFAR-100 ConvAE reconstruction",
        "score_direction": "higher reconstruction MSE means higher reconstruction hardness",
        "dataset": "CIFAR-100 train",
        "config": asdict(cfg),
        "split_seeds": split_seeds,
        "outputs": [
            "arrays/cifar100_conv_ae_recon_mse.npy",
            "dataset_specific_ae_correlation_summary.csv",
            "dataset_specific_ae_predictive_r2_summary.csv",
            "dataset_specific_ae_predictive_r2_per_split.csv",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
