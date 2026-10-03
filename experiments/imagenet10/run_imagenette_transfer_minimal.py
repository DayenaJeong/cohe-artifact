#!/usr/bin/env python
"""Run a minimal ImageNet-10 operational-transfer sanity check.

The script trains a ResNet-18 classifier on selected ImageNet-10 training
subsets and evaluates validation accuracy. It assumes train-split D_gen and
D_disc arrays are already computed in ImageFolder order.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    model_name: str
    budget: float
    subset_size: int
    seed: int
    method: str
    epochs: int
    batch_size: int
    train_backbone: bool


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.replace(",", " ").split() if part.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.replace(",", " ").split() if part.strip()]


def parse_methods(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def safe_name(method: str) -> str:
    return method.replace(" ", "").replace("/", "-").replace("(", "").replace(")", "")


def select_indices(dgen: np.ndarray, ddisc: np.ndarray, method: str, k: int, seed: int, percentile: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(dgen)
    if method == "Random":
        return rng.choice(n, size=k, replace=False)
    if method == "Gen-Hard":
        return np.argsort(-dgen, kind="mergesort")[:k]
    if method == "Disc-Hard":
        return np.argsort(-ddisc, kind="mergesort")[:k]
    if method == "OHS-XOR(p75)":
        gen_thr = np.percentile(dgen, percentile)
        disc_thr = np.percentile(ddisc, percentile)
        pool = np.flatnonzero((dgen > gen_thr) ^ (ddisc > disc_thr))
        if len(pool) >= k:
            return rng.choice(pool, size=k, replace=False)
        remainder = np.setdiff1d(np.arange(n), pool, assume_unique=False)
        extra = rng.choice(remainder, size=k - len(pool), replace=False)
        return np.concatenate([pool, extra])
    raise ValueError(f"Unknown method: {method}")


def build_model(num_classes: int, train_backbone: bool, device: torch.device) -> torch.nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    if not train_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    return model.to(device)


def train_eval(
    train_ds: ImageFolder,
    val_ds: ImageFolder,
    indices: np.ndarray,
    cfg: RunConfig,
    lr: float,
    num_workers: int,
    device: torch.device,
) -> float:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    subset = Subset(train_ds, indices.tolist())
    train_loader = DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model = build_model(len(train_ds.classes), cfg.train_backbone, device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"{cfg.method} seed={cfg.seed} budget={cfg.budget}: epoch {epoch + 1:03d}/{cfg.epochs:03d} loss={total_loss / max(len(train_loader), 1):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return 100.0 * correct / max(total, 1)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("ICML/data/imagenet10"))
    parser.add_argument("--gen", type=Path, required=True)
    parser.add_argument("--disc", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("NeurIPS/results/imagenette_transfer_minimal"))
    parser.add_argument("--budgets", default="0.1,0.3")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default="Random,Gen-Hard,Disc-Hard,OHS-XOR(p75)")
    parser.add_argument("--ohs-percentile", type=float, default=75.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-backbone", action="store_true")
    args = parser.parse_args()

    dgen = np.load(args.gen).astype(float)
    ddisc = np.load(args.disc).astype(float)

    train_tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = ImageFolder(args.data_root / "train", transform=train_tfm)
    val_ds = ImageFolder(args.data_root / "val", transform=val_tfm)
    if len(train_ds) != len(dgen) or len(train_ds) != len(ddisc):
        raise ValueError(f"Score length mismatch: train={len(train_ds)} dgen={len(dgen)} ddisc={len(ddisc)}")

    device = torch.device(args.device)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for seed in parse_int_list(args.seeds):
        for budget in parse_float_list(args.budgets):
            k = int(round(len(train_ds) * budget))
            for method in parse_methods(args.methods):
                cfg = RunConfig(
                    dataset="ImageNet-10",
                    model_name="resnet18",
                    budget=budget,
                    subset_size=k,
                    seed=seed,
                    method=method,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    train_backbone=args.train_backbone,
                )
                tag = f"imagenet10_resnet18_budget{budget:.2f}_K{k}_seed{seed}_{safe_name(method)}"
                json_path = raw_dir / f"{tag}.json"
                if json_path.exists():
                    print(f"[skip] {tag}")
                    continue
                indices = select_indices(dgen, ddisc, method, k, seed, args.ohs_percentile)
                np.save(raw_dir / f"{tag}.indices.npy", indices)
                acc = train_eval(train_ds, val_ds, indices, cfg, args.lr, args.num_workers, device)
                payload = asdict(cfg)
                payload.update({"val_accuracy": acc, "indices_file": f"{tag}.indices.npy"})
                write_json(json_path, payload)
                print(f"[done] {tag} acc={acc:.2f}")


if __name__ == "__main__":
    main()
