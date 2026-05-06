#!/usr/bin/env python3
"""Run DINOv2 representation-isolation Gate 3 transfer on CIFAR-100.

This runner intentionally reuses the existing CIFAR-100 subset-training recipe:
ResNet-18 trained from scratch on selected subsets, then evaluated on CIFAR-100
test accuracy. It adds DINOv2 isolation hard/easy selection policies and writes
one JSON plus one index file per run so aggregation can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


METHOD_ORDER = [
    "Random",
    "DINOv2-Isolation-Hard",
    "DINOv2-Isolation-Easy",
    "FirstLearning-Hard",
]


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 4
    data_root: str = "./data"
    model_name: str = "resnet18"
    num_classes: int = 100
    save_checkpoints: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def safe_name(value: str) -> str:
    return (
        value.replace(" ", "")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace("+", "plus")
    )


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(scores)[-k:]


def bottomk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(scores)[:k]


def load_proxy_csv(path: Path, score_col: str) -> np.ndarray:
    rows: list[tuple[int, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a sample_id column")
        if score_col not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain score column {score_col!r}")
        for row in reader:
            rows.append((int(row["sample_id"]), float(row[score_col])))
    rows.sort(key=lambda item: item[0])
    sample_ids = np.array([idx for idx, _ in rows], dtype=np.int64)
    if not np.array_equal(sample_ids, np.arange(len(sample_ids))):
        raise ValueError("DINOv2 proxy sample_id values must be contiguous CIFAR-100 train indices")
    return np.array([score for _, score in rows], dtype=np.float64)


def load_first_learning(path: Path | None, expected_n: int) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.load(path).astype(np.float64)
    if len(arr) != expected_n:
        raise ValueError(f"first-learning length mismatch: {len(arr)} vs {expected_n}")
    return arr


def cifar100_train_labels(data_root: Path, download: bool) -> np.ndarray:
    train_set = torchvision.datasets.CIFAR100(
        root=str(data_root),
        train=True,
        download=download,
        transform=None,
    )
    return np.array(train_set.targets, dtype=np.int64)


def normalized_entropy(labels: np.ndarray, num_classes: int = 100) -> float:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    probs = counts[counts > 0] / max(1.0, counts.sum())
    if len(probs) <= 1:
        return 0.0
    return float(-(probs * np.log(probs)).sum() / np.log(num_classes))


def subset_diagnostics(
    indices: np.ndarray,
    labels: np.ndarray,
    isolation: np.ndarray,
    first_learning: np.ndarray | None,
) -> dict[str, float | int | None]:
    subset_labels = labels[indices]
    out: dict[str, float | int | None] = {
        "classes_present": int(len(np.unique(subset_labels))),
        "class_entropy_norm": normalized_entropy(subset_labels),
        "mean_dinov2_isolation": float(np.mean(isolation[indices])),
        "full_mean_dinov2_isolation": float(np.mean(isolation)),
    }
    out["delta_vs_full_dinov2_isolation"] = float(out["mean_dinov2_isolation"] - out["full_mean_dinov2_isolation"])
    if first_learning is None:
        out.update(
            {
                "mean_first_learning": None,
                "full_mean_first_learning": None,
                "delta_vs_full_first_learning": None,
                "frac_top10_first_learning": None,
            }
        )
    else:
        top10_threshold = float(np.percentile(first_learning, 90))
        out.update(
            {
                "mean_first_learning": float(np.mean(first_learning[indices])),
                "full_mean_first_learning": float(np.mean(first_learning)),
                "delta_vs_full_first_learning": float(np.mean(first_learning[indices]) - np.mean(first_learning)),
                "frac_top10_first_learning": float(np.mean(first_learning[indices] >= top10_threshold)),
            }
        )
    return out


def build_selection_indices(
    isolation: np.ndarray,
    first_learning: np.ndarray | None,
    k: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    selections = {
        "Random": rng.choice(np.arange(len(isolation)), size=k, replace=False),
        "DINOv2-Isolation-Hard": topk_indices(isolation, k),
        "DINOv2-Isolation-Easy": bottomk_indices(isolation, k),
    }
    if first_learning is not None:
        selections["FirstLearning-Hard"] = topk_indices(first_learning, k)
    return selections


def build_loaders(indices: np.ndarray, cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    train_set = torchvision.datasets.CIFAR100(
        root=cfg.data_root,
        train=True,
        download=False,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR100(
        root=cfg.data_root,
        train=False,
        download=False,
        transform=test_transform,
    )
    train_loader = DataLoader(
        Subset(train_set, indices.tolist()),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=100,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def create_model(model_name: str, num_classes: int) -> nn.Module:
    try:
        import timm  # type: ignore

        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    except Exception:
        if model_name != "resnet18":
            raise
        model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            pred = model(inputs).argmax(dim=1)
            total += int(targets.numel())
            correct += int((pred == targets).sum().item())
    return 100.0 * correct / total


def train_and_eval(
    run_name: str,
    indices: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
    eval_epochs: set[int],
) -> dict[str, float]:
    train_loader, test_loader = build_loaders(indices, cfg)
    model = create_model(cfg.model_name, cfg.num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()
    evals: dict[str, float] = {}

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        scheduler.step()
        epoch_num = epoch + 1
        if epoch_num in eval_epochs:
            evals[f"acc_epoch_{epoch_num}"] = evaluate(model, test_loader, device)
        if epoch_num % max(1, cfg.epochs // 10) == 0:
            print(f"{run_name}: epoch {epoch_num:03d}/{cfg.epochs} loss={running / max(1, len(train_loader)):.4f}", flush=True)

    evals["acc_final"] = evaluate(model, test_loader, device)
    if cfg.save_checkpoints:
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"{run_name}.pth")
    return evals


def parse_csv_list(value: str, cast):
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-csv", type=Path, required=True)
    parser.add_argument("--score-col", type=str, default="dinov2_isolation_k20")
    parser.add_argument("--first-learning", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--budgets", type=str, default="0.1")
    parser.add_argument("--methods", type=str, default="DINOv2-Isolation-Hard,DINOv2-Isolation-Easy")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-epochs", type=str, default="20")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    isolation = load_proxy_csv(args.proxy_csv, args.score_col)
    first_learning = load_first_learning(args.first_learning, len(isolation))
    labels = cifar100_train_labels(args.data_root, args.download)
    if len(labels) != len(isolation):
        raise ValueError(f"label length mismatch: {len(labels)} vs {len(isolation)}")

    seeds = parse_csv_list(args.seeds, int)
    budgets = parse_csv_list(args.budgets, float)
    methods = parse_csv_list(args.methods, str)
    unknown = sorted(set(methods) - set(METHOD_ORDER))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; allowed methods: {METHOD_ORDER}")
    eval_epochs = {int(v) for v in parse_csv_list(args.eval_epochs, int) if int(v) > 0}

    cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        data_root=str(args.data_root),
        model_name=args.model_name,
        save_checkpoints=args.save_checkpoints,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    manifest = {
        "dataset": "CIFAR-100",
        "proxy": "DINOv2 feature-space isolation",
        "proxy_csv": str(args.proxy_csv),
        "score_col": args.score_col,
        "first_learning": str(args.first_learning) if args.first_learning else None,
        "seeds": seeds,
        "budgets": budgets,
        "methods": methods,
        "eval_epochs": sorted(eval_epochs),
        "train_cfg": asdict(cfg),
        "results": [],
    }

    for budget in budgets:
        k = max(1, min(len(isolation), int(round(budget * len(isolation)))))
        for seed in seeds:
            set_seed(seed)
            selections = build_selection_indices(isolation, first_learning, k, seed)
            for method in methods:
                indices = selections[method]
                run_name = f"cifar100_{cfg.model_name}_budget{budget:.2f}_K{k}_seed{seed}_{safe_name(method)}"
                json_path = raw_dir / f"{run_name}.json"
                index_path = raw_dir / f"{run_name}.indices.npy"
                if json_path.exists():
                    payload = json.loads(json_path.read_text())
                    manifest["results"].append(payload)
                    print(f"[skip] {json_path}", flush=True)
                    continue
                np.save(index_path, indices)
                diagnostics = subset_diagnostics(indices, labels, isolation, first_learning)
                print(f"[run] {run_name} n={len(indices)} device={device}", flush=True)
                evals = train_and_eval(run_name, indices, cfg, device, args.out_dir, eval_epochs)
                payload = {
                    "dataset": "CIFAR-100",
                    "model_name": cfg.model_name,
                    "budget": budget,
                    "K": k,
                    "seed": seed,
                    "method": method,
                    "acc": float(evals["acc_final"]),
                    "eval_accuracies": evals,
                    "diagnostics": diagnostics,
                    "index_file": index_path.name,
                }
                json_path.write_text(json.dumps(payload, indent=2))
                manifest["results"].append(payload)
                (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
                print(f"[done] {run_name} acc={evals['acc_final']:.2f}", flush=True)

    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
