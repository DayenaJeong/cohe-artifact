#!/usr/bin/env python3
"""Run CIFAR-100 ResNet-18 training with per-sample dynamics logging.

This script mirrors the lightweight CIFAR-100 training setup used in
`scripts_orth/run_ohs_cifar100.py`:

- torchvision ResNet-18
- 50 epochs by default
- SGD + Nesterov
- MultiStepLR milestones at 2/3 and 5/6 of training
- standard CIFAR-100 augmentation / normalization

Outputs are written under rebuttal_2026/exp_training_dynamics/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class IndexedCIFAR100(torchvision.datasets.CIFAR100):
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image, target, index


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def compute_first_learning_epoch(correctness: np.ndarray) -> np.ndarray:
    """Return first 1-based epoch from which correctness stays 1 forever.

    Samples that never become permanently correct receive -1.
    """

    corr_int = correctness.astype(np.int16)
    suffix_all_correct = np.flip(np.cumprod(np.flip(corr_int, axis=1), axis=1), axis=1).astype(bool)
    stable_correct = correctness & suffix_all_correct
    first_learning = np.full(correctness.shape[0], -1, dtype=np.int16)
    has_stable = stable_correct.any(axis=1)
    first_learning[has_stable] = stable_correct[has_stable].argmax(axis=1).astype(np.int16) + 1
    return first_learning


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument(
        "--out_dir",
        default="user_inputs/ICML/rebuttal_2026/exp_training_dynamics",
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    tf_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    tf_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = IndexedCIFAR100(root=args.data_root, train=True, download=True, transform=tf_train)
    test_set = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=tf_test)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    n_train = len(train_set)
    correctness = np.zeros((n_train, args.epochs), dtype=np.uint8)
    labels = np.asarray(train_set.targets, dtype=np.int16)

    model = torchvision.models.resnet18(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.wd,
        nesterov=True,
    )
    milestones = [args.epochs * 2 // 3, args.epochs * 5 // 6]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    epoch_rows = []
    best_test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y, idx in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            batch_correct = pred.eq(y)
            idx_np = idx.numpy()
            correctness[idx_np, epoch - 1] = batch_correct.detach().cpu().numpy().astype(np.uint8)

            running_loss += loss.item() * x.size(0)
            running_correct += batch_correct.sum().item()
            running_total += x.size(0)

        scheduler.step()
        train_acc = 100.0 * running_correct / max(running_total, 1)
        train_loss = running_loss / max(running_total, 1)
        test_acc = evaluate(model, test_loader, device)
        best_test_acc = max(best_test_acc, test_acc)
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "best_test_acc": float(best_test_acc),
            "lr": float(lr_now),
        }
        epoch_rows.append(row)
        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"lr={lr_now:.4g} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f} test_acc={test_acc:.2f} best={best_test_acc:.2f}"
        )

    forgetting = ((correctness[:, :-1] == 1) & (correctness[:, 1:] == 0)).sum(axis=1).astype(np.int16)
    first_learning = compute_first_learning_epoch(correctness)
    total_correct_epochs = correctness.sum(axis=1).astype(np.int16)
    ever_correct = (total_correct_epochs > 0).astype(np.uint8)
    last_epoch_correct = correctness[:, -1].astype(np.uint8)

    np.save(out_dir / "forgetting_counts.npy", forgetting)
    np.save(out_dir / "first_learning_epoch.npy", first_learning)
    np.save(out_dir / "correctness_by_epoch.npy", correctness)

    with (out_dir / "epoch_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    with (out_dir / "per_sample_dynamics.csv").open("w", newline="") as f:
        fieldnames = [
            "index",
            "label",
            "forgetting_count",
            "first_learning_epoch",
            "ever_correct",
            "last_epoch_correct",
            "total_correct_epochs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(n_train):
            writer.writerow(
                {
                    "index": idx,
                    "label": int(labels[idx]),
                    "forgetting_count": int(forgetting[idx]),
                    "first_learning_epoch": int(first_learning[idx]),
                    "ever_correct": int(ever_correct[idx]),
                    "last_epoch_correct": int(last_epoch_correct[idx]),
                    "total_correct_epochs": int(total_correct_epochs[idx]),
                }
            )

    metadata = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wd": args.wd,
        "device": device,
        "milestones": milestones,
        "best_test_acc": best_test_acc,
        "outputs": {
            "per_sample_dynamics_csv": str((out_dir / "per_sample_dynamics.csv").resolve()),
            "forgetting_counts_npy": str((out_dir / "forgetting_counts.npy").resolve()),
            "first_learning_epoch_npy": str((out_dir / "first_learning_epoch.npy").resolve()),
            "correctness_by_epoch_npy": str((out_dir / "correctness_by_epoch.npy").resolve()),
            "epoch_metrics_csv": str((out_dir / "epoch_metrics.csv").resolve()),
        },
        "note": "Correctness is recorded on the training presentation seen in each epoch (with the standard CIFAR-100 augmentation from the existing run_ohs pipeline).",
    }
    (out_dir / "training_dynamics_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
