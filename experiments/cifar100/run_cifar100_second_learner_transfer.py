#!/usr/bin/env python3
"""Run a CIFAR-100 second-learner operational-transfer audit.

This script reuses the same ranking policies as the main CIFAR-100
operational-transfer audit, but trains a configurable learner. It writes each
run to a new result directory and skips completed JSON files so interrupted
launches can resume without overwriting old outputs.
"""

from __future__ import annotations

import argparse
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


def import_timm():
    try:
        import timm  # type: ignore
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("timm is required for the second-learner audit") from exc
    return timm


METHOD_ORDER = ["Random", "Gen-Hard", "Disc-Hard", "OHS-XOR(p75)"]


@dataclass
class TrainConfig:
    model_name: str
    batch_size: int = 128
    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 4
    data_root: str = "./data"
    num_classes: int = 100
    save_checkpoints: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (x - x.mean()) / (x.std() + eps)


def topk_indices(x: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(x)[-k:]


def ohs_xor_indices(
    d_gen: np.ndarray,
    d_disc: np.ndarray,
    k: int,
    percentile: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    tau_gen = np.percentile(d_gen, percentile)
    tau_disc = np.percentile(d_disc, percentile)
    pool = np.where(np.logical_xor(d_gen > tau_gen, d_disc > tau_disc))[0]
    if len(pool) < k:
        rest = np.setdiff1d(np.arange(len(d_gen)), pool)
        extra = rng.choice(rest, size=k - len(pool), replace=False)
        return np.concatenate([pool, extra])
    return rng.choice(pool, size=k, replace=False)


def build_selection_indices(
    d_gen: np.ndarray,
    d_disc: np.ndarray,
    k: int,
    seed: int,
    percentile: float,
) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {
        "Random": rng.choice(np.arange(len(d_gen)), size=k, replace=False),
        "Gen-Hard": topk_indices(d_gen, k),
        "Disc-Hard": topk_indices(d_disc, k),
        f"OHS-XOR(p{int(percentile)})": ohs_xor_indices(d_gen, d_disc, k, percentile, rng),
    }


def build_loaders(selected_indices: np.ndarray, cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
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
        download=True,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR100(
        root=cfg.data_root,
        train=False,
        download=True,
        transform=test_transform,
    )
    train_loader = DataLoader(
        Subset(train_set, selected_indices.tolist()),
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


def train_and_eval(
    run_name: str,
    selected_indices: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
) -> float:
    timm = import_timm()
    train_loader, test_loader = build_loaders(selected_indices, cfg)
    model = timm.create_model(cfg.model_name, pretrained=False, num_classes=cfg.num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

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
        if (epoch + 1) % max(1, cfg.epochs // 10) == 0:
            print(f"{run_name}: epoch {epoch + 1:03d}/{cfg.epochs} loss={running / max(1, len(train_loader)):.4f}", flush=True)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            pred = model(inputs).argmax(dim=1)
            total += int(targets.numel())
            correct += int((pred == targets).sum().item())
    acc = 100.0 * correct / total

    if cfg.save_checkpoints:
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"{run_name}.pth")
    return acc


def safe_method_name(method: str) -> str:
    return method.replace(" ", "").replace("/", "_").replace("(", "").replace(")", "").replace("=", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=Path, required=True)
    parser.add_argument("--disc", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--budgets", type=str, default="0.1,0.5,0.7")
    parser.add_argument("--methods", type=str, default=",".join(METHOD_ORDER))
    parser.add_argument("--ohs-percentile", type=float, default=75.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TrainConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        data_root=str(args.data_root),
        save_checkpoints=args.save_checkpoints,
    )
    d_gen = np.load(args.gen).astype(np.float64)
    d_disc = np.load(args.disc).astype(np.float64)
    if len(d_gen) != len(d_disc):
        raise ValueError(f"Length mismatch: {len(d_gen)} vs {len(d_disc)}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    budgets = [float(b.strip()) for b in args.budgets.split(",") if b.strip()]
    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(requested_methods) - set(METHOD_ORDER))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {METHOD_ORDER}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    manifest = {
        "model_name": args.model_name,
        "gen": str(args.gen),
        "disc": str(args.disc),
        "seeds": seeds,
        "budgets": budgets,
        "methods": requested_methods,
        "ohs_percentile": args.ohs_percentile,
        "train_cfg": asdict(cfg),
        "results": [],
    }

    for budget in budgets:
        k = max(1, min(len(d_gen), int(round(budget * len(d_gen)))))
        for seed in seeds:
            set_seed(seed)
            selections = build_selection_indices(d_gen, d_disc, k, seed, args.ohs_percentile)
            for method in requested_methods:
                idx = selections[method]
                run_name = f"cifar100_{args.model_name}_budget{budget:.2f}_K{k}_seed{seed}_{safe_method_name(method)}"
                json_path = raw_dir / f"{run_name}.json"
                index_path = raw_dir / f"{run_name}.indices.npy"
                if json_path.exists():
                    print(f"[skip] {json_path}", flush=True)
                    payload = json.loads(json_path.read_text())
                    manifest["results"].append(payload)
                    continue
                np.save(index_path, idx)
                print(f"[run] {run_name} n={len(idx)} device={device}", flush=True)
                acc = train_and_eval(run_name, idx, cfg, device, args.out_dir)
                payload = {
                    "dataset": "CIFAR-100",
                    "model_name": args.model_name,
                    "budget": budget,
                    "K": k,
                    "seed": seed,
                    "method": method,
                    "acc": float(acc),
                    "index_file": str(index_path.name),
                }
                json_path.write_text(json.dumps(payload, indent=2))
                manifest["results"].append(payload)
                (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
                print(f"[done] {run_name} acc={acc:.2f}", flush=True)

    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
