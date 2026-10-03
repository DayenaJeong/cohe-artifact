#!/usr/bin/env python
"""Score ImageNet-10 train examples with a fixed fine-tuned discriminative target.

This script prepares the train-split D_disc arrays needed for the minimal
ImageNet-family operational-transfer sanity check. It does not run subset
selection; it only fine-tunes a ResNet-18 head and scores the train split in
ImageFolder order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


def build_model(num_classes: int, device: torch.device, train_backbone: bool) -> torch.nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    if not train_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("ICML/data/imagenet10"))
    parser.add_argument("--out-dir", type=Path, default=Path("NeurIPS/results/imagenette_transfer_minimal/scores"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-backbone", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = ImageFolder(args.data_root / "train", transform=train_tfm)
    train_eval_ds = ImageFolder(args.data_root / "train", transform=eval_tfm)
    val_ds = ImageFolder(args.data_root / "val", transform=eval_tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(len(train_ds.classes), device, args.train_backbone)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch {epoch + 1:03d}/{args.epochs:03d} train_loss={total_loss / max(len(train_loader), 1):.4f}")

    model.eval()
    ce_all: list[np.ndarray] = []
    margin_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in train_eval_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            ce = F.cross_entropy(logits, y, reduction="none")
            true_logits = logits.gather(1, y[:, None]).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, y[:, None], float("-inf"))
            margin = true_logits - masked.max(dim=1).values
            margin_hardness = -margin
            ce_all.append(ce.cpu().numpy())
            margin_all.append(margin_hardness.cpu().numpy())
            labels_all.append(y.cpu().numpy())

    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            val_correct += int((pred == y).sum().item())
            val_total += int(y.numel())

    np.save(args.out_dir / "ce_train.npy", np.concatenate(ce_all))
    np.save(args.out_dir / "margin_hardness_train.npy", np.concatenate(margin_all))
    np.save(args.out_dir / "train_labels.npy", np.concatenate(labels_all))
    np.save(args.out_dir / "val_labels.npy", np.asarray(val_ds.targets, dtype=np.int64))
    manifest = {
        "dataset": "ImageNet-10 / Imagenette-style folder",
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "classes": train_ds.classes,
        "score_direction": {
            "ce_train.npy": "higher cross-entropy means harder",
            "margin_hardness_train.npy": "higher negative margin means harder",
        },
        "model": "ImageNet-pretrained ResNet-18 with reset 10-way head",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_backbone": args.train_backbone,
        "val_accuracy": val_correct / max(val_total, 1),
    }
    (args.out_dir / "target_score_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
