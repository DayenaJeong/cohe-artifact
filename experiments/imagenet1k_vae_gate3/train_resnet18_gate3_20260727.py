#!/usr/bin/env python3
"""Locked 90-epoch single-GPU ResNet-18 Gate 3 run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from train_resnet18_timing_pilot_20260727 import CsvImageDataset, accuracy, worker_seed, VAL_ROOT


DATA_ROOT = Path(os.environ.get("COHE_IMAGENET_TRAIN_ROOT", "user_inputs/imagenet/train"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_top1: float, config: dict) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_validation_top1": best_top1,
        "config": config,
    }, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--policy", choices=["random", "proxy_hard"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=90)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "status": "RUNNING",
        "policy": args.policy,
        "seed": args.seed,
        "selection": str(args.selection.resolve()),
        "selection_sha256": sha256_file(args.selection),
        "dataset": "ImageNet-1K",
        "train_root": str(DATA_ROOT.resolve()),
        "validation_root": str(VAL_ROOT.resolve()),
        "budget": 0.30,
        "rounding_rule": "floor(class_count * budget)",
        "learner": "torchvision.models.resnet18(weights=None, num_classes=1000)",
        "pretrained": False,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "optimizer": "SGD",
        "learning_rate": 0.1 * args.batch_size / 256.0,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "scheduler": "StepLR(step_size=30, gamma=0.1)",
        "amp": True,
        "gradient_clipping": False,
        "label_smoothing": False,
        "mixup": False,
        "cutmix": False,
        "randaugment": False,
        "ema": False,
        "early_stopping": False,
        "train_transform": ["RandomResizedCrop(224)", "RandomHorizontalFlip", "ToTensor", "Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])"],
        "validation_transform": ["Resize(256)", "CenterCrop(224)", "ToTensor", "Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])"],
        "selection_only_difference": True,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "subset_checksum.txt").write_text(f"{config['selection_sha256']}  {args.selection.name}\n", encoding="utf-8")
    status_path = args.out_dir / "run_status.json"
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_tfm = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_ds = CsvImageDataset(args.selection, train_tfm)
        val_ds = ImageFolder(str(VAL_ROOT), transform=val_tfm)
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0, worker_init_fn=worker_seed)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
        model = models.resnet18(weights=None, num_classes=1000).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        if len(val_ds.classes) != 1000 or len(val_ds) != 50000:
            raise RuntimeError("validation mapping/count mismatch")

        metrics_path = args.out_dir / "metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as mf:
            fields = ["epoch", "train_loss", "validation_top1", "validation_top5", "learning_rate", "train_seconds", "validation_seconds", "optimizer_steps"]
            writer = csv.DictWriter(mf, fieldnames=fields)
            writer.writeheader()
            best_top1 = float("-inf")
            total_steps = 0
            for epoch in range(1, args.epochs + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                train_start = time.perf_counter()
                loss_sum = 0.0
                steps = 0
                for images, target in train_loader:
                    images = images.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(images)
                        loss = torch.nn.functional.cross_entropy(output, target)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss epoch={epoch}")
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum += float(loss.detach().item())
                    steps += 1
                    total_steps += 1
                torch.cuda.synchronize(device)
                train_seconds = time.perf_counter() - train_start
                scheduler.step()

                model.eval()
                val_start = time.perf_counter()
                top1 = top5 = count = 0.0
                with torch.inference_mode():
                    for images, target in val_loader:
                        images = images.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            output = model(images)
                        a1, a5 = accuracy(output, target)
                        top1 += a1
                        top5 += a5
                        count += target.numel()
                torch.cuda.synchronize(device)
                val_seconds = time.perf_counter() - val_start
                top1_pct = 100.0 * top1 / count
                top5_pct = 100.0 * top5 / count
                writer.writerow({"epoch": epoch, "train_loss": loss_sum / max(1, steps), "validation_top1": top1_pct, "validation_top5": top5_pct, "learning_rate": optimizer.param_groups[0]["lr"], "train_seconds": train_seconds, "validation_seconds": val_seconds, "optimizer_steps": steps})
                mf.flush()
                print(f"epoch={epoch} train_loss={loss_sum / max(1, steps):.6f} top1={top1_pct:.4f} top5={top5_pct:.4f} train_s={train_seconds:.3f} val_s={val_seconds:.3f} steps={steps}", flush=True)
                if top1_pct > best_top1:
                    best_top1 = top1_pct
                    save_checkpoint(args.out_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_top1, config)

            save_checkpoint(args.out_dir / "final.pt", model, optimizer, scheduler, scaler, args.epochs, best_top1, config)
        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "git_commit": "ec67adf12894e62ff06a79d7e6c8782f21e69bee",
        }
        (args.out_dir / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        config["status"] = "COMPLETE"
        config["best_validation_top1"] = best_top1
        config["optimizer_steps_total"] = total_steps
        (args.out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        status_path.write_text(json.dumps({"status": "COMPLETE", "policy": args.policy, "seed": args.seed, "best_validation_top1": best_top1, "optimizer_steps_total": total_steps, "timestamp": time.time()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        status_path.write_text(json.dumps({"status": "FAILED", "policy": args.policy, "seed": args.seed, "error": repr(exc), "timestamp": time.time()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
