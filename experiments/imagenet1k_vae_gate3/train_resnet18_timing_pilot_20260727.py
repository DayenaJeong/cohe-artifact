#!/usr/bin/env python3
"""One-epoch timing pilot for the locked ImageNet-1K Gate 3 learner."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


DATA_ROOT = Path(os.environ.get("COHE_IMAGENET_TRAIN_ROOT", "user_inputs/imagenet/train"))
VAL_ROOT = Path(os.environ.get("COHE_IMAGENET_VAL_ROOT", "user_inputs/imagenet/val"))


class CsvImageDataset(Dataset):
    def __init__(self, csv_path: Path, transform):
        with csv_path.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(DATA_ROOT / row["relative_path"]) as im:
            x = self.transform(im.convert("RGB"))
        return x, int(row["class_index"])


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def gpu_utilization() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        values = [int(x.strip()) for x in out.splitlines() if x.strip()]
        return values[0] if values else None
    except Exception:
        return None


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum().item() for k in topk]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start_total = time.perf_counter()
    result = {
        "status": "RUNNING",
        "selection": str(args.selection.resolve()),
        "data_root": str(DATA_ROOT.resolve()),
        "validation_root": str(VAL_ROOT.resolve()),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "epochs_configured": 90,
        "learner": "torchvision.models.resnet18(weights=None, num_classes=1000)",
        "pretrained": False,
        "train_transform": ["RandomResizedCrop(224)", "RandomHorizontalFlip", "ToTensor", "Normalize(ImageNet)"],
        "validation_transform": ["Resize(256)", "CenterCrop(224)", "ToTensor", "Normalize(ImageNet)"],
        "optimizer": "SGD(momentum=0.9, weight_decay=1e-4)",
        "scheduler": "StepLR(step_size=30, gamma=0.1)",
        "amp": True,
        "gradient_clipping": False,
        "extra_augmentation": False,
    }
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")

        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_ds = CsvImageDataset(args.selection, train_tfm)
        val_ds = ImageFolder(str(VAL_ROOT), transform=val_tfm)
        g = torch.Generator()
        g.manual_seed(args.seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            worker_init_fn=worker_seed,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        model = models.resnet18(weights=None, num_classes=1000).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1 * args.batch_size / 256.0, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        result.update({
            "train_sample_count": len(train_ds),
            "validation_sample_count": len(val_ds),
            "class_count": len(val_ds.classes),
            "initial_lr": optimizer.param_groups[0]["lr"],
            "gpu": torch.cuda.get_device_name(device),
        })

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_start = time.perf_counter()
        loss_sum = 0.0
        steps = 0
        util_samples: list[int] = []
        for batch_idx, (images, target) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(images)
                loss = nn.functional.cross_entropy(output, target)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach().item())
            steps += 1
            if batch_idx % 100 == 0:
                util = gpu_utilization()
                if util is not None:
                    util_samples.append(util)
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_start
        scheduler.step()

        model.eval()
        val_start = time.perf_counter()
        top1 = 0.0
        top5 = 0.0
        val_count = 0
        with torch.inference_mode():
            for images, target in val_loader:
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(images)
                a1, a5 = accuracy(output, target)
                top1 += a1
                top5 += a5
                val_count += target.numel()
        torch.cuda.synchronize(device)
        val_seconds = time.perf_counter() - val_start
        result.update({
            "status": "PASS",
            "epoch_completed": 1,
            "optimizer_steps": steps,
            "train_loss_mean": loss_sum / max(1, steps),
            "training_seconds_per_epoch": train_seconds,
            "validation_seconds": val_seconds,
            "validation_top1_percent": 100.0 * top1 / val_count,
            "validation_top5_percent": 100.0 * top5 / val_count,
            "train_images_per_second": len(train_ds) / train_seconds,
            "validation_images_per_second": val_count / val_seconds,
            "projected_90_epoch_hours": 90.0 * (train_seconds + val_seconds) / 3600.0,
            "projected_6_run_wall_hours_single_gpu_parallel": 90.0 * (train_seconds + val_seconds) / 3600.0,
            "projected_10_run_wall_hours_single_gpu_parallel": 90.0 * (train_seconds + val_seconds) / 3600.0,
            "gpu_utilization_samples_percent": util_samples,
            "gpu_utilization_mean_percent": float(np.mean(util_samples)) if util_samples else None,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "elapsed_seconds": time.perf_counter() - start_total,
        })
    except RuntimeError as exc:
        result.update({
            "status": "OOM" if "out of memory" in str(exc).lower() else "FAIL",
            "error": repr(exc),
            "elapsed_seconds": time.perf_counter() - start_total,
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        raise
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
