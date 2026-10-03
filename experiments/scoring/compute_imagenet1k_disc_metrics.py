#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class IndexedShardDataset(Dataset):
    def __init__(self, base: Dataset, shard_id: int, num_shards: int) -> None:
        self.base = base
        self.indices = np.arange(len(base), dtype=np.int64)[shard_id::num_shards]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x, y = self.base[idx]
        return x, y, idx


def build_model(name: str):
    if name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
    elif name == "wide_resnet50_2":
        weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2
        model = torchvision.models.wide_resnet50_2(weights=weights)
    elif name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    elif name == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        model = torchvision.models.vit_b_16(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model, weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    torch.backends.cudnn.benchmark = True

    model, weights = build_model(args.model)
    model = model.to(device)
    model.eval()
    tfm = weights.transforms()

    ds = torchvision.datasets.ImageNet(root=args.root, split=args.split, transform=tfm)
    shard = IndexedShardDataset(ds, shard_id=args.shard_id, num_shards=args.num_shards)
    loader = DataLoader(
        shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )

    all_indices: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_ce: list[np.ndarray] = []
    all_margin_h: list[np.ndarray] = []
    all_raw_margin: list[np.ndarray] = []
    all_gn: list[np.ndarray] = []

    for x, y, idx in tqdm(loader, desc=f"disc-metrics {args.model} shard {args.shard_id}/{args.num_shards}", ncols=100):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x.requires_grad_(True)

        logits = model(x)
        ce = F.cross_entropy(logits, y, reduction="none")

        gt_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
        masked = logits.clone()
        masked[torch.arange(logits.size(0), device=logits.device), y] = float("-inf")
        max_other = masked.max(dim=1).values
        raw_margin = gt_logits - max_other
        margin_hardness = max_other - gt_logits

        model.zero_grad(set_to_none=True)
        loss = ce.mean()
        loss.backward()
        gradnorm = x.grad.detach().view(x.size(0), -1).norm(p=2, dim=1)

        all_indices.append(idx.numpy())
        all_targets.append(y.cpu().numpy())
        all_ce.append(ce.detach().cpu().numpy())
        all_margin_h.append(margin_hardness.detach().cpu().numpy())
        all_raw_margin.append(raw_margin.detach().cpu().numpy())
        all_gn.append(gradnorm.cpu().numpy())

    np.savez_compressed(
        out_path,
        indices=np.concatenate(all_indices).astype(np.int64),
        targets=np.concatenate(all_targets).astype(np.int64),
        ce_loss=np.concatenate(all_ce).astype(np.float32),
        margin_hardness=np.concatenate(all_margin_h).astype(np.float32),
        raw_margin=np.concatenate(all_raw_margin).astype(np.float32),
        gradnorm=np.concatenate(all_gn).astype(np.float32),
        shard_id=np.int64(args.shard_id),
        num_shards=np.int64(args.num_shards),
    )

    metadata = {
        "root": str(Path(args.root).resolve()),
        "split": args.split,
        "output_npz": str(out_path.resolve()),
        "device": device,
        "model": args.model,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "n_samples": int(sum(len(x) for x in all_indices)),
        "metric_definitions": {
            "ce_loss": "per-sample cross-entropy loss",
            "margin_hardness": "max_non_gt_logit - gt_logit (higher means harder)",
            "raw_margin": "gt_logit - max_non_gt_logit (lower means harder)",
            "gradnorm": "L2 norm of input gradient under mean CE loss, equivalent up to batch-scaling to per-sample ranking",
        },
    }
    out_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
