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
from transformers import ImageGPTConfig, ImageGPTForCausalImageModeling
from transformers.models.imagegpt.image_processing_imagegpt import ImageGPTImageProcessor


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


def build_dataset(name: str, root: str, split: str):
    if name == "imagenet1k":
        return torchvision.datasets.ImageNet(root=root, split=split)
    if name == "cifar100":
        return torchvision.datasets.CIFAR100(root=root, train=(split == "train"), download=False)
    raise ValueError(f"Unsupported dataset: {name}")


def collate(batch):
    images, targets, indices = zip(*batch)
    return list(images), np.asarray(targets, dtype=np.int64), np.asarray(indices, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["imagenet1k", "cifar100"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    use_amp = device.startswith("cuda")
    dtype = torch.float16 if use_amp else torch.float32

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True

    ds = build_dataset(args.dataset, args.root, args.split)
    shard = IndexedShardDataset(ds, args.shard_id, args.num_shards)
    if args.max_samples > 0:
        shard.indices = shard.indices[: args.max_samples]

    loader = DataLoader(
        shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
    )

    config = ImageGPTConfig.from_pretrained(args.config_dir, local_files_only=True)
    model = ImageGPTForCausalImageModeling.from_pretrained(
        args.weights_dir,
        config=config,
        local_files_only=True,
    ).eval().to(device)
    if use_amp:
        model = model.half()
    processor = ImageGPTImageProcessor.from_pretrained(args.config_dir, local_files_only=True)

    all_indices: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else torch.autocast(device_type="cpu", dtype=torch.float32)
    )

    with torch.inference_mode():
        for images, targets, indices in loader:
            enc = processor(images, return_tensors="pt")
            input_ids = enc["input_ids"].to(device, non_blocking=True)
            with autocast_ctx:
                logits = model(input_ids=input_ids).logits
            token_loss = F.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                reduction="none",
            ).view(input_ids.size(0), -1)
            score = token_loss.mean(dim=1)

            all_indices.append(indices)
            all_targets.append(targets)
            all_scores.append(score.cpu().numpy().astype(np.float32))

    np.savez_compressed(
        out_path,
        indices=np.concatenate(all_indices).astype(np.int64),
        targets=np.concatenate(all_targets).astype(np.int64),
        score=np.concatenate(all_scores).astype(np.float32),
        shard_id=np.int64(args.shard_id),
        num_shards=np.int64(args.num_shards),
    )

    metadata = {
        "dataset": args.dataset,
        "root": str(Path(args.root).resolve()),
        "split": args.split,
        "output_npz": str(out_path.resolve()),
        "config_dir": str(Path(args.config_dir).resolve()),
        "weights_dir": str(Path(args.weights_dir).resolve()),
        "device": device,
        "dtype": str(dtype),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "max_samples": int(args.max_samples),
        "seed": int(args.seed),
        "n_samples": int(sum(len(x) for x in all_indices)),
        "score_definition": "Per-sample mean autoregressive next-token negative log-likelihood over ImageGPT color-quantized 32x32 image tokens.",
    }
    out_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
