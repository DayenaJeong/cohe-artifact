#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


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
    if name == "cifar100":
        train = split == "train"
        return torchvision.datasets.CIFAR100(root=root, train=train, download=False)
    if name == "imagenet1k":
        return torchvision.datasets.ImageNet(root=root, split=split)
    raise ValueError(f"Unsupported dataset: {name}")


def collate(batch):
    images, targets, indices = zip(*batch)
    return list(images), np.asarray(targets, dtype=np.int64), np.asarray(indices, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar100", "imagenet1k"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-id", default="BAAI/Emu3-VisionTokenizer")
    ap.add_argument("--min-pixels", type=int, default=256 * 256)
    ap.add_argument("--max-pixels", type=int, default=256 * 256)
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

    processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).eval().to(device)

    all_indices: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else torch.autocast(device_type="cpu", dtype=torch.float32)
    )

    with torch.inference_mode():
        for images, targets, indices in tqdm(
            loader,
            desc=f"emu3-vq-recon {args.dataset} {args.split} shard {args.shard_id}/{args.num_shards}",
            ncols=110,
        ):
            if args.min_pixels == args.max_pixels:
                side = int(round(math.sqrt(args.max_pixels)))
                if side * side == args.max_pixels:
                    images = [TF.resize(img, [side, side]) for img in images]
            x = processor(images, return_tensors="pt")["pixel_values"].to(device, non_blocking=True)
            with autocast_ctx:
                codes = model.encode(x)
                recon = model.decode(codes)
            score = (recon.float() - x.float()).pow(2).mean(dim=(1, 2, 3))

            all_indices.append(indices)
            all_targets.append(targets)
            all_scores.append(score.cpu().numpy())

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
        "model_id": args.model_id,
        "device": device,
        "dtype": str(dtype),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "min_pixels": int(args.min_pixels),
        "max_pixels": int(args.max_pixels),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "max_samples": int(args.max_samples),
        "seed": int(args.seed),
        "n_samples": int(sum(len(x) for x in all_indices)),
        "score_definition": "Per-sample pixel MSE between Emu3-VisionTokenizer input pixels and decode(encode(x)) reconstruction, using the pretrained BAAI Emu3 VQ-family vision tokenizer with fixed processor min/max pixels.",
    }
    out_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
