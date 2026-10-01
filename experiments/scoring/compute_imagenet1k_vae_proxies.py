#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_id", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    use_amp = device.startswith("cuda")

    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    ds = torchvision.datasets.ImageNet(root=args.root, split=args.split, transform=tfm)
    shard = IndexedShardDataset(ds, shard_id=args.shard_id, num_shards=args.num_shards)
    loader = DataLoader(
        shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        persistent_workers=args.num_workers > 0,
    )

    dtype = torch.float16 if use_amp else torch.float32
    print(f">>> Loading AutoencoderKL from {args.model_id} on {device} ({dtype})")
    vae = AutoencoderKL.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    vae.eval()

    all_indices: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_pixel: list[np.ndarray] = []
    all_latent: list[np.ndarray] = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else torch.autocast(device_type="cpu", dtype=torch.float32)
    )

    with torch.inference_mode():
        for x, y, idx in tqdm(loader, desc=f"vae-proxy shard {args.shard_id}/{args.num_shards}", ncols=100):
            x = x.to(device, non_blocking=True)
            with autocast_ctx:
                z = vae.encode(x).latent_dist.mode()
                recon = vae.decode(z).sample
                z_recon = vae.encode(recon).latent_dist.mode()

            pixel_err = (x.float() - recon.float()).pow(2).mean(dim=(1, 2, 3))
            latent_err = (z.float() - z_recon.float()).pow(2).mean(dim=(1, 2, 3))

            all_indices.append(idx.numpy())
            all_targets.append(y.numpy())
            all_pixel.append(pixel_err.cpu().numpy())
            all_latent.append(latent_err.cpu().numpy())

    np.savez_compressed(
        out_path,
        indices=np.concatenate(all_indices).astype(np.int64),
        targets=np.concatenate(all_targets).astype(np.int64),
        vae_recon_mse=np.concatenate(all_pixel).astype(np.float32),
        sd_vae_latent_roundtrip_mse=np.concatenate(all_latent).astype(np.float32),
        shard_id=np.int64(args.shard_id),
        num_shards=np.int64(args.num_shards),
    )

    metadata = {
        "root": str(Path(args.root).resolve()),
        "split": args.split,
        "output_npz": str(out_path.resolve()),
        "model_id": args.model_id,
        "device": device,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "n_samples": int(sum(len(x) for x in all_indices)),
        "proxy_definitions": {
            "vae_recon_mse": "mean((x - decode(encode(x)))^2) in normalized pixel space",
            "sd_vae_latent_roundtrip_mse": "mean((z(x) - z(decode(z(x))))^2) where z is the AutoencoderKL posterior mode",
        },
    }
    out_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
