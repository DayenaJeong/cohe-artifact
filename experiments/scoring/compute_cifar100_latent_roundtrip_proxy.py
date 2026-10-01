#!/usr/bin/env python3
"""Compute a CIFAR-100 latent-space generative proxy with a pretrained SD VAE.

Proxy definition:
    x -> encode -> z -> decode -> recon -> encode -> z_recon
    hardness(x) = mean((z - z_recon)^2)

This is intentionally lightweight and additive for rebuttal use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_id", default="stabilityai/sd-vae-ft-mse")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    ds = torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=False, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f">>> Loading AutoencoderKL from {args.model_id}")
    vae = AutoencoderKL.from_pretrained(args.model_id).to(device)
    vae.eval()

    scores: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="latent round-trip"):
            x = x.to(device, non_blocking=True)
            z = vae.encode(x).latent_dist.mode()
            recon = vae.decode(z).sample
            z_recon = vae.encode(recon).latent_dist.mode()
            latent_err = (z - z_recon).pow(2).mean(dim=(1, 2, 3))
            scores.append(latent_err.detach().cpu().numpy())

    arr = np.concatenate(scores).astype(np.float32)
    np.save(out_path, arr)

    metadata = {
        "proxy_name": "sd_vae_latent_roundtrip_mse",
        "definition": "mean((z(x) - z(decode(z(x))))^2) where z is the AutoencoderKL posterior mode",
        "model_id": args.model_id,
        "data_root": str(Path(args.data_root).resolve()),
        "n_samples": int(arr.shape[0]),
        "device": device,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "output_npy": str(out_path.resolve()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }
    metadata_path = out_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
