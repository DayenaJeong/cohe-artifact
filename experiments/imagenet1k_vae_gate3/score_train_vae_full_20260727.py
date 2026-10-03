#!/usr/bin/env python3
"""Generate one deterministic, scalar-only VAE score shard."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


MODEL_ID = "stabilityai/sd-vae-ft-mse"
DATA_ROOT = Path(os.environ.get("COHE_IMAGENET_TRAIN_ROOT", "user_inputs/imagenet/train"))


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, transform):
        self.rows: list[tuple[int, str, int]] = []
        with manifest.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((int(row["global_index"]), row["relative_path"], int(row["class_index"])))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        global_index, relative_path, class_index = self.rows[i]
        with Image.open(DATA_ROOT / relative_path) as im:
            x = self.transform(im.convert("RGB"))
        return x, global_index, class_index


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.metadata.exists():
        try:
            prior = json.loads(args.metadata.read_text(encoding="utf-8"))
            if prior.get("status") == "PASS":
                print(json.dumps({"status": "SKIPPED_EXISTING", "output": str(args.out), "metadata": str(args.metadata)}))
                return
        except Exception:
            pass
    start_total = time.perf_counter()
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    ds = ManifestDataset(args.manifest, tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    device = torch.device("cuda")
    model_load_start = time.perf_counter()
    vae = AutoencoderKL.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True).to(device)
    vae.eval()
    model_load_seconds = time.perf_counter() - model_load_start
    torch.cuda.reset_peak_memory_stats(device)
    indices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    finite = True
    score_min = float("inf")
    score_max = float("-inf")
    loop_start = time.perf_counter()
    with torch.inference_mode():
        for batch_no, (x, idx, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = vae.encode(x).latent_dist.mode()
                recon = vae.decode(z).sample
            batch_scores = (x.float() - recon.float()).pow(2).mean(dim=(1, 2, 3))
            values = batch_scores.detach().cpu().numpy().astype(np.float32, copy=False)
            finite = finite and bool(np.isfinite(values).all())
            score_min = min(score_min, float(values.min()))
            score_max = max(score_max, float(values.max()))
            indices.append(idx.numpy().astype(np.int64, copy=False))
            targets.append(y.numpy().astype(np.int64, copy=False))
            scores.append(values)
            if (batch_no + 1) % 100 == 0:
                processed = sum(len(a) for a in scores)
                elapsed = time.perf_counter() - loop_start
                print(f"progress shard={args.shard_id} batches={batch_no + 1} images={processed} images_per_sec={processed / elapsed:.3f}", flush=True)
    torch.cuda.synchronize(device)
    loop_seconds = time.perf_counter() - loop_start
    all_indices = np.concatenate(indices).astype(np.int64)
    all_targets = np.concatenate(targets).astype(np.int64)
    all_scores = np.concatenate(scores).astype(np.float32)
    if all_indices.shape[0] != len(ds) or not finite or len(np.unique(all_indices)) != len(ds):
        raise RuntimeError("score shard alignment or finite check failed")
    order = np.argsort(all_indices, kind="stable")
    all_indices = all_indices[order]
    all_targets = all_targets[order]
    all_scores = all_scores[order]
    tmp_out = args.out.with_name(args.out.name + ".partial.npz")
    np.savez_compressed(tmp_out, global_index=all_indices, class_index=all_targets, proxy_score=all_scores)
    tmp_out.replace(args.out)
    metadata = {
        "status": "PASS",
        "output_npz": str(args.out.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "data_root": str(DATA_ROOT.resolve()),
        "model_id": MODEL_ID,
        "model_loading": "local_files_only=True",
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "n_samples": int(all_scores.shape[0]),
        "dtype": "float32",
        "finite_scores": bool(np.isfinite(all_scores).all()),
        "unique_global_indices": int(np.unique(all_indices).shape[0]),
        "global_index_min": int(all_indices.min()),
        "global_index_max": int(all_indices.max()),
        "proxy_name": "VAE reconstruction error",
        "proxy_definition": "mean((x - decode(encode(x)))^2) in normalized pixel space",
        "score_direction": "higher reconstruction MSE = harder",
        "preprocessing": ["Resize(256)", "CenterCrop(256)", "ToTensor", "Normalize([0.5]*3,[0.5]*3)"],
        "encode_mode": "latent_dist.mode()",
        "decode_output": "vae.decode(z).sample",
        "reconstruction_saved": False,
        "latent_saved": False,
        "score_min": score_min,
        "score_max": score_max,
        "model_load_seconds": model_load_seconds,
        "scoring_loop_seconds": loop_seconds,
        "images_per_second": len(ds) / loop_seconds if loop_seconds else 0.0,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": time.perf_counter() - start_total,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
