#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP semantic-ish generative proxy for CIFAR-100 using DDIM reconstructions.

We compute:
  D_gen^CLIP(i) = 1 - cos( CLIP(x_i), CLIP(recon_i) )

Outputs:
  - clip_semantic_proxy.npy  (per-sample distances; NaN if missing recon)
  - clip_semantic_results.json (summary + optional corr with disc/gen arrays)

Expected recon file naming in --recon_dir (any one is fine):
  {idx}.png / {idx:05d}.png / {idx:06d}.png (also .jpg/.jpeg/.webp)

Example:
  python scripts_orth/clip_semantic_proxy.py \
    --recon_dir outputs_ddim_recon/cifar100_train \
    --out_dir outputs_clip_proxy \
    --split train \
    --root ./data \
    --disc train_d_disc.npy \
    --gen  train_d_gen.npy \
    --batch_size 256
"""

import os
import json
import math
import argparse
from typing import Optional, Tuple, List

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
import torchvision.transforms as T


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def find_recon_path(recon_dir: str, idx: int) -> Optional[str]:
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    stems = [f"{idx}", f"{idx:05d}", f"{idx:06d}", f"img_{idx}", f"img_{idx:05d}", f"img_{idx:06d}"]
    for stem in stems:
        for ext in exts:
            cand = os.path.join(recon_dir, stem + ext)
            if os.path.exists(cand):
                return cand
    return None


def pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    # lightweight Pearson + p-value (approx via scipy if available, else NaN p)
    try:
        from scipy.stats import pearsonr as _pearsonr
        r, p = _pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        x = x - x.mean()
        y = y - y.mean()
        denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12
        r = float((x @ y) / denom)
        return r, float("nan")


def spearmanr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    try:
        from scipy.stats import spearmanr as _spearmanr
        rho, p = _spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        # fallback: rank + pearson (p-value NaN)
        x_rank = x.argsort().argsort().astype(np.float64)
        y_rank = y.argsort().argsort().astype(np.float64)
        r, _ = pearsonr(x_rank, y_rank)
        return float(r), float("nan")


def zscore(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    mu = np.nanmean(a)
    sd = np.nanstd(a) + 1e-12
    return (a - mu) / sd


# ---------------------------
# CLIP loader (open_clip preferred, transformers fallback)
# ---------------------------
def load_clip(model_name: str = "ViT-B-32", device: str = "cuda"):
    """
    Returns: (encode_fn, preprocess_fn, backend_name)
      - encode_fn: images_tensor[B,3,H,W] -> feat[B,D] (L2-normalized)
      - preprocess_fn: PIL -> tensor[3,H,W]
    """
    # 1) open_clip
    try:
        import open_clip  # pip install open_clip_torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained="openai",
            device=device
        )
        model.eval()

        def _encode(imgs: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                feats = model.encode_image(imgs)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
                return feats

        return _encode, preprocess, "open_clip"

    except Exception:
        pass

    # 2) transformers
    try:
        from transformers import CLIPModel, CLIPProcessor  # pip install transformers

        hf_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(hf_name).to(device)
        model.eval()
        processor = CLIPProcessor.from_pretrained(hf_name)

        def _preprocess(pil: Image.Image) -> torch.Tensor:
            # returns tensor [3,H,W]
            out = processor(images=pil, return_tensors="pt")
            # pixel_values: [1,3,224,224]
            return out["pixel_values"][0]

        def _encode(imgs: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                # imgs: [B,3,224,224]
                feats = model.get_image_features(pixel_values=imgs)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
                return feats

        return _encode, _preprocess, "transformers"

    except Exception as e:
        raise RuntimeError(
            "Failed to import CLIP backend. Install one of:\n"
            "  pip install open_clip_torch\n"
            "or\n"
            "  pip install transformers\n"
        ) from e


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_dir", type=str, required=True, help="Directory containing DDIM recon images named by index.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for .npy and .json.")
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"], help="CIFAR-100 split to evaluate.")
    ap.add_argument("--root", type=str, default="./data", help="torchvision CIFAR-100 root (download cache).")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip_model", type=str, default="ViT-B-32", help="open_clip model name (default ViT-B-32).")

    # optional: correlate with existing hardness arrays
    ap.add_argument("--disc", type=str, default=None, help="Path to discriminative hardness .npy (e.g., train_d_disc.npy).")
    ap.add_argument("--gen", type=str, default=None, help="Path to generative hardness .npy (e.g., train_d_gen.npy).")

    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    # Load CLIP
    encode_img, preprocess, backend = load_clip(model_name=args.clip_model, device=device)
    print(f"Loaded CLIP backend: {backend} | model: {args.clip_model}")

    # CIFAR-100 without transform (we will preprocess via CLIP)
    is_train = (args.split == "train")
    ds = torchvision.datasets.CIFAR100(root=args.root, train=is_train, download=True, transform=None)
    n = len(ds)
    print(f"CIFAR-100 split='{args.split}' size={n}")

    # Pre-allocate distances (NaN if missing recon)
    d_clip = np.full((n,), np.nan, dtype=np.float32)

    # We’ll iterate indices and batch the *available* pairs
    batch_pils_orig: List[Image.Image] = []
    batch_pils_rec: List[Image.Image] = []
    batch_indices: List[int] = []
    missing = 0

    def flush_batch():
        nonlocal batch_pils_orig, batch_pils_rec, batch_indices, d_clip
        if len(batch_indices) == 0:
            return

        # preprocess -> tensor batch
        orig_t = torch.stack([preprocess(im) for im in batch_pils_orig], dim=0)
        rec_t  = torch.stack([preprocess(im) for im in batch_pils_rec], dim=0)

        orig_t = orig_t.to(device, non_blocking=True)
        rec_t  = rec_t.to(device, non_blocking=True)

        # encode
        with torch.no_grad():
            if device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    f_o = encode_img(orig_t)
                    f_r = encode_img(rec_t)
            else:
                f_o = encode_img(orig_t)
                f_r = encode_img(rec_t)

            # cosine distance
            # cos = sum(f_o * f_r) since L2-normalized
            cos = (f_o * f_r).sum(dim=-1).clamp(-1.0, 1.0)
            dist = (1.0 - cos).detach().float().cpu().numpy()

        for idx, val in zip(batch_indices, dist):
            d_clip[idx] = float(val)

        batch_pils_orig, batch_pils_rec, batch_indices = [], [], []

    pbar = tqdm(range(n), desc="Computing CLIP(x, recon(x)) distance", ncols=100)
    for idx in pbar:
        recon_path = find_recon_path(args.recon_dir, idx)
        if recon_path is None:
            missing += 1
            if missing % 200 == 0:
                pbar.set_postfix_str(f"missing={missing}")
            continue

        # get original PIL (CIFAR returns PIL already)
        orig_pil, _ = ds[idx]
        if not isinstance(orig_pil, Image.Image):
            orig_pil = T.ToPILImage()(orig_pil)

        rec_pil = Image.open(recon_path).convert("RGB")

        batch_pils_orig.append(orig_pil)
        batch_pils_rec.append(rec_pil)
        batch_indices.append(idx)

        if len(batch_indices) >= args.batch_size:
            flush_batch()

    flush_batch()

    n_used = int(np.isfinite(d_clip).sum())
    print(f"Done. n_used={n_used}/{n} | missing_recon={missing}")

    # Save per-sample proxy
    out_npy = os.path.join(args.out_dir, f"cifar100_{args.split}_d_gen_clip.npy")
    np.save(out_npy, d_clip)
    print(f"[Saved] {out_npy}")

    # Optional correlations
    results = {
        "dataset": "CIFAR-100",
        "split": args.split,
        "n_total": n,
        "n_used": n_used,
        "missing_recon": missing,
        "recon_dir": os.path.abspath(args.recon_dir),
        "clip": {"backend": backend, "model": args.clip_model},
        "seed": args.seed,
        "outputs": {"d_gen_clip_npy": os.path.abspath(out_npy)},
        "corrs_on_valid": {},
        "notes": {
            "definition": "D_gen^CLIP(i) = 1 - cos(CLIP(x_i), CLIP(recon_i))",
            "valid_mask": "finite D_gen^CLIP (requires recon image present)",
        }
    }

    valid_mask = np.isfinite(d_clip)
    d_clip_valid = d_clip[valid_mask].astype(np.float64)

    # correlate against provided arrays if present (aligned by index)
    if args.disc is not None and os.path.exists(args.disc):
        d_disc = np.load(args.disc).astype(np.float64)
        d_disc_valid = d_disc[valid_mask]
        r, p = pearsonr(zscore(d_clip_valid), zscore(d_disc_valid))
        rho, sp = spearmanr(zscore(d_clip_valid), zscore(d_disc_valid))
        results["corrs_on_valid"]["clip_vs_disc"] = {
            "pearson_r": r, "pearson_p": p,
            "spearman_rho": rho, "spearman_p": sp,
            "disc_path": os.path.abspath(args.disc),
        }

    if args.gen is not None and os.path.exists(args.gen):
        d_gen = np.load(args.gen).astype(np.float64)
        d_gen_valid = d_gen[valid_mask]
        r, p = pearsonr(zscore(d_clip_valid), zscore(d_gen_valid))
        rho, sp = spearmanr(zscore(d_clip_valid), zscore(d_gen_valid))
        results["corrs_on_valid"]["clip_vs_existing_gen"] = {
            "pearson_r": r, "pearson_p": p,
            "spearman_rho": rho, "spearman_p": sp,
            "gen_path": os.path.abspath(args.gen),
        }

    # save json
    out_json = os.path.join(args.out_dir, f"cifar100_{args.split}_clip_semantic_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {out_json}")

    # quick print summary
    if "clip_vs_disc" in results["corrs_on_valid"]:
        cd = results["corrs_on_valid"]["clip_vs_disc"]
        print(f"CLIP proxy vs Disc hardness: pearson={cd['pearson_r']:.6f} (p={cd['pearson_p']:.4g}), "
              f"spearman={cd['spearman_rho']:.6f} (p={cd['spearman_p']:.4g})")


if __name__ == "__main__":
    main()
