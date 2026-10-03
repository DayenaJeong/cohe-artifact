#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from diffusers import DDPMScheduler, StableDiffusionPipeline
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


def build_dataset(name: str, root: str, split: str, image_size: int):
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    if name == "cifar100":
        train = split == "train"
        return torchvision.datasets.CIFAR100(root=root, train=train, download=False, transform=tfm)
    if name == "imagenet1k":
        return torchvision.datasets.ImageNet(root=root, split=split, transform=tfm)
    raise ValueError(f"Unsupported dataset: {name}")


def parse_timesteps(spec: str) -> list[int]:
    vals = [int(v.strip()) for v in spec.split(",") if v.strip()]
    if not vals:
        raise ValueError("Need at least one timestep")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar100", "imagenet1k"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--timesteps", default="500")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    use_amp = device.startswith("cuda")
    dtype = torch.float16 if use_amp else torch.float32

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = build_dataset(args.dataset, args.root, args.split, args.image_size)
    shard = IndexedShardDataset(ds, args.shard_id, args.num_shards)
    loader = DataLoader(
        shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        persistent_workers=args.num_workers > 0,
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        local_files_only=True,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.vae = pipe.vae.to(device)
    pipe.text_encoder = pipe.text_encoder.to(device)
    pipe.unet = pipe.unet.to(device)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.unet.eval()

    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    t_values = parse_timesteps(args.timesteps)

    tokenized = pipe.tokenizer(
        [args.prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids.to(device)
    with torch.inference_mode():
        prompt_embeds_1 = pipe.text_encoder(input_ids)[0]

    all_indices: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else torch.autocast(device_type="cpu", dtype=torch.float32)
    )

    with torch.inference_mode():
        for x, y, idx in tqdm(
            loader,
            desc=f"sd-latent-denoise {args.dataset} {args.split} shard {args.shard_id}/{args.num_shards}",
            ncols=110,
        ):
            x = x.to(device, non_blocking=True)
            b = x.shape[0]
            prompt_embeds = prompt_embeds_1.repeat(b, 1, 1)

            with autocast_ctx:
                latents = pipe.vae.encode(x).latent_dist.mode() * pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            score_accum = torch.zeros(b, device=device, dtype=torch.float32)

            for t_val in t_values:
                timesteps = torch.full((b,), t_val, device=device, dtype=torch.long)
                with autocast_ctx:
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    noise_pred = pipe.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                    ).sample
                score_accum += (noise_pred.float() - noise.float()).pow(2).mean(dim=(1, 2, 3))

            score = score_accum / float(len(t_values))
            all_indices.append(idx.numpy())
            all_targets.append(y.numpy())
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
        "image_size": int(args.image_size),
        "timesteps": t_values,
        "prompt": args.prompt,
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "seed": int(args.seed),
        "n_samples": int(sum(len(x) for x in all_indices)),
        "score_definition": "Average latent noise-prediction MSE between true Gaussian noise and SD-v1.5 UNet prediction at fixed timesteps after VAE encoding; lower implies the latent diffusion prior predicts the sample more easily.",
    }
    out_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
