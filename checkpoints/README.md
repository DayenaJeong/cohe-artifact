# Checkpoints

The [`v1.0-neurips2026` GitHub Release](https://github.com/DayenaJeong/cohe-artifact/releases/tag/v1.0-neurips2026) provides 234 author-trained checkpoints in nine archives. Checkpoint binaries are not stored in ordinary Git, and third-party pretrained weights are not included.

## Coverage

Available checkpoints cover CIFAR-100 main selection, balanced controls, the alpha-mix sweep, ConvAE, part of DDPM ordering, and the ImageNet-1K VAE Gate 3 runs.

| Paper experiment | Checkpoints | Coverage | Other resources |
| --- | --- | --- | --- |
| CIFAR-100 main selection | 120 | Six methods × four budgets × five seeds | `results/seed_level/cifar100_main/`; training runners |
| Balanced controls | 30 | Three methods × two budgets × five seeds | `results/aggregate/cifar100_balanced/`; balanced-control runner |
| Alpha-mix | 45 | Five alpha values × three budgets × three seeds | `results/aggregate/alpha_mix_results.json`; alpha-mix runner |
| DDPM ordering | 18 of 24 | Random and Easy-to-Hard seeds 0–2 missing | `experiments/ddpm_ordering/per_seed_results.csv`; ordering runner |
| ImageNet-1K VAE Gate 3 | 20 | Best/final × two policies × five seeds | `experiments/imagenet1k_vae_gate3/SEED_LEVEL_RESULTS.csv`; scoring, selection, training, and verification code |
| CIFAR-100 ConvAE | 1 | Seed-0 final epoch-50 reconstruction model | `results/aggregate/convae/`; ConvAE implementation |
| DINOv2 selection | None | No saved learner endpoint was retained | Four five-seed result conditions; selection runner and rank adapters |
| ResNet-50 second learner | None | No matched saved endpoint was retained | `results/seed_level/cifar100_second_learner/`; second-learner runner |
| ImageNet-10 | None | No matched fine-tuning endpoint was retained | `results/seed_level/imagenet10/`; fine-tuning and scoring runners |

The missing DDPM ordering endpoints are Random seeds 0–2 and Easy-to-Hard seeds 0–2. Seed-level results and reproduction code remain available where listed.

## Archives and manifests

Nine archives are published with `v1.0-neurips2026`. The 120 main-selection checkpoints are split into four budget-specific archives; the remaining experiment groups use one archive each. Each archive contains tensor-only checkpoints, a local `MANIFEST.csv`, and a `README.txt` describing architectures, seeds, budgets, and endpoint rules.

- [`VERIFIED_RELEASE_CANDIDATES.csv`](VERIFIED_RELEASE_CANDIDATES.csv) lists the 234 released files and their metadata.
- [`RELEASE_ARCHIVES_MANIFEST.csv`](RELEASE_ARCHIVES_MANIFEST.csv) lists archive sizes, SHA-256 digests, and checkpoint counts.
- [`ARCHIVE_CONTENTS.csv`](ARCHIVE_CONTENTS.csv) maps checkpoints to archives.
- [`MANIFEST.csv`](MANIFEST.csv) is the broader 308-candidate discovery inventory and includes excluded historical files.

Archive contents were extracted and matched to their listed hashes before publication. The GitHub Release includes a release-level `SHA256SUMS` file for verifying downloaded assets.

## Loading

Use restricted loading with the architecture named in the archive manifest:

```python
import torch

state = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
```

The exports contain model tensors only; optimizer, scheduler, scaler, and author-specific configuration fields are excluded. Use the recovered ConvAE class for the ConvAE checkpoint.
