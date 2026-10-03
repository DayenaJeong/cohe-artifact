# Author-trained checkpoint release

**Status: approved public release assets.** No checkpoint binary belongs in ordinary Git. The 234 verified checkpoints are distributed through the [`v1.0-neurips2026` GitHub Release](https://github.com/DayenaJeong/cohe-artifact/releases/tag/v1.0-neurips2026). `MANIFEST.csv` contains every candidate from both audited projects, including excluded candidates; only tensor-verified authoritative entries are released. The original file hash and tensor-only export hash are distinct and documented.

Recovered paper endpoints cover CIFAR-100 main selection, the auxiliary alpha-mix sweep, balanced controls, ConvAE, part of DDPM ordering, and the complete ten-run ImageNet-1K best/final endpoint set. DINOv2 selection, ResNet-50 second-learner, and ImageNet-10 trained endpoint weights were not found; their runners did not retain matching saved weights. Six ordering endpoint files are missing. Do not infer a complete checkpoint release from available result tables.

ImageNet checkpoint bundles embedded author-specific config paths. Released exports contain only unchanged model tensors; optimizer/scheduler/scaler/config are excluded. Exact tensor equality was checked using restricted `torch.load(weights_only=True)`. Original research files remain unchanged. Original hashes and release-export hashes are listed in the manifest.

Hosting: GitHub Release assets on the canonical repository. Tag: `v1.0-neurips2026`; title: `NeurIPS 2026 Camera-Ready Release`. The release includes `SHA256SUMS` and public checkpoint/archive manifests. See `docs/RELEASE_RECORD.md`.

## Checkpoint coverage

Checkpoints are released only where the final reported endpoint could be traced, validated, and packaged without private metadata. Some historical weights were not retained. Corresponding seed-level numerical results and reproduction code are provided where available; an absent checkpoint does not mean that the experiment is absent from the paper.

All counts below refer to the verified public release set: **TECHNICALLY_VERIFIED=TRUE; HUMAN_RELEASE_APPROVED=TRUE**.

| Paper experiment | Checkpoints released? | Candidate coverage | Alternative reproducibility resource |
| --- | --- | --- | --- |
| CIFAR main selection | Yes | 120; six methods × four budgets × five seeds | `results/seed_level/cifar100_main/`; CIFAR training runners |
| Balanced controls | Yes | 30; three methods × two budgets × five seeds | `results/aggregate/cifar100_balanced/`; balanced-control runner |
| Alpha-mix | Yes | 45; five alpha values × three budgets × three seeds | `results/aggregate/alpha_mix_results.json`; alpha-mix runner |
| DDPM ordering | Partial | 18 of 24; RANDOM and EASY_TO_HARD seeds 0–2 missing | All eight paired seeds in `experiments/ddpm_ordering/per_seed_results.csv`; recovered ordering runner |
| ImageNet-1K VAE Gate 3 | Yes | 20; best/final × two policies × five seeds | `experiments/imagenet1k_vae_gate3/SEED_LEVEL_RESULTS.csv`; scoring/selection/training/verifier code |
| CIFAR ConvAE | Yes | One seed-0 final epoch-50 reconstruction model | `results/aggregate/convae/`; recovered ConvAE implementation |
| DINOv2 selection | No preserved endpoints | No saved learner endpoint found in audited roots | All four five-seed conditions; selection runner and rank adapters |
| ResNet-50 second learner | No preserved endpoints | No matched saved endpoint found | `results/seed_level/cifar100_second_learner/`; second-learner runner |
| ImageNet-10 | No preserved endpoints | No matched fine-tuning endpoint found | `results/seed_level/imagenet10/`; fine-tuning/scoring runners |

Canonical counts: 308 original checkpoint-like files; 243 author-trained COHE-family files; 234 final-endpoint candidates; 74 exclusions. Exclusions comprise 63 unmatched historical files, two unresolved restricted-loader/provenance files and nine older unbalanced 30% runs with different final accuracies. SHA-identical duplicates: zero. ConvAE is the one auxiliary trained model included in the 234. No upstream pretrained weights are packaged.

## Release archives

Nine archives are published with `v1.0-neurips2026`. The 120 main checkpoints are split into four budget-specific archives; other experiments retain one archive each. This preserves experimental grouping and keeps every asset below GitHub's 2 GiB limit: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases .

Each archive contains only tensor-only checkpoints, a local `MANIFEST.csv`, and a `README.txt` describing use, seeds, budgets and endpoint rules. See `VERIFIED_RELEASE_CANDIDATES.csv`, `RELEASE_ARCHIVES_MANIFEST.csv`, `ARCHIVE_CONTENTS.csv` and `SHA256SUMS`. Archive contents were extracted and matched to the checkpoint hashes. Original research bundles and failed verification attempts remain outside the release tree.
