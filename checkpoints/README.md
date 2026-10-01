# Author-trained checkpoint release proposal

**Status: local prepared assets only; not uploaded.** No checkpoint binary belongs in ordinary Git. `MANIFEST.csv` contains every candidate from both audited projects, including excluded candidates; only tensor-verified entries are proposed assets. They still have `PENDING_LICENSE_APPROVAL` status. The original file hash and the tensor-only export hash are distinct and documented.

Recovered paper endpoints cover CIFAR-100 main selection, the auxiliary alpha-mix sweep, balanced controls, ConvAE, part of DDPM ordering, and the complete ten-run ImageNet-1K best/final endpoint set. DINOv2 selection, ResNet-50 second-learner, and ImageNet-10 trained endpoint weights were not found; their runners did not retain matching saved weights. Six ordering endpoint files are missing. Do not infer a complete checkpoint release from available result tables.

ImageNet checkpoint bundles embedded author-specific config paths. Proposed exports contain only unchanged model tensors; optimizer/scheduler/scaler/config are excluded. Exact tensor equality was checked using restricted `torch.load(weights_only=True)`. Original research files remain unchanged. Original hashes and release-export hashes are listed in the manifest.

Preferred hosting: GitHub Release assets on the canonical repository. Proposed tag: `v1.0-neurips2026`; title: `NeurIPS 2026 Camera-Ready Release`. After explicit publication approval, downloadable assets and SHA256SUMS can be linked here. Current release listing: https://github.com/DayenaJeong/cohe-artifact/releases . No future download URL is asserted as live.

Before upload: resolve author-code/checkpoint distribution terms, confirm provenance-review exclusions, and obtain the requested final push/release approval. See `docs/RELEASE_PROPOSAL.md`.
