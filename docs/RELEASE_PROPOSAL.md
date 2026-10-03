# GitHub Release plan

- Canonical repository: DayenaJeong/cohe-artifact
- Existing branch base: 79b1a6d6bbc64e29f6da74e2118ee967e8cc493b
- Staging branch: camera-ready-public-release
- Proposed tag: v1.0-neurips2026
- Proposed title: NeurIPS 2026 Camera-Ready Release
- Approved release URL: https://github.com/DayenaJeong/cohe-artifact/releases/tag/v1.0-neurips2026

Source code, released scalars, final results, provenance, environments, and manifests are ordinary Git contents. Checkpoint archives and `SHA256SUMS` stay outside Git and are published as GitHub Release assets. `checkpoints/PROPOSED_ASSETS.csv` lists model sizes and hashes.

The release contains 234 technically validated checkpoints in nine experiment archives under ignored `release_assets/v1.0-neurips2026/`. The main CIFAR group is split by 10/30/50/70% budget; the five remaining experiment groups each have one archive. Every archive was extracted and checked against its member hashes. `checkpoints/RELEASE_ARCHIVES_MANIFEST.csv` is the canonical archive inventory; `checkpoints/SHA256SUMS` verifies archive and important source metadata files from the checkpoints directory. The asset-directory `SHA256SUMS` verifies upload candidates from that directory. Both exclude their own checksum file to avoid self-reference.

Use GitHub Release assets rather than normal Git for checkpoint binaries. GitHub permits up to 1000 assets per release and requires each asset to be under 2 GiB: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases . Proposed individual model files satisfy that size bound.

The nine prepared archives each satisfy the asset-size bound. `TECHNICALLY_VERIFIED=TRUE` and `HUMAN_RELEASE_APPROVED=TRUE` are recorded separately. Author-created source code uses the MIT License; this does not alter upstream dataset/model/weight terms.

Do not publish raw datasets/images, upstream pretrained weights, ImageNet scalar arrays, unknown historical weights, private inventories/logs, or the local staging audit directory. The code license does not resolve or alter upstream rights.
