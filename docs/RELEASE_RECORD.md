# GitHub Release record

- Canonical repository: https://github.com/DayenaJeong/cohe-artifact
- Release tag: `v1.0-neurips2026`
- Tagged commit: `01b7d54050d1458d8a0f15f3b2c07c95b52a5f3f`
- Release title: NeurIPS 2026 Camera-Ready Release
- Release URL: https://github.com/DayenaJeong/cohe-artifact/releases/tag/v1.0-neurips2026

Source code, released scalars, final results, provenance, environments, and lightweight manifests are ordinary Git contents. Checkpoint binaries are not committed to Git; the archives and release-level `SHA256SUMS` are published as GitHub Release assets.

The release contains 234 technically validated author-trained checkpoints in nine experiment archives. The main CIFAR group is split by 10/30/50/70% budget; the five remaining experiment groups each have one archive. Every archive was extracted and checked against its member hashes. `checkpoints/RELEASE_ARCHIVES_MANIFEST.csv` is the canonical archive inventory, and the release-level `SHA256SUMS` verifies the published assets.

All nine archives satisfy GitHub's per-asset size bound. Remote asset sizes and SHA-256 digests were checked against the public manifest after publication without re-downloading the complete archive set.

`TECHNICALLY_VERIFIED=TRUE` and `HUMAN_RELEASE_APPROVED=TRUE` are recorded separately. Author-created source code uses the MIT License. The license does not cover or alter upstream dataset, model, or pretrained-weight terms.

Raw datasets/images, upstream pretrained weights, ImageNet scalar arrays, unknown historical weights, and private inventories/logs are not included.

Checkpoint omissions remain explicit: DINOv2 selection, CIFAR-100 ResNet-50 second-learner, and ImageNet-10 endpoints were not retained; DDPM ordering lacks Random seeds 0–2 and Easy-to-Hard seeds 0–2. Numerical records and reproduction materials remain available where documented.
