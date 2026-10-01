# GitHub Release proposal

- Canonical repository: DayenaJeong/cohe-artifact
- Existing branch base: 79b1a6d6bbc64e29f6da74e2118ee967e8cc493b
- Staging branch: camera-ready-public-release
- Proposed tag: v1.0-neurips2026
- Proposed title: NeurIPS 2026 Camera-Ready Release
- Publication state: no push, merge, tag creation, release creation, or asset upload performed.

Source tree, previously intentionally released scalars, final results, provenance, environments, and manifests are proposed Git contents. Prepared checkpoint assets and `SHA256SUMS.txt` stay outside Git and must remain held until terms/provenance and publication approval are complete. `checkpoints/PROPOSED_ASSETS.csv` lists their sizes and hashes.

Use GitHub Release assets rather than normal Git for checkpoint binaries. GitHub permits up to 1000 assets per release and requires each asset to be under 2 GiB: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases . Proposed individual model files satisfy that size bound.

Do not publish raw datasets/images, upstream pretrained weights, ImageNet scalar arrays, unknown historical weights, private inventories/logs, or the local staging audit directory. A code license does not resolve upstream rights. Final approval is required before any remote mutation.
