# Final public release validation

Validation date: 2026-10-03 (Asia/Seoul)

- Public repository: https://github.com/DayenaJeong/cohe-artifact
- Fixed release tag: `v1.0-neurips2026`
- Tagged commit: `01b7d54050d1458d8a0f15f3b2c07c95b52a5f3f`
- GitHub Release: https://github.com/DayenaJeong/cohe-artifact/releases/tag/v1.0-neurips2026
- Release assets: 13 total, including 9 checkpoint archives
- Public checkpoints: 234
- Checkpoint archive bytes: 9,840,489,315

This report is post-tag validation evidence. It does not move or modify the fixed release tag.

## Validation results

| Check | Result |
| --- | --- |
| Anonymous HTTPS clone of the public repository | PASS |
| Checkout of `v1.0-neurips2026` | PASS |
| Tag resolves to the intended merged commit | PASS |
| Required README, license, citation, result, and checkpoint metadata files | PASS |
| README CPU smoke test | PASS |
| Public CLI help and experiment dispatch | PASS |
| Derived scalar-array verification | PASS |
| Cached aggregate-summary verification | PASS |
| DINOv2, DDPM ordering, and ImageNet-1K paired-statistics verification | PASS |
| Complete supplied paper-result verification | PASS |
| Official Citation File Format 1.2 schema | PASS |
| All 380 tagged source-manifest paths, sizes, and SHA-256 values | PASS |
| Checkpoint candidate count and unique hashes (234/234) | PASS |
| Release archive count (9) and checkpoint count (234) | PASS |
| Public GitHub Release page and unauthenticated release API | PASS |
| GitHub-reported archive sizes and SHA-256 digests versus tagged manifest | PASS |
| Security/private-path scan before publication | PASS; no findings |
| Duplicate checkpoint hashes | PASS; zero |
| Stale unbalanced-30% checkpoints included | PASS; none included |

No GPU training, dataset download, or expensive experiment rerun was performed. The 9.84 GB asset set was not re-downloaded: local pre-upload hashes were compared to GitHub-reported asset digests and sizes, while the unauthenticated public API and fresh tagged clone independently verified public availability and the tagged manifests.

## Documented coverage boundaries

DINOv2 selection, CIFAR-100 ResNet-50 second-learner, and ImageNet-10 endpoint checkpoints were not retained. DDPM ordering lacks Random seeds 0–2 and Easy-to-Hard seeds 0–2. The exact shuffled-DDIM negative-control runner, exact cross-architecture runner, historical timestep-sweep source/provenance, and some predictive-analysis inputs remain incomplete or are not redistributed. Raw datasets and third-party pretrained model weights are not redistributed.

`PUBLIC_RELEASE_VALIDATED=TRUE`
