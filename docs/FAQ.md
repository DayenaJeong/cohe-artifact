# FAQ

**Are checkpoint downloads available?** Yes. The [`v1.0-neurips2026` GitHub Release](https://github.com/DayenaJeong/cohe-artifact/releases/tag/v1.0-neurips2026) provides 234 author-trained checkpoints in nine archives. See [`checkpoints/README.md`](../checkpoints/README.md) for coverage, checksums, and known omissions.

**Do the CPU commands retrain models?** No. They verify scalar interfaces and completed paired-result arithmetic. GPU runners require user-obtained datasets, score inputs, and the documented environment.

**Are all datasets, proxy arrays, and scoring pipelines supplied?** No. Seven CIFAR-100 proxy arrays and selected targets are supplied. ImageNet score arrays, raw images, upstream weights, and some exact historical implementations are not included.

**Does prediction imply a useful selector?** No. DINOv2's first-learning relation remains target-specific despite all twenty tested hard-selection deltas being negative.
