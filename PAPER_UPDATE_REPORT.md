# Paper ↔ proposed release consistency

The current paper states that seven CIFAR proxy arrays, selected targets/splits, aggregate results, seed-level results, and verifiers are available, while raw images, upstream weights, ImageNet scalar arrays, and trained checkpoints are not redistributed. The existing public repository supports those current claims; this staging branch and its new checkpoint assets are not public yet.

## After approved source publication

Discussion, Appendix J/J.1, Tables 41–42, and Checklist Q4/Q5 can distinguish recovered training/scoring code from complete pipelines. Table 41's blanket `Full scoring/retraining pipelines: No` should become an explicitly partial/experiment-specific statement only after the additional code is actually released. Table 42 can identify newly released CIFAR, ordering, ImageNet-10, and ImageNet-1K training sources; upstream datasets/models/GPU and unresolved historical tags remain required.

## After approved checkpoint publication

Appendix J's `trained checkpoints` exclusion must be narrowed to checkpoints not supplied. Add exact released endpoint coverage and links in J.1. State the missing DINOv2, ResNet-50, ImageNet-10, and partial ordering checkpoint coverage rather than claiming all models are available. Update Tables 41–42 to distinguish code, scalar arrays, final weights, seed results, and full regeneration.

## Keep unchanged

No raw images/dataset archives or upstream pretrained weights are included. ImageNet scalar arrays remain excluded. Selected target coverage and historical checkpoint/license limitations remain. Q5 public access is already Yes for the current artifact; new local assets do not require another answer change. Q4 reproducibility, compute, and asset-license answers must not be changed merely because code/checkpoint candidates were prepared. No scientific numbers or manuscript source were edited in this release audit.

PAPER_UPDATE_REQUIRED = TRUE, conditional on actual approved publication of additional code/checkpoints. No claim of live checkpoint availability should be added now.
