# Checkpoint-policy audit

## Recovered rule

The recovered runner `train_resnet18_gate3_20260727.py` saves `best.pt` inside
the epoch loop when `validation_top1` strictly exceeds the previous best.  It
saves `final.pt` after epoch 90.  The launch script fixes `--epochs 90` and
uses the same runner and configuration for both policies.  The stored
`run_status.json` records the same best Top-1 metric used by `best.pt`.

Primary policy: **best validation Top-1 checkpoint**.
Sensitivity policy: **final epoch 90**.

The exact runner SHA256 is `090f33ec50a3a4e97e06c8bf0bd387da7ba14cd18b8f36650dc7f1bc9fb31a39` and the launch
pipeline SHA256 is `6c8e54967749557b3f9c0e1b44658d0d31beed68eff07510b2e8f656593f48e3`.  All ten per-run configs
have identical protocol fields after removing only the permitted dynamic
fields (policy, seed, selector path/hash, status, best metric, and total-step
counter).  Both policies use `weights=None`, 90 epochs, batch size 256,
SGD, the same scheduler, AMP, transforms, and full validation evaluation.

## Best-checkpoint rows

| seed | Hard best epoch | Hard Top-1 | Hard Top-5 | Random best epoch | Random Top-1 | Random Top-5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 70 | 53.812 | 76.982 | 82 | 61.272 | 83.538 |
| 1 | 79 | 53.230 | 76.482 | 87 | 61.482 | 83.464 |
| 2 | 66 | 53.648 | 76.742 | 88 | 61.346 | 83.520 |
| 3 | 86 | 53.698 | 76.838 | 72 | 61.612 | 83.550 |
| 4 | 75 | 53.844 | 76.956 | 86 | 61.346 | 83.378 |

The primary Top-1 mean delta is -7.765200 pp.
The final-epoch Top-1 mean delta is -7.853600 pp.
