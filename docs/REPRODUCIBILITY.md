# Reproducibility Scope

This artifact is a lightweight, anonymized COHE evaluation resource. It supports rerunning the audit mechanics from user-provided CSVs and verifying that the cached aggregate summaries used for paper-style tables are machine-readable.

## What Can Be Reproduced

- Stage 1 dependence summaries from sample-indexed proxy and target CSVs.
- Stage 2 held-out linear predictive-validity summaries from proxy, target, and split CSVs.
- Stage 3 transfer summaries from method/budget/seed accuracy CSVs.
- Conservative claim-card drafts from stage summaries.
- Verification of selected released scalar arrays for tabular gate checks.
- Inventory checks for cached aggregate paper-table summaries.
- Markdown renderings of cached aggregate summaries.

## What Cannot Be Reproduced Here

The artifact does not regenerate proxy scores, retrain classifiers, run ImageNet-scale scoring, or download upstream model checkpoints. Those steps require upstream datasets, model checkpoints, training code, and compute that are not redistributed for anonymity, licensing, and size reasons.

This artifact includes selected derived scalar arrays where available, but it does not contain datasets, raw images, pretrained weights, checkpoints, private paths, full historical training logs, or full scoring/retraining pipelines.

## Stage Scripts

- `scripts/run_stage1_dependence.py`: computes Pearson and Spearman dependence for matched samples. HSIC or other nonlinear checks are optional diagnostics and are not required by this lightweight script.
- `scripts/run_stage2_prediction.py`: fits a one-dimensional linear predictor on the train split and reports held-out R2 on the test split.
- `scripts/run_stage3_transfer_summary.py`: summarizes method accuracy and method-vs-Random deltas by budget.
- `scripts/generate_claim_card.py`: combines available stage summaries into a conservative draft claim card.
- `scripts/run_dinov2_gate3_transfer.py` and `scripts/aggregate_dinov2_gate3_transfer.py`: optional full DINOv2 Gate 3 transfer rerun for users who provide upstream CIFAR-100 data, DINOv2 scores, first-learning targets, seed-matched Random baselines, and PyTorch/torchvision dependencies.

## Toy Examples

The `examples/` directory contains small CSVs with the expected columns. They are intentionally tiny and are used only to verify script behavior and file formats.

Run:

```bash
bash scripts/smoke_test.sh
```

Outputs are written under `outputs/`, which is excluded from the upload ZIP.

## Derived Scalar Arrays

The `derived_scalar_arrays/` directory contains selected CIFAR-100 train-split scalar arrays with anonymous sample IDs, including DINOv2 isolation scores. These arrays support zero-GPU tabular verification of gate computations and reported audit summaries where the required scalar targets are included. They do not replace full regeneration of proxy scores or retraining-based Gate 3 transfer results, which still require upstream assets and GPU scoring or retraining.

## Cached Aggregate Summaries

The `cached_summaries/` directory contains table-level aggregate values reported in the manuscript. They are intended to let reviewers inspect and machine-read the reported evidence blocks without rerunning proxy-score generation or retraining.

The DINOv2 Gate 3 cached files include aggregate accuracy, paired-delta, subset-diagnostic, and gate-status summaries for the direct 10% CIFAR-100 subset-selection audit. They do not include selected indices, raw images, checkpoints, or logs.

The verification scripts check paper-style table inputs rather than recomputing expensive upstream scores:

```bash
python scripts/verify_cached_summaries.py
python scripts/render_cached_tables.py
```

These scripts verify that aggregate CSVs exist, parse correctly, and can be rendered into simple Markdown tables.
