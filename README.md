# COHE Anonymized Artifact

This artifact accompanies the paper **COHE: Auditing Cross-Objective Transfer of Sample Hardness Proxies**. COHE is an evaluation and reporting protocol for checking whether a task-agnostic hardness proxy supports a stated discriminative claim.

## What This Artifact Contains

- Stage-wise audit scripts for dependence, held-out prediction, transfer summaries, and claim-card generation.
- CSV schemas for proxy scores, target scores, split IDs, and transfer results.
- Toy CSV examples for a smoke test.
- Selected derived scalar arrays for CIFAR-100 tabular gate checks, including DINOv2 isolation and first-learning targets.
- Reporting templates for COHE audit cards, claim tiers, and checklist-style reporting.
- Cached aggregate summaries for paper-style table verification, including DINOv2 Gate 3 accuracy, transfer-delta, diagnostic, and gate-status summaries.
- Optional DINOv2 Gate 3 runner and aggregator scripts for users with upstream CIFAR-100 data, DINOv2 isolation scores, first-learning targets, and a seed-matched Random baseline.
- Reviewer-facing reproducibility, claim-tier, and FAQ documentation.
- Asset provenance and anonymity notes.

## What This Artifact Does Not Contain

This anonymized artifact does not redistribute CIFAR, ImageNet, pretrained checkpoints, model weights, or raw images. Users should obtain upstream assets from their original sources and follow their licenses/model cards.

It includes selected derived scalar arrays where available, but it does not include private paths, user-specific metadata, full historical training logs, raw upstream datasets, checkpoints, or full scoring/retraining pipelines.

## Quickstart

From the artifact root:

```bash
bash scripts/smoke_test.sh
```

The smoke test runs all stage scripts on toy CSV files, verifies cached aggregate summaries, renders cached tables, and writes outputs under `outputs/`.

## Reviewer Quickstart

```bash
bash scripts/smoke_test.sh
python scripts/verify_derived_scalar_arrays.py
python scripts/verify_cached_summaries.py
python scripts/render_cached_tables.py
```

Expected outputs:

- `outputs/smoke_test/stage1_dependence.csv`
- `outputs/smoke_test/stage2_prediction.csv`
- `outputs/smoke_test/stage3_transfer.csv`
- `outputs/smoke_test/cohe_claim_card.md`
- `outputs/cached_summary_inventory.md`
- `outputs/rendered_cached_tables.md`

These outputs are generated locally and are excluded from the upload ZIP.

## Derived Scalar Arrays

The `derived_scalar_arrays/` directory contains selected CIFAR-100 train-split scalar arrays with anonymous sample IDs, including DINOv2 isolation, first-learning targets, DDIM reconstruction, CE, margin, GradNorm, and several additional scalar proxies. These are derived tabular outputs only: no images, labels as filenames, local paths, weights, checkpoints, or raw datasets are included.

Reviewers can inspect the release and run one zero-GPU tabular Gate 2 interface check with:

```bash
python3 scripts/verify_derived_scalar_arrays.py
python3 scripts/run_stage2_prediction.py \
  --proxy derived_scalar_arrays/cifar100_dinov2_proxy_scores.csv \
  --target derived_scalar_arrays/cifar100_first_learning_targets.csv \
  --splits derived_scalar_arrays/cifar100_split_ids.csv \
  --out outputs/derived_dinov2_first_learning_prediction.csv
```

These arrays support zero-GPU tabular verification of gate computations and reported audit summaries where the needed scalar targets are included. Full regeneration of proxy scores or retraining-based Gate 3 transfer results still requires upstream datasets/models and GPU scoring or retraining.

## Expected Inputs

COHE operates on sample-indexed CSV files:

- `proxy_scores.csv`: `sample_id,proxy_name,proxy_score`
- `target_scores.csv`: `sample_id,target_name,target_score`
- `split_ids.csv`: `sample_id,split`
- `transfer_results.csv`: `method,budget,seed,accuracy`

Schema examples are provided in `schemas/`, and runnable toy examples are provided in `examples/`.

## Running Individual Stages

```bash
python3 scripts/run_stage1_dependence.py \
  --proxy examples/toy_proxy_scores.csv \
  --target examples/toy_target_scores.csv \
  --out outputs/stage1_dependence.csv

python3 scripts/run_stage2_prediction.py \
  --proxy examples/toy_proxy_scores.csv \
  --target examples/toy_target_scores.csv \
  --splits examples/toy_split_ids.csv \
  --out outputs/stage2_prediction.csv

python3 scripts/run_stage3_transfer_summary.py \
  --transfer examples/toy_transfer_results.csv \
  --out outputs/stage3_transfer.csv

python3 scripts/generate_claim_card.py \
  --stage1 outputs/stage1_dependence.csv \
  --stage2 outputs/stage2_prediction.csv \
  --stage3 outputs/stage3_transfer.csv \
  --out outputs/cohe_claim_card.md
```

## Reproducing COHE-Style Audit Tables

For a new proxy, provide sample-indexed proxy and target CSVs, split IDs for held-out prediction, and optional transfer results from a downstream action protocol. The scripts produce compact summaries that can be copied into the reporting templates in `templates/`.

When Stage 3 retraining is infeasible, report only Stage 1 and Stage 2 evidence. Such reports support alignment or predictive-validity wording only; they should not be described as operational transfer evidence.

The cached files in `cached_summaries/` are aggregate manuscript summaries. Use:

```bash
python3 scripts/verify_cached_summaries.py
python3 scripts/render_cached_tables.py
```

to check that the paper-table summaries are present and machine-readable.

## DINOv2 Gate 3 Audit

The paper's DINOv2 Gate 3 table is represented by aggregate cached files:

- `cached_summaries/dinov2_gate3_transfer_summary.csv`
- `cached_summaries/dinov2_gate3_accuracy_summary.csv`
- `cached_summaries/dinov2_gate3_transfer_delta_summary.csv`
- `cached_summaries/dinov2_gate3_diagnostics_summary.csv`
- `cached_summaries/dinov2_gate3_gate_status.json`

These files are aggregate summaries only. They do not include CIFAR-100 images, selected indices, checkpoints, or training logs.

Users who want to rerun the full transfer audit can use:

```bash
PROXY_CSV=user_inputs/cifar100_dinov2_proxy.csv \
FIRST_LEARNING=user_inputs/first_learning_epoch.npy \
RANDOM_BASELINE=user_inputs/cifar100_random_seed_level_results.csv \
DATA_ROOT=user_inputs/cifar100_data_root \
bash scripts/run_dinov2_gate3_transfer.sh
```

Full reruns require upstream assets plus optional numerical/training dependencies such as `numpy`, `scipy`, `torch`, and `torchvision`. They are not part of the CPU-only toy smoke test or the minimal `requirements.txt`.

## Documentation Map

- `docs/REPRODUCIBILITY.md`: what can and cannot be reproduced from this lightweight artifact.
- `docs/CLAIM_TIER_GUIDE.md`: the five COHE claim tiers, permitted wording, and overclaim boundaries.
- `docs/FAQ.md`: short answers to likely reviewer questions.

## Claim-Tier Note

The generated claim card uses conservative placeholder rules for toy auditing. Final claim-tier assignments should be checked by the user against the full COHE checklist and the intended downstream claim.

## Anonymity Note

The artifact is anonymized for review. It is intended to contain no author names, institutions, local absolute paths, usernames, private URLs, raw datasets, or model weights.

## Asset and License Note

See `LICENSE_OR_TERMS.md` and `provenance/asset_provenance.md`. Users are responsible for obtaining upstream datasets and checkpoints from their original sources and following the corresponding license, access terms, and model-card requirements.

## Post-submission author-response addendum

The clearly separated `rebuttal_addendum_20260727/` directory contains the
author-response artifact additions: seven CIFAR-100 scalar proxy arrays with
integer-index manifests, DINOv2 multi-budget materials, DDPM full-data-ordering
materials, attribution, and reproduction scripts. The original artifact scope
is unchanged; ImageNet-1K arrays are intentionally not redistributed because
their scalar-only release terms remain unresolved.
