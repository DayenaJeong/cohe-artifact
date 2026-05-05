# COHE Anonymized Artifact

This artifact accompanies the paper **COHE: Auditing Cross-Objective Transfer of Sample Hardness Proxies**. COHE is an evaluation and reporting protocol for checking whether a task-agnostic hardness proxy supports a stated discriminative claim.

## Reviewer Quickstart

```bash
pip install -r requirements.txt
bash scripts/smoke_test.sh
python scripts/verify_cached_summaries.py
python scripts/render_cached_tables.py
```

The smoke test uses only toy CSVs and cached aggregate summaries. CUDA is not required.

Expected outputs:

- `outputs/smoke_test/stage1_dependence.csv`
- `outputs/smoke_test/stage2_prediction.csv`
- `outputs/smoke_test/stage3_transfer.csv`
- `outputs/smoke_test/cohe_claim_card.md`
- `outputs/cached_summary_inventory.md`
- `outputs/rendered_cached_tables.md`

These outputs are generated locally and ignored by git.

## What This Artifact Contains

- Stage-wise audit scripts for dependence, held-out prediction, transfer summaries, and claim-card generation.
- CSV schemas for proxy scores, target scores, split IDs, and transfer results.
- Toy CSV examples for a smoke test.
- Reporting templates for COHE audit cards, claim tiers, and checklist-style reporting.
- Claim-tier guide and reporting templates.
- Cached aggregate summaries for paper-style table verification.
- Reviewer-facing reproducibility, claim-tier, and FAQ documentation.
- Asset provenance and anonymity notes.

## What This Artifact Does Not Contain

This anonymized artifact does not redistribute CIFAR, ImageNet, pretrained checkpoints, model weights, or raw images. Users should obtain upstream assets from their original sources and follow their licenses/model cards.

It also does not include datasets, full raw proxy arrays, private paths, user-specific metadata, or full historical training logs.

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

The cached files in `cached_summaries/` are aggregate manuscript summaries, not raw sample-level data. Use:

```bash
python3 scripts/verify_cached_summaries.py
python3 scripts/render_cached_tables.py
```

to check that the paper-table summaries are present and machine-readable.

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
