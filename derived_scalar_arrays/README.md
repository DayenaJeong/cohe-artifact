# Derived Scalar Arrays

This folder releases selected CIFAR-100 train-split scalar arrays used for COHE gate checks. The files contain anonymous `sample_id` values and derived scalar outputs only.

We release derived scalar arrays only -- no images, weights, checkpoints, or raw datasets. These arrays support zero-GPU reproduction of gate computations and reported tabular audits where the required scalar targets and cached transfer summaries are included. Full regeneration of proxy arrays or retraining-based transfer accuracies still requires upstream datasets/models and GPU scoring or retraining.

## Files

- `cifar100_scalar_audit_arrays.csv`: wide table with anonymous sample IDs, available proxy scores, and available discriminative targets.
- `cifar100_split_ids.csv`: deterministic 80/20 split for gate-script demonstrations on the released tabular arrays. Cached manuscript summaries remain the source for reported multi-split results.
- `cifar100_ddim_proxy_scores.csv` and `cifar100_ce_targets.csv`: convenience projection for DDIM-to-CE Gate 1/2 checks.
- `cifar100_dinov2_proxy_scores.csv` and `cifar100_first_learning_targets.csv`: convenience projection for the DINOv2 isolation boundary Gate 1/2 checks.
- `schema.json` and `manifest.json`: column definitions, row counts, hashes, and missing/excluded scalar columns.

## Quick Checks

```bash
python scripts/verify_derived_scalar_arrays.py
python scripts/run_stage2_prediction.py \
  --proxy derived_scalar_arrays/cifar100_dinov2_proxy_scores.csv \
  --target derived_scalar_arrays/cifar100_first_learning_targets.csv \
  --splits derived_scalar_arrays/cifar100_split_ids.csv \
  --out outputs/derived_dinov2_first_learning_prediction.csv
```

The second command is a zero-GPU interface check on one released split. It is not a replacement for the paper's cached multi-split summaries or the DINOv2 Gate 3 retraining audit.
