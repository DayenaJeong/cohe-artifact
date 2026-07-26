# COHE author-response artifact addendum (2026-07-27)

This directory is a clearly separated, post-submission author-response addendum
to the existing anonymous artifact. The original artifact files and their scope
are unchanged; the repository root README contains only a short pointer here.

## Included

- Seven verified, one-dimensional CIFAR-100 training scalar arrays: DDPM, DDIM,
  ConvAE, SD-VAE, SD-v1.5, Emu3 VQ, and DINOv2.
- Integer-index sample-order manifests and per-array metadata/checksums. These
  manifests contain no filenames, local paths, image bytes, or checkpoints.
- DINOv2 multi-budget selection indices, sanitized per-seed results, configuration,
  and a CPU-only statistics reproduction script.
- DDPM full-data ordering materials: the eight-seed Easy/Random/Shuffled results,
  the shuffled-proxy control, a privacy-safe bin-membership manifest, configuration,
  and reproduction/verification scripts.
- Attribution and an OpenReview-ready notice.

## Deliberate exclusions

No ImageNet-1K arrays, filenames, sample IDs, paths, images, reconstructions,
latents, features, model weights, checkpoints, or raw training logs are included.
The terms governing scalar redistribution for ImageNet were not sufficiently
clear to claim release. This addendum therefore does not claim ImageNet artifact
reproducibility.

The included scalar arrays are derived values only. Upstream models and datasets
remain the responsibility of the user; see `THIRD_PARTY_ATTRIBUTION.md`.

## Reproduction

From this directory:

```bash
python dinov2_multibudget/reproduce_table.py
python full_data_ordering/reproduce_statistics.py
python full_data_ordering/construct_blocked_order.py --verify
```

These commands reproduce the released tables and verify the released DDPM
ordering assignment without downloading data or retaining images. They are not
an end-to-end training rerun.
