# COHE clean five-seed DINOv2 rebuttal addendum

This is a clearly separated post-submission addendum. It supersedes the mixed provenance Ordinary 10% table for rebuttal reporting without deleting or rewriting the earlier rebuttal_addendum_20260727 materials.

Included are the clean Ordinary 10% seed-level results, the four-condition five-seed table, derived paired statistics, ten selector-index arrays, selector hashes, recovered runner/configuration hashes, sanitized command inventory, and provenance notes.

The clean Ordinary 10% mean DINO-Hard minus Random Top-1 delta is -14.87 pp (95% paired CI [-15.62, -14.12], paired p = 6.3726e-07); all five paired differences are negative. Across all four DINOv2 conditions there are 20 paired comparisons and all 20 are negative.

Historical rounded seed 0–2 Random records remain preserved in the older addendum and are explicitly excluded from the clean aggregate. No images, datasets, model weights, checkpoints, or raw training logs are redistributed. The included selectors, scalar results, and statistics are derived values only.

Run python reproduction/reproduce_clean_statistics.py to verify the clean Ordinary 10% statistics from the included CSV.
