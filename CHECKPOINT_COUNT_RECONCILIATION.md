# Canonical checkpoint count reconciliation

The counts refer to different subsets of the same 308 original checkpoint-like files. They are consistent when provenance and final-endpoint selection are kept separate. The nine author-family exclusions are unbalanced 30% conditions whose endpoints differ from the authoritative results.

| Canonical category | Count | Definition |
| --- | --- | --- |
| Original checkpoint-like files found | 308 | Rows in the local checkpoint inventory and `checkpoints/MANIFEST.csv`; excludes newly exported tensor copies and archives. |
| Author-trained COHE-family provenance | 243 | 204 initially endpoint-review candidates plus 39 provenance-verified ConvAE/ordering/ImageNet files. Broader than final endpoint coverage. |
| Technically validated final endpoints | 234 | Passed final result correspondence, exact tensor-export equality, hashes, metadata cleaning, and strict CPU architecture loading; public distribution is approved. |
| Author-family files excluded from final endpoint set | 9 | Older unbalanced 30% controls, three methods × seeds 0–2; endpoint accuracies differ from authoritative final results. |
| Other unmatched historical files | 63 | Tensor files with no established mapping to final COHE results; not included in the 243. |
| Unresolved restricted-loader/provenance files | 2 | Not included in the 243; not eligible for packaging. |
| Total excluded candidates | 74 | 9 + 63 + 2. |
| SHA-identical original-file duplicates | 0 | All 308 original SHA-256 values are distinct. |
| SHA-identical packaged-file duplicates | 0 | Rechecked in archive verification; each checkpoint is assigned to one archive. |
| Auxiliary trained models included | 1 | ConvAE, already included within the 234, not an additional count. |
| Upstream pretrained weights packaged | 0 | Upstream assets are pointers only. |
| Human-approved checkpoint distributions | 234 | Public distribution approved for all technically validated final endpoints. |

**308 = 234 + 9 + 63 + 2; 243 = 234 + 9; 74 = 9 + 63 + 2.**

## Exact nine-file difference

All nine originate in `ICML/rebuttal_2026/ttal_full_response/results/w3_balanced_control/raw/models/`. The directory name caused a broad balanced-control family classification, but each JSON sidecar explicitly says mode=unbalanced and budget=0.3. Their filenames contain `unbalanced_Random`, `unbalanced_Gen-Hard` or `unbalanced_Disc-Hard`. These are historical reruns of conditions also represented by final main results, not nine extra reported balanced conditions.

| Method | Budget | Seed | Historical accuracy | Final authoritative accuracy |
| --- | --- | --- | --- | --- |
| Disc-Hard | 30% | 0 | 34.42 | 34.26 |
| Disc-Hard | 30% | 1 | 33.29 | 34.68 |
| Disc-Hard | 30% | 2 | 34.32 | 34.34 |
| Gen-Hard | 30% | 0 | 44.25 | 44.13 |
| Gen-Hard | 30% | 1 | 43.23 | 43.73 |
| Gen-Hard | 30% | 2 | 43.09 | 44.12 |
| Random | 30% | 0 | 44.19 | 44.27 |
| Random | 30% | 1 | 43.71 | 43.23 |
| Random | 30% | 2 | 43.42 | 43.49 |


The historical numerical differences were not repaired. Original checkpoints, results and locked configurations remain unchanged. Exact relative source identifiers and the two accuracy columns are retained in `checkpoints/NONAUTHORITATIVE_NINE.csv`.

## Canonical release breakdown

120 main CIFAR endpoints + 30 balanced endpoints + 45 alpha-mix endpoints + 18 ordering endpoints + 20 ImageNet best/final endpoints + one ConvAE = **234**. Main selection's 120 endpoints are split into four budget-specific archives to respect the GitHub per-asset limit, without splitting or duplicating an individual checkpoint.

`MANIFEST.csv` is a discovery inventory (308), `VERIFIED_RELEASE_CANDIDATES.csv` is the technical-candidate set (234), `PUBLIC_RELEASE_INVENTORY.csv` is an experiment/settings inventory (43 rows), and `RELEASE_FILE_MANIFEST.csv` is a source-tree file inventory. The latter two are not checkpoint counts. The local `SELECTION_AND_REJECTION.csv` is a broader 3,229-file provenance inventory; checkpoint selection is recorded in the checkpoint manifests. New tensor-only exports (234) represent the same models as original bundles and are not additional trained models.

Use these named counts in README, manifests, release notes, and audit reports: **DISCOVERED=308; AUTHOR_FAMILY=243; TECHNICALLY_VERIFIED_FINAL_ENDPOINTS=234; EXCLUDED=74; HUMAN_APPROVED=234**. The canonical public checkpoint count is **234**, not 243 or 308. Nine release archives contain those 234 models.
