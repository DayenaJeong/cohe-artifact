# Paper ↔ public release consistency

The final public release is approved. Author-created source code uses the MIT License, and public distribution of the 234 verified author-trained checkpoints is approved. Manuscript availability claims must be updated only after the repository and GitHub Release are confirmed live.

## Artifact and reproducibility

CURRENT PAPER WORDING

```latex
The accompanying public artifact includes CIFAR-100 DDPM, DDIM, SD-VAE, SD-v1.5, Emu3 VQ, ConvAE, and DINOv2 scalar arrays with sample-index mappings; it does not include all eight primary families or ImageNet-1K score arrays. Code for the gate analyses, aggregate and seed-level results, and statistical verifiers support tabular verification. Full regeneration requires upstream datasets/models and GPU scoring or retraining; Appendix~\ref{sec:appendix-artifact-quickstart} gives the public URL and an explicit coverage matrix.
```

ACTUAL PUBLIC RELEASE STATUS

Recovered training/scoring code and 234 verified checkpoint endpoints are approved for the public release.

REQUIRED REVISION

Add recovered-code and checkpoint scope after the source and assets are confirmed publicly accessible.

FINAL WORDING

The public artifact provides gate-analysis code, recovered experiment-specific training/scoring implementations, seven CIFAR-100 proxy scalar arrays, final numerical records and CPU verifiers. Selected author-trained endpoints are supplied where preserved and approved; full regeneration still requires upstream inputs and compute, and historical coverage remains partial.

## Appendix J: Reproducibility

CURRENT PAPER WORDING

```latex
The accompanying public artifact supports tabular verification and reuse of the COHE analyses. It provides seven CIFAR-100 proxy arrays (DDPM, DDIM, SD-VAE, SD-v1.5, Emu3 VQ, ConvAE, DINOv2), sample-index mappings, selected CE/first-learning targets and split IDs, precomputed aggregate results, and DINOv2, ordering, and ImageNet-1K seed-level results. It does not redistribute all eight primary families, ImageNet-1K score arrays, raw datasets/images, pretrained weights, or trained checkpoints. Full regeneration of proxy scores and retraining-based transfer experiments requires the corresponding datasets, pretrained models, and GPU compute. The optional DINOv2 retraining code requires user-supplied upstream inputs.
```

ACTUAL PUBLIC RELEASE STATUS

234 technically validated and publicly released endpoints in nine archives; DINOv2/ResNet-50/ImageNet-10 and six ordering endpoints unavailable.

REQUIRED REVISION

Narrow the blanket trained-checkpoint exclusion to the documented missing endpoints; retain all upstream-data/weight exclusions.

FINAL WORDING

The artifact supplies selected author-trained checkpoints for main CIFAR selection, balanced controls, alpha-mix, ConvAE, 18 of 24 ordering endpoints, and all 20 ImageNet-1K best/final endpoints. DINOv2-selection, ResNet-50 second-learner and ImageNet-10 endpoints were not retained. Raw images, upstream weights and ImageNet scalar arrays are not redistributed.

## Appendix J.1: Artifact scope

CURRENT PAPER WORDING

```latex
Table~\ref{tab:appendix-reproduction-coverage} distinguishes verification using provided scalar values and aggregate results from full regeneration. DINOv2 hard-selection comparisons use five paired seeds per condition; the separate three-seed reverse-isolation diagnostic is reported independently. The accompanying artifact is publicly available at \url{https://github.com/DayenaJeong/cohe-artifact} and provides the resources summarized in Tables~\ref{tab:appendix-artifact-scope}--\ref{tab:appendix-reproduction-coverage}.
```

ACTUAL PUBLIC RELEASE STATUS

The canonical repository and v1.0-neurips2026 release provide nine checkpoint archives and SHA-256 metadata.

REQUIRED REVISION

Link the fixed release assets and checksums; state partial coverage explicitly.

FINAL WORDING

The canonical artifact repository provides a reproduction matrix and final-result source index. Approved checkpoint assets and their SHA-256 manifest are linked from its release page. Verification from provided results is distinct from full regeneration with upstream assets.

## Table 41: scoring/retraining row

CURRENT PAPER WORDING

```latex
Full scoring/retraining pipelines & No & Requires upstream models, datasets, and GPU scoring or retraining under the stated protocols. \\
```

ACTUAL PUBLIC RELEASE STATUS

Recovered code is present for major training/scoring blocks; exact negative/cross-architecture/timestep/calibration runners remain incomplete.

REQUIRED REVISION

Replace blanket No with Partial, experiment-specific; never change it to unconditional full-pipeline availability.

FINAL WORDING

Recovered scoring/retraining code | Partial | Main CIFAR, DINOv2, ordering, ImageNet-10/1K and available controls; upstream datasets/models and GPU compute required, with historical gaps documented.

## Table 42: VAE coverage

CURRENT PAPER WORDING

```latex
VAE & ImageNet-10; ImageNet-1K & No & Yes$^*$ & Definition; ImageNet-1K train/checkpoint protocol and hashes & ImageNet-10 summaries; ImageNet-1K seeds/protocol/verifier \\
```

ACTUAL PUBLIC RELEASE STATUS

ImageNet-10/1K code recovered; ImageNet-1K 20 best/final candidates; ImageNet-10 endpoints unavailable; scalar arrays excluded.

REQUIRED REVISION

Add code and selected endpoint coverage after their respective publication while retaining aggregate-only score coverage.

FINAL WORDING

VAE: ImageNet-10/1K scoring and training code, final seed results and ImageNet-1K best/final checkpoints are provided; raw data and VAE scores require user-supplied upstream inputs; ImageNet-10 endpoint weights are unavailable.

## Table 42: DINOv2 coverage

CURRENT PAPER WORDING

```latex
DINOv2 & CIFAR-100 train & Yes & No & Definition, sample-index mappings; optional retraining code requiring upstream inputs & Five-seed results; four conditions; CPU verifier \\
```

ACTUAL PUBLIC RELEASE STATUS

Four five-seed conditions and scalar/rank adapters provided; no saved trained learner checkpoints.

REQUIRED REVISION

Describe recovered selection code and explicitly missing learner endpoints.

FINAL WORDING

DINOv2: scalar arrays, sample-index mappings, recovered selection code and all four five-seed result conditions are provided; upstream CIFAR and baseline inputs are required; trained learner endpoints were not retained.

## Checklist Q4 justification

CURRENT PAPER WORDING

```latex
The paper discloses the audit statistics and final seed-level evidence; the artifact supplies scalar checks, aggregate verification, and CPU statistical verifiers. The expanded DINOv2, DDPM ordering, and ImageNet-1K protocols are specified in Appendix~\ref{sec:appendix-operational}. Full regeneration requires upstream assets; unresolved historical checkpoint provenance is identified in Appendix~\ref{sec:appendix-asset-notes}.
```

ACTUAL PUBLIC RELEASE STATUS

Released recovered code and selected verified checkpoints do not establish complete end-to-end reproducibility.

REQUIRED REVISION

Use answer Yes and update the justification without claiming every experiment is fully regenerated.

FINAL WORDING

The paper specifies the main audit protocols; the artifact supplies scalar checks, final numerical records, recovered experiment code and approved selected endpoints. Full regeneration requires upstream assets, and unresolved historical provenance and missing checkpoints are explicitly documented.

## Checklist Q5 justification

CURRENT PAPER WORDING

```latex
The accompanying public artifact provides gate-analysis code, seven CIFAR-100 proxy arrays with sample-index mappings, precomputed aggregate results, operational seed-level results, and verification utilities. Appendix Tables~\ref{tab:appendix-artifact-scope}--\ref{tab:appendix-reproduction-coverage} document exact coverage and redistribution boundaries; Appendix~\ref{sec:appendix-artifact-quickstart} gives the public URL. Full regeneration of some proxy scores and retraining experiments still requires the corresponding upstream datasets, models, and compute.
```

ACTUAL PUBLIC RELEASE STATUS

The public source and checkpoint release is fixed at v1.0-neurips2026 with the documented asset inventory.

REQUIRED REVISION

Keep the public-access answer Yes and update coverage text to the resources verified in the live release.

FINAL WORDING

The public artifact provides gate-analysis and recovered experiment code, seven CIFAR proxy arrays, final results, CPU verifiers and approved selected author-trained checkpoint assets. Its coverage matrix identifies missing inputs/checkpoints and upstream access requirements.

## Appendix evidence-scope artifact row

CURRENT PAPER WORDING

```latex
\textbf{Artifact/checklist} & Gate-analysis code; illustrative examples; seven CIFAR-100 proxy arrays (Table~\ref{tab:appendix-reproduction-coverage}); aggregate and seed-level results. & Verification of the corresponding reported analyses. & No raw datasets, images, weights, checkpoints, or full scoring/retraining pipeline. \\
```

ACTUAL PUBLIC RELEASE STATUS

Blanket no-checkpoint and no-pipeline wording is obsolete because selected checkpoints and recovered experiment code are released.

REQUIRED REVISION

Retain no raw datasets/images/upstream weights; distinguish selected trained checkpoints and partial code.

FINAL WORDING

No raw datasets/images or upstream weights; selected author-trained endpoint assets and recovered experiment code are provided with explicit coverage and historical omissions.

## Appendix asset-license table: ResNet row

CURRENT PAPER WORDING

```latex
ResNet-18/50 & Pretrained targets; subset training; triage & No checkpoints & \href{https://github.com/pytorch/vision/blob/main/LICENSE}{torchvision BSD-3-Clause code}; \href{https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE}{timm Apache-2.0 code}; weight provenance is setting-specific \\
```

ACTUAL PUBLIC RELEASE STATUS

ResNet-18 author-trained candidates exist; no ResNet-50 trained endpoints found; upstream pretrained target weights excluded.

REQUIRED REVISION

Distinguish released author-trained ResNet-18 assets from unavailable ResNet-50 endpoints; do not infer upstream weight rights.

FINAL WORDING

Selected author-trained ResNet-18 endpoints provided; ResNet-50 trained endpoints and upstream pretrained target weights not redistributed; upstream terms remain setting-specific.

The canonical public checkpoint count is 234 in nine archives, not 243. Availability statements must match the verified live release; no scientific number changes. The missing standalone timestep-sweep data source, shuffled negative-control runner, cross-architecture runner, and calibration provenance remain explicit. PAPER_UPDATE_REPORT_READY=TRUE.
