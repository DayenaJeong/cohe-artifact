# FAQ

## Why is the artifact lightweight?

COHE is an evaluation and reporting protocol. The artifact provides reusable audit scripts, schemas, templates, toy examples, selected derived scalar arrays, and aggregate table summaries without redistributing datasets, images, checkpoints, or full scoring/retraining pipelines.

## Why are raw datasets or proxy arrays not redistributed?

The paper uses public datasets and pretrained model families whose licenses and access rules remain upstream. Raw images, upstream datasets, weights, checkpoints, and full retraining pipelines are therefore excluded. The artifact includes selected derived scalar arrays where available plus aggregate summaries and reusable code.

## Why is DINOv2 not counted among the eight primary generative proxy families?

DINOv2 isolation is a non-generative representation-level boundary stress test. It checks whether COHE can surface target-specific structure beyond generation-objective proxies, but it is not part of the eight primary generative proxy families.

## Why is GradNorm not treated as operational-transfer evidence?

GradNorm is a sensitivity target, not the accuracy-selection endpoint used in the CIFAR-100 transfer audit. A GradNorm-specific transfer claim would need its own robustness, pruning, or sensitivity-oriented downstream protocol.

## What does the R2 <= 0.005 band mean?

It is a descriptive near-zero visualization band for this audit. It is comparable to split variability and shuffled-proxy null behavior in the reported experiments, but it is not a universal COHE decision threshold.

## Can users apply COHE without Stage 3?

Yes. Users can report Stage 1 dependence and Stage 2 held-out prediction as alignment or predictive-validity evidence.

## What should users report when only Stage 1--2 are feasible?

Report the proxy definition, target definition, sample matching, Pearson/Spearman statistics, density or running-median diagnostics when available, held-out R2, split variability, and the limited claim tier. Do not describe the result as operational transfer or subset-selection evidence until a downstream action protocol is evaluated.
