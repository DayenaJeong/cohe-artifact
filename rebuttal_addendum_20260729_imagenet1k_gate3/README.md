# COHE ImageNet-1K Gate 3 rebuttal addendum

This anonymized addendum reports one completed operational-transfer study:
VAE reconstruction-error class-balanced Proxy-Hard versus class-balanced
Random at a 30% retained-data budget, using a ResNet-18 trained from scratch
on ImageNet-1K and evaluated on the full 50,000-image validation split over
five paired seeds.

The included tables contain derived scalar results only.  This package does
not redistribute ImageNet images or labels, the upstream VAE checkpoint,
classifier checkpoints, raw local logs, credentials, private paths, or the
full train score array.  Full regeneration requires authorized ImageNet
access and the upstream model under its applicable terms.

The primary result is defined by the best validation Top-1 checkpoint.  The
epoch-90 result is included only as a sensitivity check.  The scope is
limited to this proxy, budget, learner, dataset, and hard-selection policy;
it does not establish universal proxy failure or transfer to other tasks,
architectures, budgets, or interventions.

Run `python3 reproduction/verify_statistics.py` for a CPU-only verification
of the included seed-level statistics.
