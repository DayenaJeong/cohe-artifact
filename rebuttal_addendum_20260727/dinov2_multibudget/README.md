# DINOv2 multi-budget materials

This directory releases selection indices and sanitized aggregate/per-seed
results for DINOv2 at 10%, 30%, and 50% budgets. It includes both the ordinary
and class-balanced comparisons. The released indices are integer positions only;
they do not contain filenames, paths, images, or model outputs.

The source results used the documented DINOv2 isolation score and a ResNet-18
from-scratch transfer protocol. The 10% ordinary result is an independently
audited saved aggregate from the original analysis; it is marked as such in the
configuration rather than presented as a newly rerun experiment.

Run `python reproduce_table.py` to recompute the paired deltas, confidence
intervals, and paired tests from the sanitized CSV.
