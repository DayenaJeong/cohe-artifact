# Final result source audit

There are 27 unique result-scope mappings and zero duplicate authoritative scope keys. Every mapped authority is a specific existing file, not a directory. DINOv2's ordinary 10/30/50% and balanced 10% conditions share one all-condition authority; DDPM has one eight-seed authority; ImageNet's best/final endpoints share one seed-level authority. Main CIFAR, second learner, balanced controls and alpha-mix each have a single explicit result authority.

Different result facets (such as balanced accuracy versus diagnostics or dynamics correlation versus prediction) have distinct scope keys. Their summaries and per-split records are derived evidence rather than competing authorities. Review-history copies remain subordinate unless explicitly named as the sole preserved authority for a historical-only diagnostic. No numerical result file was changed.

The exact standalone timestep-sweep numerical source was not recovered. Its final plotted evidence remains in the paper; no substitute source is designated here. That gap is a historical-coverage limitation, not permission to treat a different scorer as authoritative. Some calibration and cross-architecture provenance also remains incomplete. A unique mapping establishes which supplied record to use, not full experimental regeneration.

The prior five-row index did not explicitly cover second learner, balanced or alpha-mix; these scopes have now been added. Historical ordinary-10 DINOv2 rounded/mixed records must not override the all-condition five-seed authority.
