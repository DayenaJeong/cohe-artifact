# Reproducibility scope

See `REPRODUCTION_MATRIX.md`, `PUBLIC_RELEASE_INVENTORY.csv`, and `results/SOURCE_OF_TRUTH.csv` for exact final-result coverage. The CPU smoke test uses toy inputs and checks retained historical summary interfaces; it is not proof of paper-number regeneration. `python scripts/reproduce.py --experiment paper_results` verifies the final DINOv2, ordering, and ImageNet paired results directly from released seed-level records.

Scoring and training sources recovered from both research projects are provided where traceable. Some historical scripts are implementation candidates rather than verified exact runners; their provenance status is explicit in the inventory. No missing implementation or model file is reconstructed from reported numbers. Full retraining/scoring requires upstream inputs and compute, and was not run during release preparation.
