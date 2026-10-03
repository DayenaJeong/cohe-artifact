# Reproduction matrix

| component | included | note |
|---|---|---|
| seed-level Top-1/Top-5 and deltas | yes | `SEED_LEVEL_RESULTS.csv` |
| best/final statistics | yes | `FINAL_STATISTICS.json` |
| CPU statistical verification | yes | `reproduction/verify_statistics.py` |
| protocol and checkpoint rule | yes | `CHECKPOINT_POLICY_AUDIT.md` |
| score alignment | hashes only | `SCORE_ALIGNMENT.md` |
| author-trained ResNet-18 checkpoints | yes | 20 best/final endpoints in the `v1.0-neurips2026` GitHub Release |
| upstream VAE weights | no | obtain `stabilityai/sd-vae-ft-mse` from upstream under its terms |
| ImageNet images/labels | no | authorized access required |
| full train score array | no | not redistributed |
| raw logs/private paths | no | intentionally excluded |
