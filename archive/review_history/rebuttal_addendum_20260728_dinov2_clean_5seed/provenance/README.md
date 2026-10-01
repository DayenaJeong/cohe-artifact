# Provenance

The clean Ordinary 10% table combines newly rerun raw seeds 0–2 with the previously completed raw extension seeds 3–4. All five seed-level Top-1 values and paired deltas in the final table are raw-result records; the old rounded seed 0–2 Random records are preserved for audit history but are not inputs to the final aggregate.

Seeds 0–2 use the recovered runner, score CSV, configuration, selector procedure, 200 epochs, evaluation at epoch 20, batch size 128, and the preflight-verified paired initial-state hashes. Existing seeds 3–4 are copied read-only from the completed extension output. Their source manifests did not persist initial-state byte hashes, so this addendum does not claim hashes for those historical extension runs.

The release contains derived accuracies, paired statistics, selector indices, and hashes only. It does not contain images, datasets, weights, checkpoints, or raw training logs, and it does not claim full end-to-end regeneration from the anonymous artifact.
