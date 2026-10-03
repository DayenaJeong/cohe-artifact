# Reproducibility

The CPU commands in this repository verify released scalar interfaces and recompute statistics from completed experiment records. They do not retrain models. GPU scoring and training require user-provided datasets, upstream pretrained models, experiment-specific inputs, and suitable compute.

```bash
python scripts/reproduce.py --experiment dependence
python scripts/reproduce.py --experiment predictive_validity
python scripts/reproduce.py --experiment dinov2_gate3
python scripts/reproduce.py --experiment ddpm_ordering
python scripts/reproduce.py --experiment imagenet1k_vae_gate3
python scripts/reproduce.py --experiment paper_results
```

The smoke test uses toy inputs and checks the public interfaces:

```bash
bash scripts/smoke_test.sh
```

## Experiment coverage

| Experiment | Released resources | Reproduction scope |
| --- | --- | --- |
| Gate 1 dependence | Audit interfaces, seven CIFAR-100 scalar arrays, selected targets, and summaries | CPU verification is available; full scoring requires upstream inputs and some original settings remain unresolved |
| Gate 2 predictive validity | Linear interface, recovered nonlinear suite, and five-split records | The public command is a univariate interface check; the full multi-proxy analysis requires inputs that are not redistributed |
| CIFAR-100 main Gate 3 | Training runners, 120 seed-level rows, and 120 checkpoints | Training requires CIFAR-100 and user-provided score/order inputs |
| DINOv2 Gate 3 | Selection runner, rank adapters, scalar array, and all four five-seed result conditions | No trained endpoint checkpoints were retained |
| DDPM ordering | Full-data ordering runner, scalar array, all eight paired result records, and 18 checkpoints | Random seeds 0–2 and Easy-to-Hard seeds 0–2 checkpoints were not retained |
| ImageNet-10 | Fine-tuning/scoring runners and seed-level results | Upstream Imagenette and score inputs are required; endpoint checkpoints were not retained |
| ImageNet-1K VAE Gate 3 | Scoring, selection, training and verification code; five paired runs; 20 checkpoints | ImageNet data, VAE weights, and the full training score array are not redistributed |
| Positive control | Task-aligned retrieval code and available split/null summaries | Upstream ImageNet and target inputs are required |
| Shuffled negative control | Scalar inputs and a five-seed summary | The exact historical sample-wise shuffled-DDIM runner was not recovered |
| Cross-architecture control | Aggregate evidence | The exact historical runner and complete regeneration inputs were not recovered |
| Timestep and training-dynamics diagnostics | Analysis code and available summaries/curves | Historical timestep-sweep source and some target provenance remain incomplete |

## Resource map

- [`REPRODUCTION_MATRIX.md`](REPRODUCTION_MATRIX.md) lists expected CPU outputs and GPU entry points.
- [`../provenance/release_inventory.csv`](../provenance/release_inventory.csv) maps experiments to code, results, inputs, checkpoints, and limitations.
- [`../results/SOURCE_OF_TRUTH.csv`](../results/SOURCE_OF_TRUTH.csv) indexes reported result files.
- [`PAPER_TABLE_FIGURE_MAP.csv`](PAPER_TABLE_FIGURE_MAP.csv) maps paper tables and figures to repository resources.
- [`../checkpoints/README.md`](../checkpoints/README.md) describes checkpoint coverage and loading.

Some recovered scripts are close implementation sources rather than exact historical runners. Missing code, inputs, and checkpoints were not reconstructed from reported numbers. Full end-to-end reproduction is therefore experiment-dependent.
