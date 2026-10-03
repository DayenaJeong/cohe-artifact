# COHE: Auditing Non-Transitivity in Sample Difficulty Proxies for Vision Models

**Dayena Jeong and Sunglok Choi** · Accepted at **NeurIPS 2026, Evaluations & Datasets Track**.

COHE audits a specified proxy, discriminative target, and downstream protocol through dependence (Gate 1), held-out predictive validity (Gate 2), and operational transfer (Gate 3). It reports bounded claims rather than a universal pass/fail score.

Canonical repository: https://github.com/DayenaJeong/cohe-artifact

## Release scope and status

This is the official public artifact for the accepted NeurIPS 2026 paper. The fixed camera-ready source and checkpoint release is [`v1.0-neurips2026`](https://github.com/DayenaJeong/cohe-artifact/releases/tag/v1.0-neurips2026). Author-created source code is MIT-licensed; third-party datasets, models, and pretrained weights remain governed by their upstream terms.

`v1.0-neurips2026` is the fixed scientific release snapshot; the current `main` branch includes documentation-only post-release corrections, with no changes to scientific code, reported results, or checkpoint binaries.

Included resources comprise lightweight audit interfaces, recovered experiment runners, seven previously released CIFAR-100 proxy arrays with sample-index manifests, selected targets/split IDs, final seed-level and aggregate results, provenance mappings, and CPU statistical verifiers. The exact coverage and gaps are recorded in `PUBLIC_RELEASE_INVENTORY.csv` and `docs/REPRODUCTION_MATRIX.md`.

Available in the source tree are audit code, recovered experiment code, derived scalar arrays, seed-level and aggregate results, configurations/manifests, and provenance documentation.

The GitHub Release provides 234 verified author-trained checkpoints in nine archives. Raw CIFAR/ImageNet data, upstream pretrained weights, third-party assets, and unavailable historical checkpoints are not redistributed.

## Reproducibility coverage

| Experiment | Code | Data/scalars | Seed-level results | Checkpoints | Reproduction status |
| --- | --- | --- | --- | --- | --- |
| Gate 1 dependence | Audit interfaces and recovered scoring code; some original settings unresolved | Seven CIFAR arrays; selected targets | Provided summaries and available split records | Upstream scoring weights not redistributed | Verification only; full scoring requires upstream inputs |
| Gate 2 prediction | Linear interface and recovered nonlinear suite | Some inputs supplied; DDIM-CLIP and other targets missing | Five-split records provided | Predictors not packaged | Verification only; full suite needs missing inputs |
| CIFAR main Gate 3 | Recovered training runners | DDPM supplied; raw CIFAR and CE/order inputs required | 120 final rows | 120 released | Partial; upstream data required |
| DINOv2 Gate 3 | Recovered selection runner and rank adapters | DINOv2 scalar array supplied; CIFAR and baseline inputs required | All four five-seed conditions | No preserved endpoint weights | Partial; upstream data required |
| DDPM ordering | Recovered full-data runner | DDPM supplied; raw CIFAR required | All eight paired seeds | 18 of 24 endpoints released | Partial; upstream data required |
| ImageNet-10 | Recovered fine-tuning and scoring runners | Upstream Imagenette and score inputs required | Three paired seeds at two budgets | No preserved endpoint weights | Partial; upstream data required |
| ImageNet-1K Gate 3 | VAE scoring, selection and 90-epoch training code | Upstream ImageNet and score/selection inputs required | Five pairs, best/final endpoints | 20 released | Partial; upstream data required |
| Positive control | Recovered task-aligned retrieval code | Upstream ImageNet/target inputs required | Available split/null summaries | Upstream target weights not redistributed | Partial; upstream data required |
| Shuffled negative control | Exact sample-wise shuffled-DDIM runner not recovered | Only relevant scalar inputs and summaries supplied | Five-seed summary | No endpoint applicable | Historical code incomplete |
| Cross-architecture control | Exact historical runner not recovered | Aggregate evidence only | No complete sample-level regeneration inputs | Historical target weights unavailable | Historical code incomplete |
| Timestep / dynamics diagnostics | Dynamics/epoch analysis code; exact DDPM sweep and calibration provenance partial | Some summaries/curves supplied; full historical targets missing | Available split records only | Diagnostic target endpoints unavailable | Verification only; historical code incomplete |

"Full from released resources" applies to the documented CPU interface/statistical commands, not to every original experiment. Balanced controls add 30 released checkpoints, alpha-mix adds 45, and ConvAE adds one; see `checkpoints/README.md`.

## Installation

Use Python 3.10 or newer for the public CPU interface:

```bash
python -m pip install -r requirements-cpu.txt
bash scripts/smoke_test.sh
```

Training/scoring dependencies are separate. The recorded ImageNet environment is in `environment-imagenet.yml`; see `docs/ENVIRONMENTS.md`. Historical versions are documented only where recovered from logs.

## Quickstart and the three gates

```bash
python scripts/reproduce.py --experiment dependence
python scripts/reproduce.py --experiment predictive_validity
python scripts/reproduce.py --experiment dinov2_gate3
python scripts/reproduce.py --experiment ddpm_ordering
python scripts/reproduce.py --experiment imagenet1k_vae_gate3
python scripts/reproduce.py --experiment paper_results
```

The first command computes a scalar DDIM/CE dependence check from released values. The second is a univariate linear interface check on released DINOv2/first-learning values, **not** a reproduction of every paper nonlinear model. The Gate 3 commands recompute paired statistics from completed experiment records; they do not retrain models. `paper_results` checks all four final DINOv2 conditions, all eight ordering pairs, and both ImageNet-1K endpoints. Expected results and GPU rerun commands are in `docs/REPRODUCTION_MATRIX.md`.

Toy examples in `examples/` exercise interfaces and are not experimental evidence. The claim-card generator produces a draft requiring human review; the six final reporting tiers and precedence are documented in `docs/CLAIM_TIER_GUIDE.md`.

## Datasets and scalar data

Obtain CIFAR-10/100, Imagenette, and ImageNet-1K from their maintainers and follow their terms. No raw images or upstream archives are redistributed. CIFAR source order is the original torchvision CIFAR-100 training order. The seven released 50,000-entry arrays are DDPM, DDIM, SD-VAE, SD-v1.5, Emu3 VQ, ConvAE, and DINOv2 under `derived_scalar_arrays/cifar100/`; each has an index manifest.

Not all eight primary generative families are released as scalar arrays. ImageNet-1K scalar arrays are intentionally excluded. Target coverage is incomplete for some family/target combinations. Gate 2 covers scalar or low-dimensional inputs; it does not test full underlying representation information.

## Upstream pretrained models

Pretrained model identifiers, recovered revisions, unresolved provenance, and terms pointers are in `provenance/UPSTREAM_ASSETS.csv`. Users obtain these weights directly from their owners. A code license does not grant rights to upstream datasets/models.

## Checkpoints

`checkpoints/MANIFEST.csv` inventories every checkpoint candidate found in both research projects, including excluded historical candidates. Verified author-trained endpoints have tensor-only exports with original and export checksums. See `checkpoints/README.md` and `docs/RELEASE_RECORD.md`. No checkpoint binary is committed to ordinary Git; the nine archives are hosted only as GitHub Release assets.

Canonical counts: 308 original checkpoint-like files; 243 with author-trained COHE-family provenance; 234 technically validated final endpoints released; 74 excluded. The nine-file difference consists of older unbalanced 30% controls with different final accuracies. See `CHECKPOINT_COUNT_RECONCILIATION.md`, `checkpoints/VERIFIED_RELEASE_CANDIDATES.csv`, `checkpoints/RELEASE_ARCHIVES_MANIFEST.csv`, and `RELEASE_APPROVAL_CHECKLIST.md`.

## Paper result mapping

`PUBLIC_RELEASE_INVENTORY.csv` maps experiments to recovered code, configs, final results, checkpoints, seeds, and limitations. `docs/PAPER_TABLE_FIGURE_MAP.csv` maps the current paper's table and figure labels. `results/SOURCE_OF_TRUTH.csv` identifies authoritative final result locations. `archive/review_history/` preserves earlier records; it must not override final results.

## Known limitations

Code recovery is partial for historical diagnostics, and some exact historical checkpoint tags remain unresolved. Available training code still needs upstream inputs, GPU compute, and experiment-specific environment validation. Not every trained checkpoint was saved: DINOv2 selection, the ResNet-50 second learner, and the ImageNet-10 fine-tuning runs have no matching saved endpoint weights in the audited directories. Ordering contains only part of the final checkpoint set. Full regeneration is not claimed for every reported block.

The standalone shuffled-DDIM negative-control runner and cross-architecture runner are incomplete historical recoveries. Some timestep/calibration provenance remains unresolved, and the full predictive suite requires inputs not redistributed here. Completed numerical records remain available independently of checkpoint/code coverage.

## Licenses and citation

Author-created source code is released under the [MIT License](LICENSE). That license does not apply to or relicense third-party datasets, models, or pretrained weights. Public distribution of the 234 author-trained checkpoints has been approved; see `LICENSE_OR_TERMS.md` and the upstream provenance notes for scope.

Citation metadata is provided in `CITATION.cff`. No DOI or proceedings URL is asserted before those metadata are available.
