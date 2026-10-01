# COHE: Auditing Non-Transitivity in Sample Difficulty Proxies for Vision Models

**Dayena Jeong and Sunglok Choi** · Accepted at **NeurIPS 2026, Evaluations & Datasets Track**.

COHE audits a specified proxy, discriminative target, and downstream protocol through dependence (Gate 1), held-out predictive validity (Gate 2), and operational transfer (Gate 3). It reports bounded claims rather than a universal pass/fail score.

Canonical repository: https://github.com/DayenaJeong/cohe-artifact

## Release scope and status

This branch prepares the public camera-ready source and checkpoint release. It has not been pushed, tagged, or published as a GitHub Release. Code licensing and checkpoint distribution terms still require author decisions; see `LICENSE_OR_TERMS.md`. Checkpoint manifests and local prepared-asset checks do not mean that checkpoint downloads are currently available.

Included resources comprise lightweight audit interfaces, recovered experiment runners, seven previously released CIFAR-100 proxy arrays with sample-index manifests, selected targets/split IDs, final seed-level and aggregate results, provenance mappings, and CPU statistical verifiers. The exact coverage and gaps are recorded in `PUBLIC_RELEASE_INVENTORY.csv` and `docs/REPRODUCTION_MATRIX.md`.

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

`checkpoints/MANIFEST.csv` inventories every checkpoint candidate found in both research projects, including excluded historical candidates. Verified author-trained endpoints have local tensor-only exports with original and export checksums. See `checkpoints/README.md` and `docs/RELEASE_PROPOSAL.md`. No checkpoint binary is committed to ordinary Git, and no assets are published yet.

## Paper result mapping

`PUBLIC_RELEASE_INVENTORY.csv` maps experiments to recovered code, configs, final results, checkpoints, seeds, and limitations. `docs/PAPER_TABLE_FIGURE_MAP.csv` maps the current paper's table and figure labels. `results/SOURCE_OF_TRUTH.csv` identifies authoritative final result locations. `archive/review_history/` preserves earlier records; it must not override final results.

## Known limitations

Code recovery is partial for historical diagnostics, and some exact historical checkpoint tags remain unresolved. Available training code still needs upstream inputs, GPU compute, and experiment-specific environment validation. Not every trained checkpoint was saved: DINOv2 selection, the ResNet-50 second learner, and the ImageNet-10 fine-tuning runs have no matching saved endpoint weights in the audited directories. Ordering contains only part of the final checkpoint set. Full regeneration is not claimed for every reported block.

## Licenses and citation

Author-created code: **HUMAN_DECISION_REQUIRED**; no license is invented. Checkpoint distribution terms require a separate decision. See `LICENSE_OR_TERMS.md` and upstream provenance notes.

Citation metadata is provided in `CITATION.cff`. No DOI or proceedings URL is asserted before those metadata are available.
