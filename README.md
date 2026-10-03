# COHE: Auditing Non-Transitivity in Sample Difficulty Proxies for Vision Models

**Dayena Jeong and Sunglok Choi** · Accepted at **NeurIPS 2026, Evaluations & Datasets Track**.

COHE audits a specified proxy, discriminative target, and downstream protocol through dependence (Gate 1), held-out predictive validity (Gate 2), and operational transfer (Gate 3). It reports bounded claims rather than a universal pass/fail score.

This is the official artifact for the accepted NeurIPS 2026 paper. The fixed camera-ready source and checkpoint release is [`v1.0-neurips2026`](https://github.com/DayenaJeong/cohe-artifact/releases/tag/v1.0-neurips2026).

`v1.0-neurips2026` is the fixed scientific release snapshot; the current `main` branch includes documentation-only post-release corrections, with no changes to scientific code, reported results, or checkpoint binaries.

The repository includes the COHE audit code, recovered experiment implementations, seven CIFAR-100 scalar arrays, aggregate and seed-level results, and author-trained checkpoints for retained endpoints. Reproduction coverage varies by experiment because raw datasets, some historical inputs, and third-party pretrained weights are not redistributed.

## Installation

Use Python 3.10 or newer for the public CPU interface:

```bash
python -m pip install -r requirements-cpu.txt
bash scripts/smoke_test.sh
```

Training and scoring dependencies are separate. The recorded ImageNet environment is in `environment-imagenet.yml`; see [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md). Historical versions are documented only where recovered from logs.

## Quick start

```bash
python scripts/reproduce.py --experiment dependence
python scripts/reproduce.py --experiment predictive_validity
python scripts/reproduce.py --experiment dinov2_gate3
python scripts/reproduce.py --experiment ddpm_ordering
python scripts/reproduce.py --experiment imagenet1k_vae_gate3
python scripts/reproduce.py --experiment paper_results
```

The first command computes a scalar DDIM/CE dependence check from released values. The second is a univariate linear interface check on released DINOv2/first-learning values, **not** a reproduction of every paper nonlinear model. The Gate 3 commands recompute paired statistics from completed experiment records; they do not retrain models. `paper_results` checks all four DINOv2 conditions, all eight ordering pairs, and both ImageNet-1K endpoints.

Toy examples in `examples/` exercise interfaces and are not experimental evidence. The claim-card script is a reporting aid; the six reporting tiers and their precedence are described in [`docs/CLAIM_TIER_GUIDE.md`](docs/CLAIM_TIER_GUIDE.md).

## Reproduction and experiments

[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) summarizes experiment-level coverage, required upstream inputs, and known gaps. [`docs/REPRODUCTION_MATRIX.md`](docs/REPRODUCTION_MATRIX.md) lists expected outputs and GPU entry points. The machine-readable experiment inventory is in [`provenance/release_inventory.csv`](provenance/release_inventory.csv), and [`results/SOURCE_OF_TRUTH.csv`](results/SOURCE_OF_TRUTH.csv) indexes the reported result files.

CPU commands verify released scalars and completed-result statistics. Full scoring or training requires the appropriate datasets, upstream models, inputs, environment, and compute.

## Data and checkpoints

Obtain CIFAR-10/100, Imagenette, and ImageNet-1K from their maintainers and follow their terms. No raw images or upstream archives are redistributed. CIFAR source order is the original torchvision CIFAR-100 training order. The seven released 50,000-entry arrays are DDPM, DDIM, SD-VAE, SD-v1.5, Emu3 VQ, ConvAE, and DINOv2 under `derived_scalar_arrays/cifar100/`; each has an index manifest.

Not all eight primary generative families are released as scalar arrays. ImageNet-1K scalar arrays are intentionally excluded. Target coverage is incomplete for some family/target combinations. Gate 2 covers scalar or low-dimensional inputs; it does not test full underlying representation information.

The [GitHub Release](https://github.com/DayenaJeong/cohe-artifact/releases/tag/v1.0-neurips2026) provides 234 author-trained checkpoints in nine archives. No checkpoint binary is committed to ordinary Git. See [`checkpoints/README.md`](checkpoints/README.md) for coverage, missing endpoints, manifests, and loading instructions.

Pretrained model identifiers, recovered revisions, unresolved provenance, and terms pointers are in [`provenance/UPSTREAM_ASSETS.csv`](provenance/UPSTREAM_ASSETS.csv). Users obtain these weights directly from their owners.

## Limitations

Code recovery is partial for historical diagnostics, and some exact historical checkpoint tags remain unresolved. Available training code still needs upstream inputs, GPU compute, and experiment-specific environment validation. DINOv2 selection, the ResNet-50 second learner, and the ImageNet-10 fine-tuning runs have no retained endpoint weights. DDPM ordering includes 18 of 24 expected endpoints. Full regeneration is not claimed for every reported block.

The standalone shuffled-DDIM negative-control runner and cross-architecture runner are incomplete historical recoveries. Some timestep/calibration provenance remains unresolved, and the full predictive suite requires inputs not redistributed here. Completed numerical records remain available independently of checkpoint or code coverage.

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

Author-created source code is released under the [MIT License](LICENSE). That license does not apply to or relicense third-party datasets, models, or pretrained weights. See [`LICENSE_OR_TERMS.md`](LICENSE_OR_TERMS.md) and the upstream provenance notes for scope.
