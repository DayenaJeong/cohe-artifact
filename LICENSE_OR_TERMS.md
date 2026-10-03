# Release scope and terms

## Author-created code

Author-created source code in this repository is released under the MIT License (SPDX identifier: `MIT`). See the root `LICENSE` file.

## Author-trained checkpoints

The GitHub Release distributes 234 provenance-verified, author-trained checkpoints whose public distribution has been approved. The archives contain tensor-only exports; third-party pretrained weights are not bundled. See `checkpoints/README.md`, `checkpoints/VERIFIED_RELEASE_CANDIDATES.csv`, and the release notes for exact coverage and verification details.

## Derived numerical data

Released CIFAR-100 arrays, sample-index manifests, aggregate results, and seed-level statistics are retained with provenance. ImageNet-1K scalar arrays are not redistributed. Nothing in the repository license grants rights to upstream datasets, models, or weights.

## Upstream assets

Raw CIFAR-10/CIFAR-100, ImageNet, and Imagenette images or archives; third-party pretrained weights; model caches; private logs; and credentials are excluded. Obtain upstream assets from their maintainers and comply with their respective licenses, model cards, and access terms. ImageNet access information is at https://image-net.org/download.php and CIFAR is at https://www.cs.toronto.edu/~kriz/cifar.html .

The repository MIT License does not relicense CIFAR, ImageNet, Imagenette, `google/ddpm-cifar10-32`, `facebook/dinov2-small`, `stabilityai/sd-vae-ft-mse`, `runwayml/stable-diffusion-v1-5`, OpenCLIP, `BAAI/Emu3-VisionTokenizer`, `openai/imagegpt-small`, or any other third-party dataset, model, or pretrained weight. Historical OpenCLIP/auxiliary target provenance and the ImageGPT metadata discrepancy remain documented in `provenance/UPSTREAM_ASSETS.csv`.
