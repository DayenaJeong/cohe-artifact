# Release scope and terms

## Author-created code

HUMAN_DECISION_REQUIRED: the existing repository does not identify an intended author-code license. No MIT, Apache, or other grant is inferred. A repository LICENSE can be added after the authors choose one. Code availability and permission to reuse are distinct.

## Author-trained checkpoints

Local candidate assets are tensor-only exports of provenance-verified paper endpoints. No pretrained initialization is redistributed within those exports. Their distribution license and applicable training-data terms still require an author decision. Pending files must not be uploaded simply because provenance and privacy checks pass.

## Derived numerical data

Previously intentionally released CIFAR-100 arrays, sample-index manifests, aggregate results, and seed-level statistics are retained with provenance. No broader license covering all upstream assets is asserted. ImageNet-1K scalar arrays remain excluded pending resolution of their release terms.

## Upstream assets

Raw CIFAR/ImageNet/Imagenette images, dataset archives, pretrained weights, model caches, private logs, and credentials are excluded. Obtain upstream assets from their maintainers and comply with their terms. ImageNet's published access conditions restrict database use to non-commercial research/education: https://image-net.org/download.php . This note does not infer a checkpoint distribution license from those conditions. CIFAR source: https://www.cs.toronto.edu/~kriz/cifar.html .

Model code licenses do not alone establish dataset/checkpoint redistribution rights. Historical OpenCLIP/auxiliary target weight provenance and the ImageGPT metadata discrepancy remain unresolved; see `provenance/UPSTREAM_ASSETS.csv`.
