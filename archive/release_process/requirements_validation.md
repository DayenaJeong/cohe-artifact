# Requirements validation

| Dependency | Consumer / validation rationale |
| --- | --- |
| numpy | Array operations throughout analysis/scoring/training; no historical pin invented. |
| scipy | Dependence, paired statistics and diagnostic correlations. |
| pandas | Aggregate/result readers and predictive analysis. |
| scikit-learn | Held-out prediction, nonlinear predictors and controls. |
| matplotlib | CPU predictive/dynamics analysis plots; retained for these public analysis modules. |
| torch | Training/scoring and restricted checkpoint load; CPU import tested. |
| torchvision | Datasets, transforms and model architectures. |
| timm | Main CIFAR and second-learner architectures; exact older version remains unknown. |
| tqdm | Scoring/training progress consumers. |
| Pillow | Image loaders and diagnostic image outputs. |
| diffusers | DDIM, SD and VAE scorer imports. |
| transformers | DINOv2/Emu3/ImageGPT and CLIP fallback imports. |
| accelerate | Diffusers/Transformers pretrained loading support; dependency runtime import tested. |
| open_clip_torch | open_clip imported by DDIM semantic scorer and CLIP fallback. |


`requirements.txt` delegates to `requirements-cpu.txt`; train delegates to CPU; scoring delegates to train. PyYAML was removed from the CPU list because no staged runtime Python module imports it. Metadata can be inspected with a user's YAML tooling; the external local audit used an already-installed YAML parser without changing project dependencies. Matplotlib remains because public CPU analysis code imports it.

There is no root `environment.yml`; `environment-imagenet.yml` is the preserved experiment environment. Its verified Python 3.10.19, torch 2.5.1+cu124 and torchvision 0.20.1+cu124 pins are unchanged. No unrecorded historical version has been guessed. CPU packages were import-tested in the existing CPU environment, and all training/scoring dependencies were import-tested in the recorded ImageNet-capable environment with CUDA_VISIBLE_DEVICES empty. Safe guarded-source import and CLI tests passed; no dataset load or expensive training was launched.

The installation command was checked for requirements-file resolution, dependency import availability and pip CLI syntax. No package installation, environment creation or fresh dependency solver run was performed. Tests establish current local interface compatibility, not a new clean-room historical environment reconstruction.
