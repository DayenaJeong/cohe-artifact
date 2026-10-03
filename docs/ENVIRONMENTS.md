# Environments

The CPU interface was tested with the local Python 3.13 environment; packages in `requirements-cpu.txt` are intentionally not claimed as historical pins. Public entry points require Python 3.10+. GPU training environments are experiment-specific.

The ten completed ImageNet-1K runs record Python 3.10.19, torch 2.5.1+cu124, torchvision 0.20.1+cu124, and CUDA 12.4 in every per-run environment file. `environment-imagenet.yml` pins only those recovered versions; other library versions are not guessed. Install the CUDA wheel variants following the official PyTorch package index instructions.

Earlier DINOv2/ordering audit metadata records Python 3.8.20, torch 2.0.0+cu118, torchvision 0.15.0, NumPy 1.24.3, and SciPy 1.10.1. This historical record is not a claim that the new public entry-point environment recreates GPU trajectories. DINOv2 uses the recovered torchvision fallback where timm was unavailable, with pretrained=False/weights=None. CIFAR second-learner and older main-transfer runners use timm; their exact historical timm version was not recovered.

`requirements-train.txt` and `requirements-scoring.txt` list imports recovered from staged code, with unpinned versions when unknown. Emu3 scoring additionally needs its upstream implementation/model access. Do not treat installing these dependency lists as a completed end-to-end rerun validation.
