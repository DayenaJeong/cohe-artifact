# Asset Provenance

The paper uses public benchmark datasets and pretrained model families for evaluation. This artifact does not redistribute upstream datasets, raw images, pretrained checkpoints, or model weights.

| Asset family | Role in audit | Redistributed here | User responsibility |
|---|---|---|---|
| CIFAR-10/100 | Main dependence, prediction, and subset-selection audits | No | Follow upstream dataset terms |
| ImageNet-1K / ImageNet-10 | Large-scale validation scoring and limited ImageNet-family operational checks | No | Follow upstream dataset access rules |
| DDPM / DDIM checkpoints | Denoising, score-norm, and reconstruction proxies | No | Follow upstream model terms |
| Stable Diffusion VAE / SD-v1.5 | Latent round-trip and latent-denoising proxies | No | Follow upstream model-card terms |
| CLIP | Semantic reconstruction-discrepancy proxy component | No | Follow upstream license/model-card terms |
| Emu3 | VQ-tokenizer reconstruction diagnostic family | No | Follow upstream license/model-card terms |
| ImageGPT | Autoregressive token-likelihood diagnostic family | No | Follow upstream license/model-card terms |
| ResNet classifiers | Discriminative targets and subset-training models | No | Reproduce from standard training or upstream checkpoints |
