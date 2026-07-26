# CIFAR-100 scalar proxy arrays

Each array has shape `(50000,)`, dtype `float32`, and uses the canonical
CIFAR-100 training order (`sample_id == global_index`). Each corresponding
manifest has 50,000 rows and contains only integer sample/class indices.

| Proxy | Released array | Score direction | Scalar-only scope |
|---|---|---|---|
| DDPM | `arrays/ddpm_scores.npy` | higher predicted-noise L2 is harder | yes |
| DDIM | `arrays/ddim_scores.npy` | higher reconstruction/noise MSE is harder | yes |
| ConvAE | `arrays/convae_scores.npy` | higher pixel reconstruction MSE is harder | yes |
| SD-VAE | `arrays/sdvae_scores.npy` | higher latent round-trip MSE is harder | yes |
| SD-v1.5 | `arrays/sd15_scores.npy` | higher latent denoising MSE is harder | yes |
| Emu3 VQ | `arrays/emu3_vq_scores.npy` | higher reconstruction MSE is harder | yes |
| DINOv2 | `arrays/dinov2_scores.npy` | higher k-nearest-neighbor isolation is harder | yes |

Exact definitions, model identifiers, preprocessing, source checksums, and
alignment checks are recorded in `metadata/`. Source paths and checkpoints are
intentionally not copied into this addendum.
