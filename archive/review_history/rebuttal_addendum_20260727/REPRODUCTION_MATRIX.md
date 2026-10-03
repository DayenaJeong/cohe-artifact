# Reproduction and release matrix

| Family | Array | Manifest | Count | Dtype | Direction | Alignment |
|---|---|---|---:|---|---|---|
| DDPM | `cifar100_proxy_arrays/arrays/ddpm_scores.npy` | `cifar100_proxy_arrays/manifests/ddpm_manifest.csv.gz` | 50000 | float32 | higher predicted-noise norm = harder | PASS: canonical index and 100 x 500 classes |
| DDIM | `cifar100_proxy_arrays/arrays/ddim_scores.npy` | `cifar100_proxy_arrays/manifests/ddim_manifest.csv.gz` | 50000 | float32 | higher noise/reconstruction MSE = harder | PASS: canonical index and 100 x 500 classes |
| ConvAE | `cifar100_proxy_arrays/arrays/convae_scores.npy` | `cifar100_proxy_arrays/manifests/convae_manifest.csv.gz` | 50000 | float32 | higher reconstruction MSE = harder | PASS: canonical index and 100 x 500 classes |
| SD-VAE | `cifar100_proxy_arrays/arrays/sdvae_scores.npy` | `cifar100_proxy_arrays/manifests/sdvae_manifest.csv.gz` | 50000 | float32 | higher latent round-trip MSE = harder | PASS: canonical index and 100 x 500 classes |
| SD-v1.5 | `cifar100_proxy_arrays/arrays/sd15_scores.npy` | `cifar100_proxy_arrays/manifests/sd15_manifest.csv.gz` | 50000 | float32 | higher latent denoising MSE = harder | PASS: canonical index and 100 x 500 classes |
| Emu3 VQ | `cifar100_proxy_arrays/arrays/emu3_vq_scores.npy` | `cifar100_proxy_arrays/manifests/emu3_vq_manifest.csv.gz` | 50000 | float32 | higher reconstruction MSE = harder | PASS: canonical index and 100 x 500 classes |
| DINOv2 | `cifar100_proxy_arrays/arrays/dinov2_scores.npy` | `cifar100_proxy_arrays/manifests/dinov2_manifest.csv.gz` | 50000 | float32 | higher isolation score = harder/more isolated | PASS: canonical index and 100 x 500 classes |

## Local checks performed

- Every released proxy array is one-dimensional with 50,000 finite values.
- Every released manifest has 50,000 rows, unique integer sample IDs, and the canonical `sample_id == global_index` convention.
- CIFAR-100 class alignment was independently checked as 100 classes with 500 training samples each.
- The six existing scalar-table columns (DDIM, ConvAE, SD-VAE, SD-v1.5, Emu3 VQ, DINOv2) match the audited wide scalar table after the documented float32 representation; DDPM is the separately audited Figure-4-linked array.
- The ConvAE source was float64; only its float32 cast is released, with the source checksum retained in metadata.
- No source paths, filenames, images, reconstructions, latent tensors, feature tensors, weights, checkpoints, or raw logs are copied.

## Reproduction commands

```bash
python dinov2_multibudget/reproduce_table.py
python full_data_ordering/reproduce_statistics.py
python full_data_ordering/construct_blocked_order.py --verify
```

The scripts reproduce released statistics and ordering membership only; they do not claim a full training rerun from the anonymous artifact.
