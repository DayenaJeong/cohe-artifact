# Excluded files and reasons

- ImageNet-1K scalar arrays and manifests: excluded because the applicable
  scalar-only redistribution terms were not sufficiently clear to claim safe
  release.
- Raw CIFAR-100 or ImageNet images, image bytes, reconstructions, latent tensors,
  feature tensors, model weights, checkpoints, and raw training logs: excluded.
- Source manifests and logs containing absolute local paths: replaced with
  sanitized relative metadata and integer-index manifests.
- The original ConvAE source array was float64; the released ConvAE array is a
  documented float32 cast and the source checksum is recorded in metadata.
- DDIM-CLIP is not among the seven proxy families released in this addendum.
