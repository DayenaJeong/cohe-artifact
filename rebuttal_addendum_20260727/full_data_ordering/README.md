# DDPM full-data ordering and shuffled-proxy control

This directory contains the eight-seed full-data CIFAR-100 ordering comparison
(Easy, Random, and Shuffled), including the per-seed shuffled-proxy control.
It also contains a privacy-safe ten-bin membership manifest derived from the
released DDPM scalar array. Only integer positions and ranks are released.

`reproduce_statistics.py` recomputes the paired statistics from the sanitized
per-seed table. `construct_blocked_order.py --verify` reconstructs the locked
ten-bin ordering from `../cifar100_proxy_arrays/arrays/ddpm_scores.npy` and
checks the released membership manifest. No images, filenames, paths, weights,
or checkpoints are required.
