# Public reproduction matrix

Run commands from the repository root. CPU commands recompute completed-result statistics; training commands are separate and were checked only for parsing/import/configuration, without launching training.

| Paper result | Verified CPU command | Expected output | Regeneration boundary |
|---|---|---|---|
| Four DINOv2 hard-selection conditions; Table 5/24 | `python scripts/reproduce.py --experiment dinov2_gate3` | -14.87/-17.25/-11.12/-11.40 pp; all 20 deltas negative | CIFAR, DINO scalar ranks, matched Random baseline and GPU retraining required |
| DDPM ordering; Table 26 | `python scripts/reproduce.py --experiment ddpm_ordering` | Easy–Random -0.77; Shuffled–Random -0.5025; Easy–Shuffled -0.2675 pp; eight pairs | CIFAR + released DDPM scalar; all 50k once/epoch; 200 epochs; 10 bins |
| ImageNet-1K best/final; Table 27 | `python scripts/reproduce.py --experiment paper_results` | Best Hard 53.6464, Random 61.4116, delta -7.7652, CI [-8.1704339,-7.3599661]; epoch90 delta -7.8536 | ImageNet train/validation + VAE score manifest; 30% per class; 90 epochs; five pairs |
| Scalar DDIM/CE interface | `python scripts/reproduce.py --experiment dependence` | Writes Pearson/Spearman summary from included values | Does not regenerate upstream scores |
| Scalar DINOv2/first-learning interface | `python scripts/reproduce.py --experiment predictive_validity` | Writes single linear held-out interface result | Not the full paper nonlinear/multi-proxy fit |
| Final predictive table | Provided `results/aggregate/predictive_validity/` summary/per-split records | CE multi-proxy R² .0002; margin .0034; first-learning .0459 (rounded) | Original manifest includes DDIM-CLIP, whose array is not released; full rerun requires missing input |
| CIFAR ResNet-18/50 and ImageNet-10 selection | `results/seed_level/` provides final rows and summaries | Exact reported seeds/budgets | Recovered training runners require user-obtained datasets/scalars; no training was rerun |
| Other diagnostics | `PUBLIC_RELEASE_INVENTORY.csv` and current table map | Available summaries/source candidates listed individually | Historical exact runner/target checkpoint gaps remain explicit |

## GPU commands — explicit user action only

Main CIFAR transfer (the recorded runner uses final epoch 200):

```bash
python experiments/cifar100/alpha_mix_baseline.py --gen derived_scalar_arrays/cifar100/arrays/ddpm_scores.npy --disc user_inputs/cifar100_ce.npy --data_root user_inputs/cifar100 --out_dir outputs/cifar100 --seeds 0,1,2,3,4 --budgets 0.1,0.3,0.5,0.7 --alphas '' --epochs 200
```

Separate uncertainty/coreset and balanced-control runners in the same folder need their declared score/order inputs. The second-learner runner is `run_cifar100_second_learner_transfer.py`; use its `--help` for explicit parameters. These code paths do not automatically supply missing score inputs or historical timm versions.

Ordering:

```bash
COHE_CIFAR_ROOT=user_inputs/cifar100 python experiments/ddpm_ordering/run_curriculum_with_weights.py --policy EASY_TO_HARD --seed 0 --out-dir outputs/ordering/easy_seed0 --device cuda
```

Repeat declared policies RANDOM, EASY_TO_HARD, SHUFFLED_PROXY_ORDER for seeds 0–7. The released scalar default is DDPM; other policies do not establish another paper result.

ImageNet (supply the exact selection CSV whose original checksum is recorded in the run inventory):

```bash
COHE_IMAGENET_TRAIN_ROOT=user_inputs/imagenet/train COHE_IMAGENET_VAL_ROOT=user_inputs/imagenet/val python experiments/imagenet1k_vae_gate3/train_resnet18_gate3_20260727.py --selection user_inputs/proxy_hard_30_seed_independent.csv --out-dir outputs/imagenet/hard_seed0 --policy proxy_hard --seed 0 --epochs 90
```

Repeat proxy_hard/random for seeds 0–4 with the recorded matched selection inputs. Best-validation Top-1 is primary; final epoch 90 is sensitivity. Score and selection builders are included, but score manifests/Images are user supplied. This documentation does not claim a completed fresh end-to-end reproduction.
