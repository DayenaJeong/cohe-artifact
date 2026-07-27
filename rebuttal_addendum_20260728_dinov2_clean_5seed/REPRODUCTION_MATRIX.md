# Clean DINOv2 reproduction matrix

| Condition | Seeds | Learner | Budget | Final metric | Seed-level table | Selection |
|---|---:|---|---:|---|---|---|
| Ordinary 10% | 0–4 | ResNet-18 from scratch | 0.10 | epoch-200 CIFAR-100 Top-1 | statistics/ordinary10_clean_5seed_seed_level.csv | selectors/ |
| Ordinary 30% | 0–4 | ResNet-18 from scratch | 0.30 | epoch-200 CIFAR-100 Top-1 | statistics/dinov2_clean_5seed_all_conditions.csv | preserved in earlier addendum |
| Ordinary 50% | 0–4 | ResNet-18 from scratch | 0.50 | epoch-200 CIFAR-100 Top-1 | statistics/dinov2_clean_5seed_all_conditions.csv | preserved in earlier addendum |
| Class-balanced 10% | 0–4 | ResNet-18 from scratch | 0.10 | epoch-200 CIFAR-100 Top-1 | statistics/dinov2_clean_5seed_all_conditions.csv | preserved in earlier addendum |

The clean Ordinary 10% condition uses raw seed 0–2 reruns plus raw seed 3–4 extension outputs. Historical rounded records are not used. The other three conditions are read-only imports from the previously verified five-seed table; they were not rerun.

The included CPU-only reproduction checks table arithmetic and the paired statistics. Full score generation and model retraining still require upstream CIFAR-100 data, proxy score CSV, runner dependencies, and the configuration documented by the hash inventory.
