# ImageNet-1K Gate 3 raw validation report

Overall validation: **PASS**

| check | result |
|---|---|
| train sample count | PASS — 1,281,167 |
| validation sample count | PASS — 50,000 |
| class count and validation integrity | PASS — 1,000 classes; 0 broken validation files |
| score array/manifest alignment | PASS — float32 shape (1,281,167,), finite, contiguous global indices |
| VAE proxy identity | PASS — `stabilityai/sd-vae-ft-mse` reconstruction error |
| selection budget and quotas | PASS — 30%; 384,300 samples per policy; matched class quotas |
| seed coverage | PASS — seeds 0–4, both policies |
| run completion | PASS — 10/10; each 90 epochs and 135,180 optimizer steps |
| duplicate/partial/failed runs | PASS — none found |
| policy configuration equivalence | PASS — only policy/seed/selector fields differ |
| pretrained weights | PASS — `weights=None` in every config |
| train/validation separation | PASS — selections reference train paths; validation is separate |

Key source hashes:

- score array: `ab343f5915b8932773e5c200c2fdcefdef95afc24a6fef7e86e89a4278385954`
- score manifest: `fd4daee452b18fb9afc355b677ebe49bfa056a33f39ffb9aa4cde628015a5b18`
- proxy config: `576ea7225c3bf9b5121d13f5080364269f304290f9a118bf1fe410c56cd6ed51`
- runner: `090f33ec50a3a4e97e06c8bf0bd387da7ba14cd18b8f36650dc7f1bc9fb31a39`
- launch pipeline: `6c8e54967749557b3f9c0e1b44658d0d31beed68eff07510b2e8f656593f48e3`

The source-of-truth file is `imagenet1k_gate3_source_of_truth.json`.  The
old blocked files are preserved under `audit/superseded_blocked_20260726/`.
