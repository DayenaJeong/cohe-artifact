# Release command validation

All tests used CPU-only execution with GPU visibility disabled. Logs and detailed argv are kept in the local audit, outside publication files. No dataset download, package install or training ran.

| Documented command / check | Result | Scope / expected output |
| --- | --- | --- |
| `pip_help` | PASS | pip syntax/dependency-file/import validation; installation deliberately not executed |
| `cpu_smoke` | PASS | Executed; outputs/smoke_test/, outputs/cached_summary_inventory.md and outputs/rendered_cached_tables.md verified |
| `reproduce_help` | PASS | --help / imports / argument parsing only; no scoring or training |
| `dependence` | PASS | Executed; outputs/ddim_ce_dependence.csv |
| `predictive_validity` | PASS | Executed; outputs/dinov2_linear_interface_prediction.csv |
| `dinov2_gate3` | PASS | Executed final-record statistics; stdout result verification; no retraining |
| `ddpm_ordering` | PASS | Executed final-record statistics; stdout result verification; no retraining |
| `imagenet1k_vae_gate3` | PASS | Executed final-record statistics; stdout result verification; no retraining |
| `paper_results` | PASS | Executed final-record statistics; stdout result verification; no retraining |
| `help_run_training_dynamics` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_analyze_epochwise_dynamics` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_audit_driver` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_hsic_test` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_run_predictive_power_fast` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_run_curriculum_with_weights` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_construct_blocked_order` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_run_dinov2_gate3_transfer` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_run_dinov2_stress_test` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_positive_task_aligned_control` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_score_imagenette_train_targets` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_run_imagenette_transfer_minimal` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_compute_ddim_hardness` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_compute_imagenet1k_disc_metrics` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_compute_disc_hardness` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_compute_emu3_vq_recon_proxy` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_compute_imagegpt_nll_proxy` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_compute_sd_latent_denoise_proxy` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_compute_cifar100_latent_roundtrip_proxy` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_compute_imagenet1k_vae_proxies` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_clip_semantic_proxy` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_run_w3_coreset_baseline` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_alpha_mix_baseline` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_run_cifar100_second_learner_transfer` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_run_cifar100_dataset_specific_ae_proxy` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_run_w3_uncertainty_baseline` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_run_w3_balanced_control_job` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_train_resnet18_timing_pilot_20260727` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_train_resnet18_gate3_20260727` | PASS | --help / imports / argument parsing only; no scoring or training |
| `help_score_train_vae_full_20260727` | PASS | --help / imports / argument parsing only; no scoring or training |
| `documented_args_alpha_mix_baseline` | PASS | Full documented training argv parsed under fail-fast guard immediately after parse_args; datasets and training never reached |
| `documented_args_run_curriculum_with_weights` | PASS | Full documented training argv parsed under fail-fast guard immediately after parse_args; datasets and training never reached |
| `documented_args_train_resnet18_gate3_20260727` | PASS | Full documented training argv parsed under fail-fast guard immediately after parse_args; datasets and training never reached |
| `dependencies_import` | PASS | --help / imports / argument parsing only; no scoring or training |
| `cpu_dependencies_import` | PASS | --help / imports / argument parsing only; no scoring or training |
| `guarded_source_imports` | PASS | --help / imports / argument parsing only; no scoring or training |

45 checks passed; 33 guarded source modules imported. All claimed README Python entry points have functioning help and their CPU commands executed. GPU/data-dependent commands have only parser/import/configuration validation. The first validation harness omitted normal script-directory import resolution for one ImageNet command; its failing attempt is preserved, the harness was corrected and only that check was repeated. No manuscript/code algorithm change resulted.
