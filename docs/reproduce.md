# Original Experiment Workflow

This document maps the released scripts to the original FastGainRouter experiments. It is a reference for understanding the implementation, not a complete clean-environment reproduction guide. See the [README](../README.md) for the method and final results.

## Environment and directory assumptions

The scripts were written for an existing project tree rooted at `/data/donggyun/SIDL_project`, with NAFNet under `baseline/NAFNet` and experiment files under `my_experiment/skip_restore_v2_tiny`.

The public repository exposes the latter experiment's scripts and configurations at its root. Cloning it does not recreate the original server tree. Some shell scripts also reference the original Conda installation and environment.

Dependencies include PyTorch, the NAFNet/BasicSR implementation, OpenCV, scikit-image, scikit-learn, LPIPS, pandas, matplotlib, and joblib. [requirements.txt](../requirements.txt) lists packages but does not lock the original environment or install a particular NAFNet source revision.

## Workflow map

The following groups describe script roles; they are not a standalone sequence of commands.

| Stage | Released scripts | External prerequisites or notes |
| --- | --- | --- |
| Tiny configuration and training | `make_tiny_train_configs_v1.py`, `train_tiny_with_early_stop_v1.py`, `run_tiny_train_v1.sh` | Prepared inner splits and the original NAFNet training environment |
| Tiny/Base official-validation inference | `make_tiny_base_val_configs_v1.py`, `run_tiny_base_val_inference_v1.sh` | Trained Tiny and Base checkpoints |
| Validation metrics | `build_full_val_tiny_base_table_v1.py`, `add_lpips_full_val_tiny_base_v1.py` | Paired inputs, references, and branch outputs |
| Train-inner gain targets | `make_traininner_tiny_base_configs_v1.py`, `run_traininner_tiny_base_inference_v1.sh`, `build_traininner_tiny_base_gain_table_v1.py` | Prepared train-inner data and branch checkpoints |
| Training features | `extract_traininner_fast_features_v1.py` | Train-inner gain table and input images |
| Gain-router fitting and threshold calibration | `train_fast_gain_router_traininner_v1.py` | Train/tuning/evaluation tables, feature files, and latency measurements |
| Latency correction | `recompute_traininner_router_latency_corrected_v1.py` | Assumes the original summary did not already include feature-extraction latency |
| Final summaries | `finalize_compact_main_results_v1.py`, `finalize_results_traininner_v1.py` | Corrected summary and branch-audit summary |
| Qualitative figures | `make_final_qualitative_v1.py`, `make_final_paper_figures_v3_with_arrows_v2.py` | Original per-image outputs, policy selections, and selected figure assets |
| Additional branch audit | `make_high_val_configs_v1.py`, `run_high_val_inference_v1.sh`, `build_candidate_audit_high_v1.py` | Separate High checkpoint; not a branch of the final Input/Tiny/Base router |

All script names above refer to [scripts/](../scripts/).

## Release boundaries

The release does not include the complete original project tree. In particular, it does not supply a complete workflow for constructing the case-level inner splits, training Base, or producing all tuning/evaluation inputs expected by the router script.

Router inputs include `esval_tiny_base_policy_table_v1_lpips.csv`, `esval_fast_features_v1.csv`, and `full_val_fast_features_v1.csv`. These files are not included, and their complete preparation workflow is not supplied.

Model artifacts and bulk branch outputs are excluded. Inspecting the released source and aggregate results does not require them; rerunning the experiments does.

## Latency accounting

The final reported latency is feature extraction + batch router prediction + selected branch inference, not pure model-forward latency.

The router training script has fallback latency values when measurement files are absent. Such fallback values must not be treated as experimental measurements. The feature-latency correction script assumes feature cost was initially zero; applying that correction to a summary that already includes feature cost would double-count it.

For final published numbers, consult the [compact result table](../results/final_compact_main_table_v1.md).
