# Reproduction Guide

[TODO] Adjust paths if your project root differs from `/data/donggyun/SIDL_project`.

## 0. Environment

Create a Python environment with PyTorch, BasicSR/NAFNet dependencies, OpenCV, scikit-image, scikit-learn, LPIPS, pandas, tqdm, matplotlib, and joblib.

See `requirements.txt`.

## 1. Prepare SIDL dataset

Expected external dataset location:

```text
/data/donggyun/SIDL_project/data_full/
├── train/
├── val/
└── test/
```

The raw dataset is not included in this repository.

## 2. Train Tiny model

```bash
python scripts/make_tiny_train_configs_v1.py
bash scripts/run_tiny_train_v1.sh
```

## 3. Run Tiny/Base validation inference

```bash
python scripts/make_tiny_base_val_configs_v1.py
bash scripts/run_tiny_base_val_inference_v1.sh
```

## 4. Build validation tables and LPIPS

```bash
python scripts/build_full_val_tiny_base_table_v1.py
python scripts/add_lpips_full_val_tiny_base_v1.py
```

## 5. Candidate High audit

```bash
python scripts/make_high_val_configs_v1.py
bash scripts/run_high_val_inference_v1.sh
python scripts/build_candidate_audit_high_v1.py
```

## 6. Train-inner full router pipeline

```bash
python scripts/make_traininner_tiny_base_configs_v1.py
bash scripts/run_traininner_tiny_base_inference_v1.sh
python scripts/build_traininner_tiny_base_gain_table_v1.py
python scripts/extract_traininner_fast_features_v1.py
python scripts/train_fast_gain_router_traininner_v1.py
python scripts/recompute_traininner_router_latency_corrected_v1.py
```

## 7. Final result tables and figures

```bash
python scripts/finalize_compact_main_results_v1.py
python scripts/finalize_results_traininner_v1.py
```

## 8. Qualitative examples

```bash
python scripts/make_final_qualitative_v1.py
```

[TODO] Curate final qualitative examples manually before including them in the report.
