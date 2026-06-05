#!/bin/bash
set -e

source /home/donggyun/miniconda3/etc/profile.d/conda.sh
conda activate sidl

ROOT=/data/donggyun/SIDL_project
cd "$ROOT"

export PYTHONPATH=$ROOT/baseline/NAFNet:$PYTHONPATH

echo "============================================================"
echo "[INFO] Start Tiny NAFNet training"
echo "[INFO] Time: $(date)"
echo "============================================================"

echo "[INFO] Data check"
echo "train_inner input:  $(find -L data_full/train/all_custom_train_inner/input -type f | wc -l)"
echo "train_inner target: $(find -L data_full/train/all_custom_train_inner/target -type f | wc -l)"
echo "es_val input:       $(find -L data_full/train/all_custom_es_val_inner/input -type f | wc -l)"
echo "es_val target:      $(find -L data_full/train/all_custom_es_val_inner/target -type f | wc -l)"

echo "[INFO] GPU status before training"
nvidia-smi

python "$ROOT/my_experiment/skip_restore_v2_tiny/scripts/train_tiny_with_early_stop_v1.py" \
  --name tiny_lr1e3 \
  --config "$ROOT/my_experiment/skip_restore_v2_tiny/configs/train/tiny_lr1e3.yml" \
  --exp_name NAFNet-SIDL-all-tiny-width4-lr1e3-es20000-v1 \
  --max_bad 5 \
  --min_delta 0.01 \
  --port 4411

echo "[INFO] tiny_lr1e3 finished at $(date)"
nvidia-smi

python "$ROOT/my_experiment/skip_restore_v2_tiny/scripts/train_tiny_with_early_stop_v1.py" \
  --name tiny_lr5e4 \
  --config "$ROOT/my_experiment/skip_restore_v2_tiny/configs/train/tiny_lr5e4.yml" \
  --exp_name NAFNet-SIDL-all-tiny-width4-lr5e4-es20000-v1 \
  --max_bad 5 \
  --min_delta 0.01 \
  --port 4412

echo "[INFO] tiny_lr5e4 finished at $(date)"

echo "[INFO] Best checkpoints:"
find "$ROOT/baseline/NAFNet/experiments" \
  -path "*NAFNet-SIDL-all-tiny-width4-*-es20000-v1*" \
  -name "best_net_g.pth" \
  -print -ls

echo "[INFO] Final GPU status"
nvidia-smi
