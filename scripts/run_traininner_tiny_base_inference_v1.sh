#!/bin/bash
set -e

source /home/donggyun/miniconda3/etc/profile.d/conda.sh
conda activate sidl

ROOT=/data/donggyun/SIDL_project
NAFNET_DIR=$ROOT/baseline/NAFNet
CONFIG_DIR=$ROOT/my_experiment/skip_restore_v2_tiny/configs/traininner_full

cd "$NAFNET_DIR"
export PYTHONPATH="$NAFNET_DIR:$PYTHONPATH"

for branch in tiny base; do
  OPT="$CONFIG_DIR/${branch}_traininner_full_v1.yml"

  echo ""
  echo "============================================================"
  echo "[INFO] train_inner full inference branch=$branch"
  echo "============================================================"

  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=4461 \
    basicsr/test.py \
    -opt "$OPT" \
    --launcher pytorch
done

echo "[DONE] train_inner full tiny/base inference"
