#!/bin/bash
set -e

source /home/donggyun/miniconda3/etc/profile.d/conda.sh
conda activate sidl

ROOT=/data/donggyun/SIDL_project
NAFNET_DIR=$ROOT/baseline/NAFNet
CONFIG_DIR=$ROOT/my_experiment/skip_restore_v2_tiny/configs/high_eval

cd "$NAFNET_DIR"
export PYTHONPATH="$NAFNET_DIR:$PYTHONPATH"

TASKS=("clean" "dust" "finger" "mixed" "scratch" "water")
DIFFS=("easy" "medium" "hard")

echo "============================================================"
echo "[INFO] Start High official val inference"
echo "[INFO] Time: $(date)"
echo "============================================================"

for task in "${TASKS[@]}"; do
  for diff in "${DIFFS[@]}"; do
    OPT="$CONFIG_DIR/high_${task}_${diff}_val.yml"

    if [ ! -f "$OPT" ]; then
      echo "[WARN] missing config: $OPT"
      continue
    fi

    echo ""
    echo "========== High inference: task=$task diff=$diff =========="
    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc_per_node=1 \
      --master_port=4451 \
      basicsr/test.py \
      -opt "$OPT" \
      --launcher pytorch
  done
done

echo ""
echo "============================================================"
echo "[INFO] High official val inference done"
echo "[INFO] Time: $(date)"
echo "============================================================"
