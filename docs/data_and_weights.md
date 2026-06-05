# Data and Weights

## Dataset

The SIDL dataset is not included in this repository. Download and place it externally.

Expected path in the original experiment:

```text
/data/donggyun/SIDL_project/data_full/
```

## Checkpoints

Model checkpoints are not included in GitHub.

Excluded examples:

```text
baseline/NAFNet/experiments/*/models/*.pth
my_experiment/skip_restore_v2_tiny/models/fast_gain_router_traininner_rfgain_v1.joblib
```

The final RFGain router `.joblib` is excluded because it is large. Re-run the router training script to regenerate it.

## Large Intermediate Outputs

Full inference outputs under `baseline/NAFNet/results/` are excluded. They can be regenerated using the scripts in this repository.
