# Data, Models, and Excluded Artifacts

## External research resources

FastGainRouter uses the SIDL dataset introduced by Choi, Park, and Kim (AAAI 2025), and NAFNet-based restoration branches following Chen et al. (ECCV 2022). References are provided in the [project paper](../paper/FastGainRouter_Input-Adaptive_Efficient_Restoration_for_Smartphone_Images_with_Dirty_Lenses.pdf) and [bibliography](../paper/references.bib).

SIDL data and the original NAFNet implementation must be obtained separately from their authors and used under their respective terms. Neither the dataset nor a vendored copy of NAFNet is included here.

## Data protocol

The experiments use project-specific case-level `train_inner` and `es_val_inner` subdivisions of the official training split, plus the official validation split for final evaluation. The public release does not include split manifests or the complete split-preparation workflow.

Original scripts reference data under `/data/donggyun/SIDL_project/data_full/`. This is an original experiment path, not a directory automatically created by cloning the repository.

## Excluded artifacts

- Raw dataset images.
- Tiny, Base, and additional branch checkpoints.
- Fitted router `.joblib` files.
- Bulk per-image restoration outputs, feature tables, policy tables, and logs.

The published aggregate result summaries, selected figures, and paper are included. The repository therefore supports inspection of the research implementation and results, but does not provide everything needed to regenerate all excluded artifacts.

See [Original Experiment Workflow](reproduce.md) for the code dependencies and release boundaries.
