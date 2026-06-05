# FastGainRouter for SIDL Dirty-Lens Restoration

This repository contains scripts, configs, final result summaries, and figures for **FastGainRouter**, an input-adaptive restoration budget selection framework for SIDL dirty-lens image restoration.

## Overview

FastGainRouter selects one of three branches for each degraded input image:

- **Input**: bypass restoration and return the degraded input.
- **Tiny**: lightweight NAFNet restoration branch.
- **Base**: stronger NAFNet restoration branch.

The router predicts restoration gains from lightweight image statistics and selects a branch using threshold-based decision rules.

## Main Claim

Not every dirty-lens image requires the same restoration budget. Some inputs are better left unchanged, many can be handled by Tiny, and only a subset requires Base.

## Main Results

See:

- `results/final_compact_main_table_v1.csv`
- `results/final_compact_key_claims_v1.txt`
- `figures/final_compact/`

## Repository Structure

```text
configs/   BasicSR/NAFNet config files used in the final experiments
scripts/   Training, inference, table generation, router training, and finalization scripts
results/   Final CSV/MD/TXT summaries used in the report
figures/   Final plots and selected qualitative figures
docs/      Reproduction notes and data/weight instructions
paper/     LaTeX paper source or placeholder
```

## Dataset

The SIDL dataset is **not included** in this repository. Place the dataset manually under the expected project structure before running scripts.

See `docs/data_and_weights.md`.

## Reproduction

See `docs/reproduce.md`.

## Notes

Large files are intentionally excluded:

- raw SIDL dataset
- NAFNet checkpoints
- full inference outputs
- generated PNG outputs
- logs
- bulk per-threshold CSV files
- large router `.joblib` artifacts

This repository is intended for code and result-summary release, not for storing full experiment artifacts.
# FastGainRouter-SIDL
