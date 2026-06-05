# Paper Assets Status

This document lists the tables and figures needed by the current FastGainRouter paper draft.

## Required Tables

| Table | File | Status |
|---|---|---|
| Split Usage | `paper/tables/table_split_usage.tex` | generated |
| Candidate Branch Audit | `paper/tables/table_candidate_branch_audit.tex` | generated |
| Branch Optimal Ratio | `paper/tables/table_branch_optimal_ratio.tex` | generated |
| Gain Analysis | `paper/tables/table_gain_analysis.tex` | generated |
| Main Quantitative Results | `paper/tables/table_main_quantitative_results.tex` | generated |

## Required Figures

| Figure | File | Status |
|---|---|---|
| Method Overview | `figures/method_overview_v1.png`, `paper/figures/method_overview_v1.png` | generated if matplotlib available |
| PSNR-Latency Trade-off | `figures/final_compact/fig_compact_psnr_latency_v1.png` | exists |
| Action Ratio | `figures/final_compact/fig_compact_action_ratio_v1.png` | exists |
| Qualitative Comparison | `figures/final_qualitative_v1/qualitative_grid_v1.png` | generated and selected |

## Optional Figures

| Figure | File | Status |
|---|---|---|
| Candidate PSNR vs Forward Latency | `figures/final_traininner/fig_candidate_psnr_forward_latency_traininner_v1.png` | exists |
| Candidate LPIPS vs Forward Latency | `figures/final_traininner/fig_candidate_lpips_forward_latency_traininner_v1.png` | exists |

## Notes

- The qualitative figure is currently the main remaining paper asset.
- Latency should be described as a measured component-based estimate.
- `references.bib` contains `[CHECK BIBTEX]` notes and should be verified before final submission.


## Final Selected Qualitative Rows

| Row | Category | Task / Difficulty | Message |
|---|---|---|---|
| A | Skip/Input branch | water / easy | Input is preferable; restoration is unnecessary. |
| B | Tiny sufficient | scratch / medium | Tiny substantially improves Input and is close to Base in PSNR. |
| C | Base needed | dust / medium | Base provides clear additional gain over Tiny. |
| D | Efficient small-loss | clean / hard | Efficient policy selects Tiny instead of Base with small gap. |
