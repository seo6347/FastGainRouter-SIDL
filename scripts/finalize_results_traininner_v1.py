from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data/donggyun/SIDL_project")
BASE = ROOT / "my_experiment" / "skip_restore_v2_tiny"

RESULT_DIR = BASE / "results"
FIG_DIR = BASE / "figures" / "final_traininner"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAIN_SUMMARY = RESULT_DIR / "fast_gain_router_summary_traininner_corrected_v1.csv"
CANDIDATE_SUMMARY = RESULT_DIR / "candidate_branch_summary_v1.csv"

OUT_MAIN_CSV = RESULT_DIR / "final_main_summary_traininner_v1.csv"
OUT_MAIN_MD = RESULT_DIR / "final_main_summary_traininner_v1.md"

OUT_CAND_CSV = RESULT_DIR / "final_candidate_branch_summary_traininner_v1.csv"
OUT_CAND_MD = RESULT_DIR / "final_candidate_branch_summary_traininner_v1.md"

if not MAIN_SUMMARY.exists():
    raise FileNotFoundError(MAIN_SUMMARY)
if not CANDIDATE_SUMMARY.exists():
    raise FileNotFoundError(CANDIDATE_SUMMARY)

df = pd.read_csv(MAIN_SUMMARY)
cand = pd.read_csv(CANDIDATE_SUMMARY)

method_order = [
    "Always Input",
    "Always Tiny",
    "Always Base",
    "RFGain psnr>=base-0.05",
    "RFGain psnr>=base-0.1",
    "RFGain psnr>=base-0.2",
    "RFGain psnr>=base-0.3",
    "Oracle PSNR I/T/B",
]

display_name = {
    "Always Input": "Always Input",
    "Always Tiny": "Always Tiny",
    "Always Base": "Always Base",
    "RFGain psnr>=base-0.05": "Ours-QF",
    "RFGain psnr>=base-0.1": "Ours-Balanced",
    "RFGain psnr>=base-0.2": "Ours-Efficient",
    "RFGain psnr>=base-0.3": "Ours-Aggressive",
    "Oracle PSNR I/T/B": "Oracle",
}

main = df[df["method"].isin(method_order)].copy()
main["order"] = main["method"].map({m: i for i, m in enumerate(method_order)})
main = main.sort_values("order").drop(columns=["order"])
main["display"] = main["method"].map(display_name)

base = main[main["method"] == "Always Base"].iloc[0]
base_psnr = float(base["psnr"])
base_ssim = float(base["ssim"])
base_lpips = float(base["lpips"])
base_lat = float(base["latency_ms"])

main["delta_psnr_vs_base"] = main["psnr"] - base_psnr
main["delta_ssim_vs_base"] = main["ssim"] - base_ssim
main["delta_lpips_vs_base"] = main["lpips"] - base_lpips
main["delta_latency_ms_vs_base"] = main["latency_ms"] - base_lat
main["latency_reduction_pct_vs_base"] = (base_lat - main["latency_ms"]) / base_lat * 100

main_out = main[[
    "display",
    "method",
    "psnr",
    "ssim",
    "lpips",
    "latency_ms",
    "delta_psnr_vs_base",
    "delta_ssim_vs_base",
    "delta_lpips_vs_base",
    "latency_reduction_pct_vs_base",
    "input_ratio",
    "tiny_ratio",
    "base_ratio",
]].copy()

round_cols = [
    "psnr", "ssim", "lpips", "latency_ms",
    "delta_psnr_vs_base", "delta_ssim_vs_base",
    "delta_lpips_vs_base",
    "latency_reduction_pct_vs_base",
    "input_ratio", "tiny_ratio", "base_ratio",
]
for c in round_cols:
    main_out[c] = main_out[c].astype(float).round(4)

main_out.to_csv(OUT_MAIN_CSV, index=False)
OUT_MAIN_MD.write_text(main_out.to_markdown(index=False))

print("[FINAL MAIN SUMMARY]")
print(main_out)
print("[SAVED]", OUT_MAIN_CSV)
print("[SAVED]", OUT_MAIN_MD)

# Candidate summary table
cand_base = cand[cand["branch"] == "base"].iloc[0]
cand["delta_psnr_vs_base"] = cand["psnr"] - float(cand_base["psnr"])
cand["delta_lpips_vs_base"] = cand["lpips"] - float(cand_base["lpips"])
cand["pure_latency_delta_vs_base"] = cand["pure_forward_median_ms"] - float(cand_base["pure_forward_median_ms"])

cand_table = cand[[
    "branch",
    "psnr",
    "ssim",
    "lpips",
    "checkpoint_mb",
    "pure_forward_median_ms",
    "delta_psnr_vs_base",
    "delta_lpips_vs_base",
    "pure_latency_delta_vs_base",
]].copy()

for c in [
    "psnr", "ssim", "lpips", "checkpoint_mb",
    "pure_forward_median_ms", "delta_psnr_vs_base",
    "delta_lpips_vs_base", "pure_latency_delta_vs_base",
]:
    cand_table[c] = cand_table[c].astype(float).round(4)

cand_table.to_csv(OUT_CAND_CSV, index=False)
OUT_CAND_MD.write_text(cand_table.to_markdown(index=False))

print()
print("[FINAL CANDIDATE SUMMARY]")
print(cand_table)
print("[SAVED]", OUT_CAND_CSV)
print("[SAVED]", OUT_CAND_MD)

# Plot 1: PSNR-latency
plot_methods = [
    "Always Input",
    "Always Tiny",
    "Always Base",
    "RFGain psnr>=base-0.05",
    "RFGain psnr>=base-0.1",
    "RFGain psnr>=base-0.2",
    "RFGain psnr>=base-0.3",
]
plot_df = main[main["method"].isin(plot_methods)].copy()

plt.figure(figsize=(8, 5))
plt.scatter(plot_df["latency_ms"], plot_df["psnr"], s=85)
for _, r in plot_df.iterrows():
    label = display_name.get(r["method"], r["method"])
    plt.annotate(label, (r["latency_ms"], r["psnr"]), textcoords="offset points", xytext=(6, 5), fontsize=9)

plt.xlabel("Component-based latency estimate (ms / image)")
plt.ylabel("PSNR (dB)")
plt.title("PSNR-Latency Trade-off")
plt.grid(True, alpha=0.3)
plt.tight_layout()
out = FIG_DIR / "fig_psnr_latency_pareto_traininner_v1.png"
plt.savefig(out, dpi=220)
plt.close()
print("[SAVED]", out)

# Plot 2: action ratio
ratio_methods = [
    "RFGain psnr>=base-0.05",
    "RFGain psnr>=base-0.1",
    "RFGain psnr>=base-0.2",
    "RFGain psnr>=base-0.3",
]
ratio_df = main[main["method"].isin(ratio_methods)].copy()
ratio_df["display"] = ratio_df["method"].map(display_name)

x = np.arange(len(ratio_df))
input_r = ratio_df["input_ratio"].astype(float).values
tiny_r = ratio_df["tiny_ratio"].astype(float).values
base_r = ratio_df["base_ratio"].astype(float).values

plt.figure(figsize=(8, 5))
plt.bar(x, input_r, label="Input")
plt.bar(x, tiny_r, bottom=input_r, label="Tiny")
plt.bar(x, base_r, bottom=input_r + tiny_r, label="Base")
plt.xticks(x, ratio_df["display"], rotation=15)
plt.ylabel("Action ratio")
plt.title("Selected Action Ratios")
plt.legend()
plt.tight_layout()
out = FIG_DIR / "fig_action_ratio_traininner_v1.png"
plt.savefig(out, dpi=220)
plt.close()
print("[SAVED]", out)

# Plot 3: candidate PSNR vs pure forward latency
plt.figure(figsize=(7, 5))
plt.scatter(cand["pure_forward_median_ms"], cand["psnr"], s=85)
for _, r in cand.iterrows():
    plt.annotate(r["branch"], (r["pure_forward_median_ms"], r["psnr"]), textcoords="offset points", xytext=(6, 5), fontsize=10)
plt.xlabel("Pure forward latency (ms / image)")
plt.ylabel("PSNR (dB)")
plt.title("Candidate Quality vs Model Latency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
out = FIG_DIR / "fig_candidate_psnr_forward_latency_traininner_v1.png"
plt.savefig(out, dpi=220)
plt.close()
print("[SAVED]", out)

# Plot 4: candidate LPIPS vs pure forward latency
plt.figure(figsize=(7, 5))
plt.scatter(cand["pure_forward_median_ms"], cand["lpips"], s=85)
for _, r in cand.iterrows():
    plt.annotate(r["branch"], (r["pure_forward_median_ms"], r["lpips"]), textcoords="offset points", xytext=(6, 5), fontsize=10)
plt.xlabel("Pure forward latency (ms / image)")
plt.ylabel("LPIPS (lower is better)")
plt.title("Candidate LPIPS vs Model Latency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
out = FIG_DIR / "fig_candidate_lpips_forward_latency_traininner_v1.png"
plt.savefig(out, dpi=220)
plt.close()
print("[SAVED]", out)

# Key findings
key_txt = RESULT_DIR / "final_key_findings_traininner_v1.txt"

def get_row(method):
    return main[main["method"] == method].iloc[0]

qf = get_row("RFGain psnr>=base-0.05")
bal = get_row("RFGain psnr>=base-0.1")
eff = get_row("RFGain psnr>=base-0.2")
agg = get_row("RFGain psnr>=base-0.3")

key_txt.write_text(f"""Final key findings based on train_inner full router

1. Candidate branch audit
- Tiny improves PSNR over Input by {cand.loc[cand.branch.eq('tiny'), 'psnr'].iloc[0] - cand.loc[cand.branch.eq('input'), 'psnr'].iloc[0]:.3f} dB.
- Base improves PSNR over Tiny by {cand.loc[cand.branch.eq('base'), 'psnr'].iloc[0] - cand.loc[cand.branch.eq('tiny'), 'psnr'].iloc[0]:.3f} dB.
- High gives the highest PSNR, but has worse LPIPS and much larger checkpoint size.
- Therefore Input/Tiny/Base is used as the main branch set.

2. Main router results
- Quality-first RFGain base-0.05: PSNR {qf['delta_psnr_vs_base']:+.3f} dB vs Always Base, latency reduction {qf['latency_reduction_pct_vs_base']:.1f}%.
- Balanced RFGain base-0.1: PSNR {bal['delta_psnr_vs_base']:+.3f} dB vs Always Base, latency reduction {bal['latency_reduction_pct_vs_base']:.1f}%.
- Efficient RFGain base-0.2: PSNR {eff['delta_psnr_vs_base']:+.3f} dB vs Always Base, latency reduction {eff['latency_reduction_pct_vs_base']:.1f}%.
- Aggressive RFGain base-0.3: PSNR {agg['delta_psnr_vs_base']:+.3f} dB vs Always Base, latency reduction {agg['latency_reduction_pct_vs_base']:.1f}%.
""")

print("[SAVED]", key_txt)
print("[DONE] finalized traininner results")
