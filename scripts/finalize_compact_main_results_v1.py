from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data/donggyun/SIDL_project")
BASE = ROOT / "my_experiment" / "skip_restore_v2_tiny"

RESULT_DIR = BASE / "results"
FIG_DIR = BASE / "figures" / "final_compact"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY = RESULT_DIR / "fast_gain_router_summary_traininner_corrected_v1.csv"

if not SUMMARY.exists():
    raise FileNotFoundError(SUMMARY)

df = pd.read_csv(SUMMARY)

methods = [
    "Always Input",
    "Always Tiny",
    "Always Base",
    "RFGain psnr>=base-0.1",
    "RFGain psnr>=base-0.2",
]

display = {
    "Always Input": "Always Input",
    "Always Tiny": "Always Tiny",
    "Always Base": "Always Base",
    "RFGain psnr>=base-0.1": "Ours-Balanced",
    "RFGain psnr>=base-0.2": "Ours-Efficient",
}

main = df[df["method"].isin(methods)].copy()
main["order"] = main["method"].map({m: i for i, m in enumerate(methods)})
main = main.sort_values("order").drop(columns=["order"])
main["display"] = main["method"].map(display)

# Baseline references
input_row = main[main["method"] == "Always Input"].iloc[0]
tiny_row = main[main["method"] == "Always Tiny"].iloc[0]
base_row = main[main["method"] == "Always Base"].iloc[0]

base_psnr = float(base_row["psnr"])
base_lpips = float(base_row["lpips"])
base_lat = float(base_row["latency_ms"])

tiny_psnr = float(tiny_row["psnr"])
tiny_lpips = float(tiny_row["lpips"])
tiny_lat = float(tiny_row["latency_ms"])

main["delta_psnr_vs_base"] = main["psnr"] - base_psnr
main["delta_lpips_vs_base"] = main["lpips"] - base_lpips
main["latency_reduction_vs_base_pct"] = (base_lat - main["latency_ms"]) / base_lat * 100

main["delta_psnr_vs_tiny"] = main["psnr"] - tiny_psnr
main["delta_lpips_vs_tiny"] = main["lpips"] - tiny_lpips
main["latency_reduction_vs_tiny_pct"] = (tiny_lat - main["latency_ms"]) / tiny_lat * 100

out_cols = [
    "display",
    "method",
    "psnr",
    "ssim",
    "lpips",
    "latency_ms",
    "delta_psnr_vs_base",
    "latency_reduction_vs_base_pct",
    "delta_psnr_vs_tiny",
    "latency_reduction_vs_tiny_pct",
    "input_ratio",
    "tiny_ratio",
    "base_ratio",
]

out = main[out_cols].copy()

for c in out.columns:
    if c not in ["display", "method"]:
        out[c] = out[c].astype(float).round(4)

csv_path = RESULT_DIR / "final_compact_main_table_v1.csv"
md_path = RESULT_DIR / "final_compact_main_table_v1.md"

out.to_csv(csv_path, index=False)
md_path.write_text(out.to_markdown(index=False))

print("[COMPACT MAIN TABLE]")
print(out)
print("[SAVED]", csv_path)
print("[SAVED]", md_path)

# Plot 1: PSNR-latency
plt.figure(figsize=(8, 5))
plt.scatter(main["latency_ms"], main["psnr"], s=100)

for _, r in main.iterrows():
    plt.annotate(
        r["display"],
        (r["latency_ms"], r["psnr"]),
        textcoords="offset points",
        xytext=(6, 5),
        fontsize=10,
    )

plt.xlabel("Component-based latency estimate (ms / image)")
plt.ylabel("PSNR (dB)")
plt.title("Quality-Latency Trade-off")
plt.grid(True, alpha=0.3)
plt.tight_layout()

fig_path = FIG_DIR / "fig_compact_psnr_latency_v1.png"
plt.savefig(fig_path, dpi=240)
plt.close()
print("[SAVED]", fig_path)

# Plot 2: action ratio for ours only
ours = main[main["method"].str.startswith("RFGain")].copy()

x = np.arange(len(ours))
input_r = ours["input_ratio"].astype(float).values
tiny_r = ours["tiny_ratio"].astype(float).values
base_r = ours["base_ratio"].astype(float).values

plt.figure(figsize=(7, 4.5))
plt.bar(x, input_r, label="Input")
plt.bar(x, tiny_r, bottom=input_r, label="Tiny")
plt.bar(x, base_r, bottom=input_r + tiny_r, label="Base")

plt.xticks(x, ours["display"])
plt.ylabel("Action ratio")
plt.title("Branch Usage of FastGainRouter")
plt.legend()
plt.tight_layout()

fig_path = FIG_DIR / "fig_compact_action_ratio_v1.png"
plt.savefig(fig_path, dpi=240)
plt.close()
print("[SAVED]", fig_path)

# Short text summary
txt = RESULT_DIR / "final_compact_key_claims_v1.txt"

bal = main[main["method"] == "RFGain psnr>=base-0.1"].iloc[0]
eff = main[main["method"] == "RFGain psnr>=base-0.2"].iloc[0]

txt.write_text(f"""Compact final claims

Ours-Balanced:
- PSNR {bal['psnr']:.4f}, latency {bal['latency_ms']:.2f} ms.
- Compared with Always Base: PSNR {bal['delta_psnr_vs_base']:+.4f} dB, latency reduction {bal['latency_reduction_vs_base_pct']:.2f}%.
- Compared with Always Tiny: PSNR {bal['delta_psnr_vs_tiny']:+.4f} dB.

Ours-Efficient:
- PSNR {eff['psnr']:.4f}, latency {eff['latency_ms']:.2f} ms.
- Compared with Always Base: PSNR {eff['delta_psnr_vs_base']:+.4f} dB, latency reduction {eff['latency_reduction_vs_base_pct']:.2f}%.
- Compared with Always Tiny: PSNR {eff['delta_psnr_vs_tiny']:+.4f} dB, latency reduction {eff['latency_reduction_vs_tiny_pct']:.2f}%.
""")

print("[SAVED]", txt)
print("[DONE]")
