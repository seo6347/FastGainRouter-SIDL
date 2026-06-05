from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path("/data/donggyun/SIDL_project")
BASE = ROOT / "my_experiment" / "skip_restore_v2_tiny"

TABLE = BASE / "tables" / "full_val_tiny_base_policy_table_v1_lpips.csv"
SUMMARY = BASE / "results" / "fast_gain_router_summary_traininner_v1.csv"

OUT_LAT = BASE / "results" / "fast_feature_latency_eval_v1.csv"
OUT_SUMMARY = BASE / "results" / "fast_gain_router_summary_traininner_corrected_v1.csv"
OUT_MAIN = BASE / "results" / "final_main_summary_traininner_corrected_v1.csv"

N_RUNS = 3

def extract_fast_features(path: str):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    lap = cv2.Laplacian(gray, cv2.CV_32F)

    return {
        "gray_mean": float(gray_f.mean()),
        "gray_std": float(gray_f.std()),
        "dark_ratio_64": float((gray < 64).mean()),
        "bright_ratio_225": float((gray > 225).mean()),
        "sat_mean": float(sat.mean()),
        "sat_std": float(sat.std()),
        "value_mean": float(val.mean()),
        "value_std": float(val.std()),
        "laplacian_var": float(lap.var()),
    }

if not TABLE.exists():
    raise FileNotFoundError(TABLE)
if not SUMMARY.exists():
    raise FileNotFoundError(SUMMARY)

df = pd.read_csv(TABLE)
paths = df["input_path"].tolist()

print("[INFO] measuring feature latency on full_val")
print("[INFO] rows:", len(paths))

lat_rows = []

for run in range(1, N_RUNS + 1):
    t0 = time.perf_counter()

    for p in tqdm(paths, desc=f"feature_latency_run{run}"):
        _ = extract_fast_features(p)

    t1 = time.perf_counter()
    elapsed = t1 - t0
    ms = elapsed / len(paths) * 1000

    lat_rows.append({
        "run": run,
        "num_images": len(paths),
        "feature_seconds": elapsed,
        "feature_ms_per_image": ms,
    })

lat_df = pd.DataFrame(lat_rows)
lat_df.to_csv(OUT_LAT, index=False)

feature_ms = float(lat_df["feature_ms_per_image"].median())

print()
print("[FEATURE LATENCY]")
print(lat_df)
print("median feature_ms:", feature_ms)
print("[SAVED]", OUT_LAT)

summary = pd.read_csv(SUMMARY)

# Preserve original latency for debugging.
summary["latency_ms_without_feature"] = summary["latency_ms"]

def is_dynamic_method(method: str):
    return method.startswith("RidgeGain") or method.startswith("RFGain")

dynamic_mask = summary["method"].apply(is_dynamic_method)

# Current traininner script used feature_ms=0.0, so add measured feature latency.
summary.loc[dynamic_mask, "latency_ms"] = summary.loc[dynamic_mask, "latency_ms"] + feature_ms

# Recompute deltas vs Always Base.
base_latency = float(summary.loc[summary["method"] == "Always Base", "latency_ms"].iloc[0])
base_psnr = float(summary.loc[summary["method"] == "Always Base", "psnr"].iloc[0])
base_lpips = float(summary.loc[summary["method"] == "Always Base", "lpips"].iloc[0])

summary["delta_psnr_vs_base"] = summary["psnr"] - base_psnr
summary["delta_lpips_vs_base"] = summary["lpips"] - base_lpips
summary["delta_latency_ms_vs_base"] = summary["latency_ms"] - base_latency
summary["latency_reduction_pct_vs_base"] = (base_latency - summary["latency_ms"]) / base_latency * 100

summary.to_csv(OUT_SUMMARY, index=False)

keep = [
    "Always Input",
    "Always Tiny",
    "Always Base",
    "RFGain psnr>=base-0.03",
    "RFGain psnr>=base-0.05",
    "RFGain psnr>=base-0.1",
    "RFGain psnr>=base-0.2",
    "RFGain psnr>=base-0.3",
    "RFGain utility_lambda=0.0",
    "RFGain utility_lambda=0.1",
    "Oracle PSNR I/T/B",
]

main = summary[summary["method"].isin(keep)].copy()

cols = [
    "method",
    "psnr",
    "ssim",
    "lpips",
    "latency_ms",
    "delta_psnr_vs_base",
    "delta_lpips_vs_base",
    "latency_reduction_pct_vs_base",
    "input_ratio",
    "tiny_ratio",
    "base_ratio",
    "latency_ms_without_feature",
]

main = main[cols]

for c in [
    "psnr", "ssim", "lpips", "latency_ms",
    "delta_psnr_vs_base", "delta_lpips_vs_base",
    "latency_reduction_pct_vs_base",
    "input_ratio", "tiny_ratio", "base_ratio",
    "latency_ms_without_feature",
]:
    main[c] = main[c].astype(float).round(4)

main.to_csv(OUT_MAIN, index=False)

print()
print("[CORRECTED MAIN SUMMARY]")
print(main)

print()
print("[SAVED]")
print(OUT_SUMMARY)
print(OUT_MAIN)
