from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path("/data/donggyun/SIDL_project")
BASE = ROOT / "my_experiment" / "skip_restore_v2_tiny"

TABLE = BASE / "tables" / "traininner_tiny_base_gain_table_v1.csv"
OUT = BASE / "features" / "traininner_fast_features_v1.csv"
LAT = BASE / "results" / "traininner_fast_feature_latency_v1.csv"

OUT.parent.mkdir(parents=True, exist_ok=True)
LAT.parent.mkdir(parents=True, exist_ok=True)

def extract_fast_features(path):
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

df = pd.read_csv(TABLE)
rows = []

print("[INFO] rows:", len(df))
print("[INFO] output:", OUT)

t0 = time.perf_counter()

for idx, row in tqdm(df.iterrows(), total=len(df), desc="traininner_fast_features"):
    feats = extract_fast_features(row["input_path"])
    out = {
        "row_id": idx,
        "input_path": row["input_path"],
        "filename": row.get("filename", ""),
        "task": row.get("task", ""),
        "case_key": row.get("case_key", ""),
    }
    out.update(feats)
    rows.append(out)

t1 = time.perf_counter()

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT, index=False)

elapsed = t1 - t0
ms_per_image = elapsed / len(df) * 1000

pd.DataFrame([{
    "split": "traininner_full",
    "num_images": len(df),
    "feature_seconds": elapsed,
    "feature_ms_per_image": ms_per_image,
}]).to_csv(LAT, index=False)

print("[OK] saved:", OUT)
print("[OK] shape:", out_df.shape)
print("[LATENCY] ms/image:", ms_per_image)
print("[SAVED]", LAT)
