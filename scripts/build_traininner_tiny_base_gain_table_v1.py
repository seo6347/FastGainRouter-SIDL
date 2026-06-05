from pathlib import Path
import math
import csv
import re
import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path("/data/donggyun/SIDL_project")
DATA_ROOT = ROOT / "data_full"
RESULTS_ROOT = ROOT / "baseline" / "NAFNet" / "results"
OUT_DIR = ROOT / "my_experiment" / "skip_restore_v2_tiny" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIR = DATA_ROOT / "train" / "all_custom_train_inner" / "input"
GT_DIR = DATA_ROOT / "train" / "all_custom_train_inner" / "target"

TINY_DIR = RESULTS_ROOT / "NAFNet-SIDL-v2-tiny-lr1e3-traininner-full-v1" / "visualization" / "sidl-v2-train-inner"
BASE_DIR = RESULTS_ROOT / "NAFNet-SIDL-v2-base-width8-traininner-full-v1" / "visualization" / "sidl-v2-train-inner"

OUT = OUT_DIR / "traininner_tiny_base_gain_table_v1.csv"

def read_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def calc_psnr(img, gt):
    img = img.astype(np.float64)
    gt = gt.astype(np.float64)
    mse = np.mean((img - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))

def find_output(result_dir, filename):
    direct = result_dir / filename
    if direct.exists():
        return direct

    stem = Path(filename).stem
    candidates = sorted([
        p for p in result_dir.glob(f"{stem}*.png")
        if not p.name.endswith("_gt.png")
    ])
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"output not found: {filename} in {result_dir}")

def parse_task(filename):
    m = re.match(r"^(clean|dust|finger|mixed|scratch|water)__", Path(filename).name)
    return m.group(1) if m else "unknown"

def parse_case_key(filename):
    name = Path(filename).stem
    if "__" in name:
        name = name.split("__", 1)[1]
    parts = name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:2])
    return name

for p in [INPUT_DIR, GT_DIR, TINY_DIR, BASE_DIR]:
    if not p.exists():
        raise FileNotFoundError(p)

input_files = sorted(INPUT_DIR.glob("*.png"))
rows = []
missing = []

print("[INFO] input files:", len(input_files))
print("[INFO] tiny dir:", TINY_DIR)
print("[INFO] base dir:", BASE_DIR)
print("[INFO] output:", OUT)

for input_path in tqdm(input_files, desc="traininner_gain_table"):
    name = input_path.name
    gt_path = GT_DIR / name

    if not gt_path.exists():
        missing.append((name, "gt_missing"))
        continue

    try:
        tiny_path = find_output(TINY_DIR, name)
        base_path = find_output(BASE_DIR, name)
    except FileNotFoundError as e:
        missing.append((name, str(e)))
        continue

    inp = read_img(input_path)
    gt = read_img(gt_path)
    tiny = read_img(tiny_path)
    base = read_img(base_path)

    psnr_input = calc_psnr(inp, gt)
    psnr_tiny = calc_psnr(tiny, gt)
    psnr_base = calc_psnr(base, gt)

    vals = {"input": psnr_input, "tiny": psnr_tiny, "base": psnr_base}
    oracle = max(vals, key=vals.get)

    rows.append({
        "split": "traininner_full",
        "task": parse_task(name),
        "case_key": parse_case_key(name),
        "filename": name,
        "input_path": str(input_path),
        "gt_path": str(gt_path),
        "tiny_path": str(tiny_path),
        "base_path": str(base_path),
        "psnr_input": psnr_input,
        "psnr_tiny": psnr_tiny,
        "psnr_base": psnr_base,
        "oracle_label_psnr": oracle,
        "gain_tiny_vs_input": psnr_tiny - psnr_input,
        "gain_base_vs_tiny": psnr_base - psnr_tiny,
        "gain_base_vs_input": psnr_base - psnr_input,
    })

fieldnames = [
    "split", "task", "case_key", "filename",
    "input_path", "gt_path", "tiny_path", "base_path",
    "psnr_input", "psnr_tiny", "psnr_base", "oracle_label_psnr",
    "gain_tiny_vs_input", "gain_base_vs_tiny", "gain_base_vs_input",
]

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("[OK] saved:", OUT)
print("[OK] rows:", len(rows))
print("[OK] missing:", len(missing))

if missing:
    miss = OUT.with_name(OUT.stem + "_missing.csv")
    with miss.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "reason"])
        writer.writerows(missing)
    print("[WARN] missing saved:", miss)

labels = [r["oracle_label_psnr"] for r in rows]
print("[Oracle label ratio]")
for a in ["input", "tiny", "base"]:
    print(a, labels.count(a), labels.count(a)/len(labels) if labels else 0)

print("[Mean PSNR]")
for k in ["psnr_input", "psnr_tiny", "psnr_base"]:
    print(k, float(np.mean([r[k] for r in rows])))
