from pathlib import Path
import pandas as pd
import torch
import lpips
from PIL import Image
import numpy as np
from tqdm import tqdm

ROOT = Path("/data/donggyun/SIDL_project")
TABLE_DIR = ROOT / "my_experiment" / "skip_restore_v2_tiny" / "tables"

CSV_PATH = TABLE_DIR / "full_val_tiny_base_policy_table_v1.csv"
OUT_PATH = TABLE_DIR / "full_val_tiny_base_policy_table_v1_lpips.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device:", device)

loss_fn = lpips.LPIPS(net="alex").to(device)
loss_fn.eval()

def load_img_for_lpips(path: str):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)

@torch.no_grad()
def calc_lpips(path_a: str, path_b: str) -> float:
    a = load_img_for_lpips(path_a)
    b = load_img_for_lpips(path_b)
    val = loss_fn(a, b)
    return float(val.item())

if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

df = pd.read_csv(CSV_PATH)
print("[INFO] rows:", len(df))
print("[INFO] output:", OUT_PATH)

lpips_input = []
lpips_tiny = []
lpips_base = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="full_val_tiny_base LPIPS"):
    gt_path = row["gt_path"]
    lpips_input.append(calc_lpips(row["input_path"], gt_path))
    lpips_tiny.append(calc_lpips(row["tiny_path"], gt_path))
    lpips_base.append(calc_lpips(row["base_path"], gt_path))

df["lpips_input"] = lpips_input
df["lpips_tiny"] = lpips_tiny
df["lpips_base"] = lpips_base

names = ["input", "tiny", "base"]
vals = df[["lpips_input", "lpips_tiny", "lpips_base"]].values
df["oracle_label_lpips"] = [names[int(np.argmin(v))] for v in vals]

df["lpips_oracle_tiny_base"] = df[["lpips_tiny", "lpips_base"]].min(axis=1)
df["lpips_oracle_input_tiny_base"] = df[["lpips_input", "lpips_tiny", "lpips_base"]].min(axis=1)

df.to_csv(OUT_PATH, index=False)

print("[OK] saved:", OUT_PATH)

print()
print("[LPIPS Summary: lower is better]")
print("Always Input:", df["lpips_input"].mean())
print("Always Tiny: ", df["lpips_tiny"].mean())
print("Always Base: ", df["lpips_base"].mean())
print("Oracle Tiny/Base:", df["lpips_oracle_tiny_base"].mean())
print("Oracle Input/Tiny/Base:", df["lpips_oracle_input_tiny_base"].mean())

print()
print("[LPIPS Oracle label ratio]")
print(df["oracle_label_lpips"].value_counts())
print(df["oracle_label_lpips"].value_counts(normalize=True))
