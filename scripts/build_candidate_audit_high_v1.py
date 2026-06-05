from pathlib import Path
import math
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import lpips
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception as e:
    raise ImportError("scikit-image is required. Try: pip install scikit-image") from e

ROOT = Path("/data/donggyun/SIDL_project")
BASE = ROOT / "my_experiment" / "skip_restore_v2_tiny"
RESULTS_ROOT = ROOT / "baseline" / "NAFNet" / "results"

TABLE = BASE / "tables" / "full_val_tiny_base_policy_table_v1_lpips.csv"

OUT_TABLE = BASE / "tables" / "full_val_candidate_audit_v1.csv"
OUT_SUMMARY = BASE / "results" / "candidate_branch_summary_v1.csv"
OUT_ORACLE = BASE / "results" / "candidate_oracle_ratios_v1.csv"
OUT_GAIN = BASE / "results" / "candidate_gain_summary_v1.csv"

OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

PURE_LAT = BASE / "results" / "pure_forward_latency_v1.csv"

CKPTS = {
    "tiny": ROOT / "baseline" / "NAFNet" / "experiments" / "NAFNet-SIDL-all-tiny-width4-lr1e3-es20000-v1" / "models" / "best_net_g.pth",
    "base": ROOT / "baseline" / "NAFNet" / "experiments" / "NAFNet-SIDL-all-low-es20000-v1" / "models" / "best_net_g.pth",
    "high": ROOT / "baseline" / "NAFNet" / "experiments" / "NAFNet-SIDL-all-high-es20000-v1" / "models" / "best_net_g.pth",
}


def read_img(path: Path):
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


def calc_ssim(img, gt):
    return float(ssim_fn(img, gt, channel_axis=2, data_range=255))


def high_result_dir(task, diff):
    return RESULTS_ROOT / f"NAFNet-SIDL-v2-high-width16-{task}-{diff}-val" / "visualization" / f"sidl-{task}-{diff}-val"


def find_output(result_dir: Path, filename: str):
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

    raise FileNotFoundError(f"High output not found for {filename} in {result_dir}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = lpips.LPIPS(net="alex").to(device).eval()


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
    return float(loss_fn(a, b).item())


if not TABLE.exists():
    raise FileNotFoundError(TABLE)

df = pd.read_csv(TABLE)
print("[INFO] rows:", len(df))
print("[INFO] output:", OUT_TABLE)

high_paths = []
psnr_high = []
ssim_high = []
lpips_high = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="candidate_audit_high"):
    task = row["task"]
    diff = row["difficulty"]
    filename = row["filename"]

    hdir = high_result_dir(task, diff)
    if not hdir.exists():
        raise FileNotFoundError(hdir)

    high_path = find_output(hdir, filename)
    gt_path = Path(row["gt_path"])

    high_img = read_img(high_path)
    gt_img = read_img(gt_path)

    high_paths.append(str(high_path))
    psnr_high.append(calc_psnr(high_img, gt_img))
    ssim_high.append(calc_ssim(high_img, gt_img))
    lpips_high.append(calc_lpips(str(high_path), str(gt_path)))

df["high_path"] = high_paths
df["psnr_high"] = psnr_high
df["ssim_high"] = ssim_high
df["lpips_high"] = lpips_high

# Gains
df["gain_high_vs_tiny"] = df["psnr_high"] - df["psnr_tiny"]
df["gain_high_vs_base"] = df["psnr_high"] - df["psnr_base"]
df["gain_high_vs_input"] = df["psnr_high"] - df["psnr_input"]

# Oracle over four candidates
psnr_cols = ["psnr_input", "psnr_tiny", "psnr_base", "psnr_high"]
lpips_cols = ["lpips_input", "lpips_tiny", "lpips_base", "lpips_high"]

names = ["input", "tiny", "base", "high"]

df["oracle_label_psnr_4way"] = [
    names[int(np.argmax(v))]
    for v in df[psnr_cols].values
]
df["oracle_psnr_4way"] = df[psnr_cols].max(axis=1)

df["oracle_label_lpips_4way"] = [
    names[int(np.argmin(v))]
    for v in df[lpips_cols].values
]
df["oracle_lpips_4way"] = df[lpips_cols].min(axis=1)

df.to_csv(OUT_TABLE, index=False)

# Branch summary
rows = []
pure = pd.read_csv(PURE_LAT) if PURE_LAT.exists() else pd.DataFrame()

def pure_ms_for(branch):
    if not PURE_LAT.exists():
        return np.nan
    mapping = {
        "tiny": "tiny_width4",
        "base": "base_width8",
        "high": "old_high_width16",
    }
    model_name = mapping.get(branch)
    if model_name is None:
        return 0.0
    s = pure[pure["model"] == model_name]
    if len(s) == 0:
        return np.nan
    return float(s["median_ms"].iloc[0])

for branch in ["input", "tiny", "base", "high"]:
    if branch == "input":
        ckpt_mb = 0.0
        pure_ms = 0.0
    else:
        ckpt_mb = CKPTS[branch].stat().st_size / 1024 / 1024 if CKPTS[branch].exists() else np.nan
        pure_ms = pure_ms_for(branch)

    rows.append({
        "branch": branch,
        "psnr": df[f"psnr_{branch}"].mean(),
        "ssim": df[f"ssim_{branch}"].mean(),
        "lpips": df[f"lpips_{branch}"].mean(),
        "checkpoint_mb": ckpt_mb,
        "pure_forward_median_ms": pure_ms,
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT_SUMMARY, index=False)

oracle_rows = []
for col in ["oracle_label_psnr", "oracle_label_lpips", "oracle_label_psnr_4way", "oracle_label_lpips_4way"]:
    vc = df[col].value_counts(normalize=True)
    for branch in ["input", "tiny", "base", "high"]:
        oracle_rows.append({
            "oracle_type": col,
            "branch": branch,
            "ratio": float(vc.get(branch, 0.0)),
            "count": int((df[col] == branch).sum()),
        })

oracle_df = pd.DataFrame(oracle_rows)
oracle_df.to_csv(OUT_ORACLE, index=False)

gain_rows = []
for col in [
    "gain_tiny_vs_input",
    "gain_base_vs_tiny",
    "gain_high_vs_tiny",
    "gain_high_vs_base",
    "gain_high_vs_input",
]:
    s = df[col]
    gain_rows.append({
        "gain": col,
        "mean": s.mean(),
        "median": s.median(),
        "q25": s.quantile(0.25),
        "q75": s.quantile(0.75),
        "ratio_gt_0": float((s > 0).mean()),
        "ratio_gt_0p1": float((s > 0.1).mean()),
        "ratio_gt_0p2": float((s > 0.2).mean()),
        "ratio_gt_0p5": float((s > 0.5).mean()),
    })

gain_df = pd.DataFrame(gain_rows)
gain_df.to_csv(OUT_GAIN, index=False)

print("[OK] saved:", OUT_TABLE)
print()
print("[BRANCH SUMMARY]")
print(summary)
print()
print("[ORACLE RATIOS]")
print(oracle_df)
print()
print("[GAIN SUMMARY]")
print(gain_df)
