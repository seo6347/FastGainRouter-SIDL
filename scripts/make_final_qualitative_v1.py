from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path("/data/donggyun/SIDL_project")
BASE = ROOT / "my_experiment" / "skip_restore_v2_tiny"

RESULT_DIR = BASE / "results"
FIG_DIR = BASE / "figures" / "final_qualitative_v1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TABLE = BASE / "tables" / "full_val_tiny_base_policy_table_v1_lpips.csv"

# Representative policies
POLICY_BASE_01 = sorted(RESULT_DIR.glob("fast_gain_router_full_val_RFGain_psnrgebase_0p1*_v1.csv"))
POLICY_BASE_02 = sorted(RESULT_DIR.glob("fast_gain_router_full_val_RFGain_psnrgebase_0p2*_v1.csv"))

if not POLICY_BASE_01:
    # fallback: broader search
    POLICY_BASE_01 = sorted(RESULT_DIR.glob("fast_gain_router_full_val_*base_0p1*_v1.csv"))
if not POLICY_BASE_02:
    POLICY_BASE_02 = sorted(RESULT_DIR.glob("fast_gain_router_full_val_*base_0p2*_v1.csv"))

if not POLICY_BASE_01:
    raise FileNotFoundError("Could not find RFGain base-0.1 selected policy CSV")
if not POLICY_BASE_02:
    raise FileNotFoundError("Could not find RFGain base-0.2 selected policy CSV")

MAIN_POLICY = POLICY_BASE_01[0]
EFF_POLICY = POLICY_BASE_02[0]

print("[INFO] main policy:", MAIN_POLICY)
print("[INFO] efficiency policy:", EFF_POLICY)

main_df = pd.read_csv(MAIN_POLICY)
eff_df = pd.read_csv(EFF_POLICY)
base_df = pd.read_csv(TABLE)

def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_box(img, box, color=(0, 255, 0), thickness=3):
    out = img.copy()
    x, y, w, h = box
    cv2.rectangle(out, (x, y), (x+w, y+h), color, thickness)
    return out

def add_label(img, text):
    h, w = img.shape[:2]
    bar_h = 38
    canvas = np.ones((h + bar_h, w, 3), dtype=np.uint8) * 255
    canvas[bar_h:, :, :] = img
    cv2.putText(
        canvas,
        text,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return canvas

def best_crop_box(img_a, img_b, crop=128, stride=32):
    """Pick crop where two images differ most. Good for highlighting restoration effect."""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    diff = np.abs(gray_a - gray_b)

    h, w = diff.shape
    crop = min(crop, h, w)

    best_score = -1
    best = (0, 0, crop, crop)

    for y in range(0, h - crop + 1, stride):
        for x in range(0, w - crop + 1, stride):
            score = diff[y:y+crop, x:x+crop].mean()
            if score > best_score:
                best_score = score
                best = (x, y, crop, crop)

    return best

def crop_resize(img, box, scale=2):
    x, y, w, h = box
    crop = img[y:y+h, x:x+w]
    return cv2.resize(crop, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

def make_panel(row, out_path, title_prefix=""):
    input_img = read_rgb(row["input_path"])
    tiny_img = read_rgb(row["tiny_path"])
    base_img = read_rgb(row["base_path"])
    gt_img = read_rgb(row["gt_path"])

    if row["pred_label"] == "input":
        ours_img = input_img
    elif row["pred_label"] == "tiny":
        ours_img = tiny_img
    elif row["pred_label"] == "base":
        ours_img = base_img
    else:
        raise ValueError(row["pred_label"])

    # Select crop based on difference between Input and GT.
    # This tends to show degraded region clearly.
    box = best_crop_box(input_img, gt_img, crop=128, stride=32)

    full_imgs = [
        draw_box(input_img, box),
        draw_box(tiny_img, box),
        draw_box(base_img, box),
        draw_box(ours_img, box),
        draw_box(gt_img, box),
    ]

    labels = [
        f"Input {row['psnr_input']:.2f}/{row['lpips_input']:.3f}",
        f"Tiny {row['psnr_tiny']:.2f}/{row['lpips_tiny']:.3f}",
        f"Base {row['psnr_base']:.2f}/{row['lpips_base']:.3f}",
        f"Ours({row['pred_label']}) {row['selected_psnr']:.2f}/{row['selected_lpips']:.3f}",
        "GT",
    ]

    full_panels = [add_label(im, lab) for im, lab in zip(full_imgs, labels)]

    zoom_imgs = [
        crop_resize(input_img, box),
        crop_resize(tiny_img, box),
        crop_resize(base_img, box),
        crop_resize(ours_img, box),
        crop_resize(gt_img, box),
    ]
    zoom_panels = [add_label(im, "zoom") for im in zoom_imgs]

    # Make zoom width match full panel width by padding/resizing.
    # Full patch is usually 512 wide. Zoom crop is 256 wide. Resize zoom to same width for alignment.
    target_w = full_panels[0].shape[1]
    aligned_zoom = []
    for z in zoom_panels:
        zh, zw = z.shape[:2]
        z2 = cv2.resize(z, (target_w, int(zh * target_w / zw)), interpolation=cv2.INTER_CUBIC)
        aligned_zoom.append(z2)

    top = np.concatenate(full_panels, axis=1)
    bottom = np.concatenate(aligned_zoom, axis=1)

    # Pad heights if needed
    if bottom.shape[1] != top.shape[1]:
        raise RuntimeError("width mismatch")

    canvas = np.concatenate([top, bottom], axis=0)

    if title_prefix:
        title_h = 42
        title = np.ones((title_h, canvas.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(title, title_prefix, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        canvas = np.concatenate([title, canvas], axis=0)

    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def save_group(name, df_group, n=4, title_prefix=""):
    out_dir = FIG_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (_, row) in enumerate(df_group.head(n).iterrows(), start=1):
        safe_name = f"{i:02d}_{row['task']}_{row['difficulty']}_{Path(row['filename']).stem}_{row['pred_label']}.png"
        out_path = out_dir / safe_name
        make_panel(row, out_path, title_prefix=title_prefix)

        rows.append({
            "group": name,
            "rank": i,
            "task": row["task"],
            "difficulty": row["difficulty"],
            "filename": row["filename"],
            "pred_label": row["pred_label"],
            "psnr_input": row["psnr_input"],
            "psnr_tiny": row["psnr_tiny"],
            "psnr_base": row["psnr_base"],
            "selected_psnr": row["selected_psnr"],
            "lpips_input": row["lpips_input"],
            "lpips_tiny": row["lpips_tiny"],
            "lpips_base": row["lpips_base"],
            "selected_lpips": row["selected_lpips"],
            "gap_selected_vs_base": row["selected_psnr"] - row["psnr_base"],
            "figure": str(out_path),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("[SAVED GROUP]", name, "n=", len(summary))
    return summary

# A. Skip: main policy selected input, input is strong and restoration not needed
skip = main_df[
    (main_df["pred_label"] == "input") &
    (main_df["psnr_input"] > 24)
].copy()
skip["input_advantage"] = skip["psnr_input"] - skip[["psnr_tiny", "psnr_base"]].max(axis=1)
skip = skip.sort_values(["input_advantage", "psnr_input"], ascending=[False, False])

# B. Tiny sufficient: selected tiny, tiny improves input, base-tiny gap small
tiny_sufficient = main_df[
    (main_df["pred_label"] == "tiny") &
    ((main_df["psnr_tiny"] - main_df["psnr_input"]) > 1.0) &
    ((main_df["psnr_base"] - main_df["psnr_tiny"]).abs() < 0.15)
].copy()
tiny_sufficient["score"] = (tiny_sufficient["psnr_tiny"] - tiny_sufficient["psnr_input"]) - (tiny_sufficient["psnr_base"] - tiny_sufficient["psnr_tiny"]).abs()
tiny_sufficient = tiny_sufficient.sort_values("score", ascending=False)

# C. Base needed: selected base, base clearly better than tiny
base_needed = main_df[
    (main_df["pred_label"] == "base") &
    ((main_df["psnr_base"] - main_df["psnr_tiny"]) > 0.7) &
    ((main_df["psnr_base"] - main_df["psnr_input"]) > 1.0)
].copy()
base_needed = base_needed.sort_values("gain_base_vs_tiny", ascending=False)

# D. Efficiency: base-0.2, selected lower-cost action with around -0.1 to -0.2 dB gap vs base
eff = eff_df[eff_df["pred_label"].isin(["input", "tiny"])].copy()
eff["gap_selected_vs_base"] = eff["selected_psnr"] - eff["psnr_base"]
eff["target_gap"] = (eff["gap_selected_vs_base"] + 0.15).abs()
eff = eff[
    (eff["gap_selected_vs_base"] < -0.05) &
    (eff["gap_selected_vs_base"] > -0.30)
].sort_values("target_gap", ascending=True)

summaries = []
summaries.append(save_group("A_skip_input", skip, n=4, title_prefix="Skip: input is already preferable"))
summaries.append(save_group("B_tiny_sufficient", tiny_sufficient, n=4, title_prefix="Tiny sufficient: Tiny is close to Base"))
summaries.append(save_group("C_base_needed", base_needed, n=4, title_prefix="Base needed: Base gives clear gain"))
summaries.append(save_group("D_efficiency_0p2", eff, n=4, title_prefix="Efficiency mode: small PSNR loss for lower cost"))

all_summary = pd.concat(summaries, ignore_index=True)
all_summary.to_csv(FIG_DIR / "summary_final_qualitative_v1.csv", index=False)

print()
print("[DONE]")
print("FIG_DIR:", FIG_DIR)
print("SUMMARY:", FIG_DIR / "summary_final_qualitative_v1.csv")
print(all_summary)
