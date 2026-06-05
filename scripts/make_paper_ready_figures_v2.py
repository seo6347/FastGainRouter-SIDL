from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
STAGING = Path("/data/donggyun/FastGainRouter-SIDL")
ORIG = Path("/data/donggyun/SIDL_project")

CAND_ROOT = ORIG / "my_experiment/skip_restore_v2_tiny/figures/final_qualitative_candidates_lpips_taskdiverse_v2"

OUT = STAGING / "figures" / "paper_ready"
PAPER_FIG = STAGING / "paper" / "figures"

OUT.mkdir(parents=True, exist_ok=True)
PAPER_FIG.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Matplotlib paper style
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.6,
})

# Colorblind-friendly palette
COL_INPUT = "#4C78A8"
COL_TINY = "#F58518"
COL_BASE = "#54A24B"
COL_OURS = "#B279A2"
COL_GRAY = "#666666"

# ------------------------------------------------------------
# 1. Paper-ready qualitative figure with metrics
# ------------------------------------------------------------

SELECTED = [
    {
        "row": "A",
        "title": "A. Skip: Ours selects Input",
        "group": "A_skip_input",
        "filename": "candidate_13_water_easy_Case163_W_030_bal-input_eff-input.png",
        "ours_source": "balanced",
    },
    {
        "row": "B",
        "title": "B. Tiny sufficient: Ours selects Tiny",
        "group": "B_tiny_sufficient",
        "filename": "candidate_05_scratch_medium_Case247_S_031_bal-tiny_eff-tiny.png",
        "ours_source": "balanced",
    },
    {
        "row": "C",
        "title": "C. Base needed: Ours selects Base",
        "group": "C_base_needed",
        "filename": "candidate_12_dust_medium_Case288_D_017_bal-base_eff-base.png",
        "ours_source": "balanced",
    },
    {
        "row": "D",
        "title": "D. Efficient: Ours selects Tiny",
        "group": "D_efficiency_small_loss",
        "filename": "candidate_24_clean_hard_Case193_C_031_bal-base_eff-tiny.png",
        "ours_source": "efficient",
    },
]

# Original candidate panel column layout:
# 0 Input, 1 Tiny, 2 Base, 3 Balanced, 4 Efficient, 5 GT
COL_INDEX = {
    "input": 0,
    "tiny": 1,
    "base": 2,
    "balanced": 3,
    "efficient": 4,
    "gt": 5,
}

CELL_SIZE = 220
LABEL_H = 42
ROW_TITLE_H = 40
SEP_H = 12
FONT = cv2.FONT_HERSHEY_SIMPLEX

def read_summary_row(group: str, filename: str) -> pd.Series:
    summary_path = CAND_ROOT / group / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    df = pd.read_csv(summary_path)

    # Try matching by figure basename.
    if "figure" in df.columns:
        mask = df["figure"].astype(str).map(lambda x: Path(x).name == filename)
        if mask.any():
            return df[mask].iloc[0]

    # Fallback: match by filename stem components.
    stem = Path(filename).stem
    for _, row in df.iterrows():
        if str(row.get("filename", "")) in filename or str(row.get("filename", "")).replace(".png", "") in stem:
            return row

    raise RuntimeError(f"Could not find summary row for {group}/{filename}")

def add_cell_label(img, title, psnr=None, lpips=None):
    h, w = img.shape[:2]
    canvas = np.ones((h + LABEL_H, w, 3), dtype=np.uint8) * 255
    canvas[LABEL_H:, :, :] = img

    if psnr is None or lpips is None:
        text1 = title
        text2 = ""
    else:
        text1 = title
        text2 = f"{psnr:.2f} / {lpips:.3f}"

    cv2.putText(canvas, text1, (6, 17), FONT, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    if text2:
        cv2.putText(canvas, text2, (6, 35), FONT, 0.43, (50, 50, 50), 1, cv2.LINE_AA)
    return canvas

def add_row_title(row_img, title):
    h, w = row_img.shape[:2]
    title_img = np.ones((ROW_TITLE_H, w, 3), dtype=np.uint8) * 255
    cv2.putText(title_img, title, (6, 26), FONT, 0.70, (0, 0, 0), 2, cv2.LINE_AA)
    return np.concatenate([title_img, row_img], axis=0)

def extract_zoom_cells(panel_path: Path, row: pd.Series, ours_source: str):
    img = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(panel_path)

    H, W = img.shape[:2]
    title_h = 48
    remaining_h = H - title_h
    top_h = remaining_h // 2
    bottom_start = title_h + top_h
    bottom = img[bottom_start:, :, :]

    n_cols = 6
    col_w = W // n_cols
    zoom_label_h = 42

    # Collect metrics.
    metrics = {
        "Input": (float(row["psnr_input"]), float(row["lpips_input"])),
        "Tiny": (float(row["psnr_tiny"]), float(row["lpips_tiny"])),
        "Base": (float(row["psnr_base"]), float(row["lpips_base"])),
        "GT": (None, None),
    }

    if ours_source == "balanced":
        ours_psnr = float(row["balanced_psnr"])
        ours_lpips = float(row["balanced_lpips"])
        ours_col = COL_INDEX["balanced"]
    elif ours_source == "efficient":
        ours_psnr = float(row["efficient_psnr"])
        ours_lpips = float(row["efficient_lpips"])
        ours_col = COL_INDEX["efficient"]
    else:
        raise ValueError(ours_source)

    metrics["Ours"] = (ours_psnr, ours_lpips)

    layout = [
        ("Input", COL_INDEX["input"]),
        ("Tiny", COL_INDEX["tiny"]),
        ("Base", COL_INDEX["base"]),
        ("Ours", ours_col),
        ("GT", COL_INDEX["gt"]),
    ]

    cells = []
    for label, col in layout:
        x0 = col * col_w
        x1 = (col + 1) * col_w if col < n_cols - 1 else W
        cell = bottom[zoom_label_h:, x0:x1, :]
        cell = cv2.resize(cell, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)
        psnr, lpips = metrics[label]
        cell = add_cell_label(cell, label, psnr, lpips)
        cells.append(cell)

    return cells

rows = []
records = []

for item in SELECTED:
    panel_path = CAND_ROOT / item["group"] / item["filename"]
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)

    row = read_summary_row(item["group"], item["filename"])
    cells = extract_zoom_cells(panel_path, row, item["ours_source"])
    row_img = np.concatenate(cells, axis=1)
    row_img = add_row_title(row_img, item["title"])
    rows.append(row_img)

    records.append({
        "row": item["row"],
        "title": item["title"],
        "group": item["group"],
        "filename": item["filename"],
        "ours_source": item["ours_source"],
        "source_panel": str(panel_path),
    })

target_w = rows[0].shape[1]
sep = np.ones((SEP_H, target_w, 3), dtype=np.uint8) * 255

parts = []
for i, row_img in enumerate(rows):
    if row_img.shape[1] != target_w:
        row_img = cv2.resize(row_img, (target_w, int(row_img.shape[0] * target_w / row_img.shape[1])), interpolation=cv2.INTER_AREA)
    parts.append(row_img)
    if i != len(rows) - 1:
        parts.append(sep)

qual_grid = np.concatenate(parts, axis=0)

qual_out = OUT / "qualitative_grid_metrics_v1.png"
cv2.imwrite(str(qual_out), qual_grid)
pd.DataFrame(records).to_csv(OUT / "qualitative_grid_metrics_v1.csv", index=False)

# Copy as canonical paper qualitative figure.
shutil.copy2(qual_out, PAPER_FIG / "qualitative_grid_v1.png")

print("[SAVED]", qual_out)
print("[COPIED]", PAPER_FIG / "qualitative_grid_v1.png")

# ------------------------------------------------------------
# 2. Focused PSNR-latency plot
# ------------------------------------------------------------

main_path = STAGING / "results" / "final_compact_main_table_v1.csv"
main = pd.read_csv(main_path)

# Focus on meaningful restoration trade-off; Always Input is in the table.
focus_names = ["Always Tiny", "Always Base", "Ours-Balanced", "Ours-Efficient"]
df = main[main["display"].isin(focus_names)].copy()

style = {
    "Always Tiny":     {"color": COL_TINY, "marker": "o"},
    "Always Base":     {"color": COL_BASE, "marker": "s"},
    "Ours-Balanced":   {"color": COL_OURS, "marker": "D"},
    "Ours-Efficient":  {"color": COL_INPUT, "marker": "^"},
}

fig, ax = plt.subplots(figsize=(5.0, 3.6))

for _, r in df.iterrows():
    name = r["display"]
    ax.scatter(
        r["latency_ms"], r["psnr"],
        s=80,
        color=style[name]["color"],
        marker=style[name]["marker"],
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
        label=name,
    )

# Manual labels to avoid overlap.
offsets = {
    "Always Tiny": (7, -14),
    "Always Base": (7, 4),
    "Ours-Balanced": (7, 4),
    "Ours-Efficient": (7, 4),
}
for _, r in df.iterrows():
    name = r["display"]
    dx, dy = offsets.get(name, (6, 5))
    ax.annotate(
        name,
        (r["latency_ms"], r["psnr"]),
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=9,
    )

ax.set_xlabel("Component-based latency (ms / image)")
ax.set_ylabel("PSNR (dB)")
ax.grid(True, alpha=0.28)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Tight axes around restoration methods.
xmin = df["latency_ms"].min() - 3
xmax = df["latency_ms"].max() + 5
ymin = df["psnr"].min() - 0.04
ymax = df["psnr"].max() + 0.04
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Small note.
ax.text(
    0.02, 0.03,
    "Always Input omitted for readability\n(reported in Table).",
    transform=ax.transAxes,
    fontsize=7.5,
    color=COL_GRAY,
    va="bottom",
)

fig.tight_layout()
out_png = OUT / "fig_psnr_latency_focus_v2.png"
out_pdf = OUT / "fig_psnr_latency_focus_v2.pdf"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

shutil.copy2(out_png, PAPER_FIG / "fig_psnr_latency_focus_v2.png")
shutil.copy2(out_pdf, PAPER_FIG / "fig_psnr_latency_focus_v2.pdf")

print("[SAVED]", out_png)
print("[SAVED]", out_pdf)

# ------------------------------------------------------------
# 3. Horizontal action ratio plot with percentages
# ------------------------------------------------------------

ours = main[main["display"].isin(["Ours-Balanced", "Ours-Efficient"])].copy()
ours = ours.set_index("display").loc[["Ours-Balanced", "Ours-Efficient"]].reset_index()

fig, ax = plt.subplots(figsize=(5.2, 2.35))

y = np.arange(len(ours))
left = np.zeros(len(ours))

segments = [
    ("Input", "input_ratio", COL_INPUT),
    ("Tiny", "tiny_ratio", COL_TINY),
    ("Base", "base_ratio", COL_BASE),
]

for label, col, color in segments:
    vals = ours[col].astype(float).values
    bars = ax.barh(y, vals, left=left, color=color, edgecolor="white", height=0.52, label=label)

    for i, v in enumerate(vals):
        if v >= 0.08:
            ax.text(
                left[i] + v / 2,
                y[i],
                f"{v*100:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if label != "Tiny" else "black",
            )
    left += vals

ax.set_yticks(y)
ax.set_yticklabels(ours["display"])
ax.set_xlim(0, 1)
ax.set_xlabel("Branch selection ratio")
ax.invert_yaxis()
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.22)

fig.tight_layout()
out_png = OUT / "fig_action_ratio_horizontal_v2.png"
out_pdf = OUT / "fig_action_ratio_horizontal_v2.pdf"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

shutil.copy2(out_png, PAPER_FIG / "fig_action_ratio_horizontal_v2.png")
shutil.copy2(out_pdf, PAPER_FIG / "fig_action_ratio_horizontal_v2.pdf")

print("[SAVED]", out_png)
print("[SAVED]", out_pdf)

# ------------------------------------------------------------
# 4. Status README
# ------------------------------------------------------------

readme = OUT / "README.md"
readme.write_text(
    "# Paper-ready Figures v2\n\n"
    "Generated figures:\n\n"
    "- qualitative_grid_metrics_v1.png: compact qualitative crop grid with PSNR/LPIPS labels.\n"
    "- fig_psnr_latency_focus_v2.png/pdf: focused quality-latency plot excluding Always Input for readability.\n"
    "- fig_action_ratio_horizontal_v2.png/pdf: horizontal branch usage plot with percentages.\n\n"
    "Copied canonical paper figures to `paper/figures/`.\n",
    encoding="utf-8"
)

print("[DONE] Paper-ready figures generated.")
