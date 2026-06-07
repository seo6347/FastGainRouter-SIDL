#!/usr/bin/env python3
"""
make_final_paper_figures_v3.py

Purpose
-------
Regenerate paper-ready figures for the FastGainRouter-SIDL paper.

This script focuses on the remaining visual issues:
1. Rebuild qualitative_grid_v1.png without row-title bands such as
   "A. Skip: Ours selects Input".
2. Keep only the compact zoom crops so the image is readable in a paper.
3. Add larger column labels and PSNR/LPIPS values inside each cell header.
4. Regenerate PSNR-latency and action-ratio plots in a cleaner paper style
   without large plot titles.
5. Copy all final figures into both:
   - /data/donggyun/FastGainRouter-SIDL/figures/paper_ready/
   - /data/donggyun/FastGainRouter-SIDL/paper/figures/

Expected input paths
--------------------
Staging repo:
  /data/donggyun/FastGainRouter-SIDL

Original candidate panels:
  /data/donggyun/SIDL_project/my_experiment/skip_restore_v2_tiny/figures/final_qualitative_candidates_lpips_taskdiverse_v2

Selected qualitative examples:
  A_skip_input/candidate_13_water_easy_Case163_W_030_bal-input_eff-input.png
  B_tiny_sufficient/candidate_05_scratch_medium_Case247_S_031_bal-tiny_eff-tiny.png
  C_base_needed/candidate_12_dust_medium_Case288_D_017_bal-base_eff-base.png
  D_efficiency_small_loss/candidate_24_clean_hard_Case193_C_031_bal-base_eff-tiny.png

Run
---
cd /data/donggyun/FastGainRouter-SIDL
python scripts/make_final_paper_figures_v3.py
"""

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
STAGING_QUAL = STAGING / "figures" / "final_qualitative_v1"

# Optional original output mirror
ORIG_FINAL_QUAL = ORIG / "my_experiment/skip_restore_v2_tiny/figures/final_qualitative_v1"

for p in [OUT, PAPER_FIG, STAGING_QUAL, ORIG_FINAL_QUAL]:
    p.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Matplotlib paper style
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.55,
})

COL_INPUT = "#4C78A8"
COL_TINY = "#F58518"
COL_BASE = "#54A24B"
COL_OURS = "#B279A2"
COL_GRAY = "#666666"


# ------------------------------------------------------------
# Selected qualitative examples
# ------------------------------------------------------------
SELECTED = [
    {
        "row": "A",
        "case_name": "Skip / Input branch",
        "group": "A_skip_input",
        "filename": "candidate_13_water_easy_Case163_W_030_bal-input_eff-input.png",
        "ours_source": "balanced",  # balanced and efficient are both input
    },
    {
        "row": "B",
        "case_name": "Tiny sufficient",
        "group": "B_tiny_sufficient",
        "filename": "candidate_05_scratch_medium_Case247_S_031_bal-tiny_eff-tiny.png",
        "ours_source": "balanced",  # balanced and efficient are both tiny
    },
    {
        "row": "C",
        "case_name": "Base needed",
        "group": "C_base_needed",
        "filename": "candidate_12_dust_medium_Case288_D_017_bal-base_eff-base.png",
        "ours_source": "balanced",  # balanced and efficient are both base
    },
    {
        "row": "D",
        "case_name": "Efficient small-loss",
        "group": "D_efficiency_small_loss",
        "filename": "candidate_24_clean_hard_Case193_C_031_bal-base_eff-tiny.png",
        "ours_source": "efficient",  # show efficient policy output
    },
]

# Candidate panel column layout:
# 0 Input, 1 Tiny, 2 Base, 3 Balanced, 4 Efficient, 5 GT
COL_INDEX = {
    "input": 0,
    "tiny": 1,
    "base": 2,
    "balanced": 3,
    "efficient": 4,
    "gt": 5,
}

CELL_SIZE = 315
LABEL_H = 62
SEP_H = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX


def read_summary_row(group: str, filename: str) -> pd.Series:
    summary_path = CAND_ROOT / group / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    df = pd.read_csv(summary_path)

    if "figure" in df.columns:
        mask = df["figure"].astype(str).map(lambda x: Path(x).name == filename)
        if mask.any():
            return df[mask].iloc[0]

    stem = Path(filename).stem
    for _, row in df.iterrows():
        f = str(row.get("filename", ""))
        if f in filename or f.replace(".png", "") in stem:
            return row

    raise RuntimeError(f"Could not find summary row for {group}/{filename}")


def put_text(img, text, xy, scale, color=(0, 0, 0), thickness=1):
    cv2.putText(img, text, xy, FONT, scale, color, thickness, cv2.LINE_AA)


def add_cell_label(img, title, psnr=None, lpips=None, row_letter=None):
    h, w = img.shape[:2]
    canvas = np.ones((h + LABEL_H, w, 3), dtype=np.uint8) * 255
    canvas[LABEL_H:, :, :] = img

    # Optional row letter in the first cell only. Kept compact, not a full row title.
    x0 = 7
    if row_letter is not None:
        put_text(canvas, f"{row_letter}.", (x0, 23), 0.72, (0, 0, 0), 2)
        x0 += 38

    put_text(canvas, title, (x0, 23), 0.66, (0, 0, 0), 2)

    if psnr is not None and lpips is not None:
        put_text(canvas, f"{psnr:.2f} / {lpips:.3f}", (x0, 49), 0.55, (55, 55, 55), 1)

    return canvas


def extract_zoom_cells(panel_path: Path, row: pd.Series, ours_source: str, row_letter: str):
    img = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(panel_path)

    H, W = img.shape[:2]

    # Candidate panel generation format:
    # title_h + full comparison row + zoom comparison row.
    title_h = 48
    remaining_h = H - title_h
    top_h = remaining_h // 2
    bottom_start = title_h + top_h
    bottom = img[bottom_start:, :, :]

    n_cols = 6
    col_w = W // n_cols
    zoom_label_h = 42

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
    for j, (label, col) in enumerate(layout):
        x0 = col * col_w
        x1 = (col + 1) * col_w if col < n_cols - 1 else W
        cell = bottom[zoom_label_h:, x0:x1, :]

        cell = cv2.resize(cell, (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)
        psnr, lpips = metrics[label]
        cell = add_cell_label(cell, label, psnr, lpips, row_letter if j == 0 else None)
        cells.append(cell)

    return cells


def build_qualitative_grid():
    rows = []
    records = []

    for item in SELECTED:
        panel_path = CAND_ROOT / item["group"] / item["filename"]
        if not panel_path.exists():
            raise FileNotFoundError(panel_path)

        row = read_summary_row(item["group"], item["filename"])
        cells = extract_zoom_cells(panel_path, row, item["ours_source"], item["row"])
        row_img = np.concatenate(cells, axis=1)
        rows.append(row_img)

        records.append({
            "row": item["row"],
            "case_name": item["case_name"],
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
            row_img = cv2.resize(
                row_img,
                (target_w, int(row_img.shape[0] * target_w / row_img.shape[1])),
                interpolation=cv2.INTER_AREA,
            )
        parts.append(row_img)
        if i != len(rows) - 1:
            parts.append(sep)

    grid = np.concatenate(parts, axis=0)

    out_png = OUT / "qualitative_grid_v1.png"
    out_csv = OUT / "qualitative_grid_v1.csv"
    cv2.imwrite(str(out_png), grid)
    pd.DataFrame(records).to_csv(out_csv, index=False)

    for dst in [
        PAPER_FIG / "qualitative_grid_v1.png",
        STAGING_QUAL / "qualitative_grid_v1.png",
        ORIG_FINAL_QUAL / "qualitative_grid_v1.png",
    ]:
        shutil.copy2(out_png, dst)

    for dst in [
        STAGING_QUAL / "selected_qualitative_summary_v1.csv",
        ORIG_FINAL_QUAL / "selected_qualitative_summary_v1.csv",
    ]:
        shutil.copy2(out_csv, dst)

    readme = (
        "# Final Qualitative Figure\n\n"
        "This figure is generated for the paper.\n\n"
        "Rows:\n"
        "A. Skip/Input branch: water/easy\n"
        "B. Tiny sufficient: scratch/medium\n"
        "C. Base needed: dust/medium\n"
        "D. Efficient small-loss: clean/hard\n\n"
        "Columns:\n"
        "Input | Tiny | Base | Ours | GT\n\n"
        "Numbers in each cell denote PSNR / LPIPS.\n"
    )
    for dst in [STAGING_QUAL / "README.md", ORIG_FINAL_QUAL / "README.md"]:
        dst.write_text(readme, encoding="utf-8")

    print("[SAVED]", out_png)
    print("[COPIED]", PAPER_FIG / "qualitative_grid_v1.png")
    print("[COPIED]", STAGING_QUAL / "qualitative_grid_v1.png")
    print("[COPIED]", ORIG_FINAL_QUAL / "qualitative_grid_v1.png")


# ------------------------------------------------------------
# Focused PSNR-latency plot
# ------------------------------------------------------------
def build_psnr_latency_plot():
    main_path = STAGING / "results" / "final_compact_main_table_v1.csv"
    main = pd.read_csv(main_path)

    focus_names = ["Always Tiny", "Always Base", "Ours-Balanced", "Ours-Efficient"]
    df = main[main["display"].isin(focus_names)].copy()

    style = {
        "Always Tiny": {"color": COL_TINY, "marker": "o"},
        "Always Base": {"color": COL_BASE, "marker": "s"},
        "Ours-Balanced": {"color": COL_OURS, "marker": "D"},
        "Ours-Efficient": {"color": COL_INPUT, "marker": "^"},
    }

    fig, ax = plt.subplots(figsize=(5.0, 3.45))

    for _, r in df.iterrows():
        name = r["display"]
        ax.scatter(
            r["latency_ms"],
            r["psnr"],
            s=82,
            color=style[name]["color"],
            marker=style[name]["marker"],
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )

    # ------------------------------------------------------------
    # PSNR-latency improvement arrows
    # ------------------------------------------------------------
    # The arrows are intentionally subtle: they indicate movement in the
    # PSNR-latency plane without visually overstating the small PSNR gaps.
    point_map = {
        r["display"]: (float(r["latency_ms"]), float(r["psnr"]))
        for _, r in df.iterrows()
    }

    def add_tradeoff_arrow(start_name, end_name, label_xy=None):
        start = point_map[start_name]
        end = point_map[end_name]

        delta_psnr = end[1] - start[1]
        rel_latency_reduction = (start[0] - end[0]) / start[0] * 100.0
        label = f"+{delta_psnr:.3f} dB, -{rel_latency_reduction:.1f}%"

        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.1,
                linestyle="--",
                color="0.35",
                shrinkA=6,
                shrinkB=6,
                mutation_scale=12,
            ),
            zorder=2,
        )

        # Place labels away from the dashed arrow paths so they do not cover the arrows.
        # Coordinates are in data units: (latency ms/img, PSNR dB).
        if label_xy is None:
            label_xy = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)

        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            fontsize=8,
            color="0.25",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
            zorder=4,
        )

    # Always Base -> Ours-Balanced: place label below the arrow path.
    add_tradeoff_arrow(
        "Always Base",
        "Ours-Balanced",
        label_xy=(111.2, 23.095),
    )

    # Always Tiny -> Ours-Efficient: place label to the lower-left of the arrow path.
    add_tradeoff_arrow(
        "Always Tiny",
        "Ours-Efficient",
        label_xy=(103, 23.01),
    )

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
            zorder=4,
        )

    ax.set_xlabel("Component-based latency (ms / image)")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmin = df["latency_ms"].min() - 3
    xmax = df["latency_ms"].max() + 5
    ymin = df["psnr"].min() - 0.04
    ymax = df["psnr"].max() + 0.04
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

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
# Horizontal action ratio plot
# ------------------------------------------------------------
def build_action_ratio_plot():
    main_path = STAGING / "results" / "final_compact_main_table_v1.csv"
    main = pd.read_csv(main_path)

    ours = main[main["display"].isin(["Ours-Balanced", "Ours-Efficient"])].copy()
    ours = ours.set_index("display").loc[["Ours-Balanced", "Ours-Efficient"]].reset_index()

    fig, ax = plt.subplots(figsize=(5.25, 2.25))
    y = np.arange(len(ours))
    left = np.zeros(len(ours))

    segments = [
        ("Input", "input_ratio", COL_INPUT),
        ("Tiny", "tiny_ratio", COL_TINY),
        ("Base", "base_ratio", COL_BASE),
    ]

    for label, col, color in segments:
        vals = ours[col].astype(float).values
        ax.barh(y, vals, left=left, color=color, edgecolor="white", height=0.52, label=label)

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


def write_status():
    readme = OUT / "README.md"
    readme.write_text(
        "# Paper-ready Figures v3\n\n"
        "Generated figures:\n\n"
        "- qualitative_grid_v1.png: compact qualitative crop grid without row-title bands, with larger PSNR/LPIPS labels.\n"
        "- fig_psnr_latency_focus_v2.png/pdf: focused PSNR-latency plot with subtle improvement arrows.\n"
        "- fig_action_ratio_horizontal_v2.png/pdf: horizontal branch usage plot with percentages.\n\n"
        "Canonical copies are stored in `paper/figures/`.\n",
        encoding="utf-8",
    )


def main():
    if not STAGING.exists():
        raise FileNotFoundError(STAGING)
    if not CAND_ROOT.exists():
        raise FileNotFoundError(CAND_ROOT)

    # build_qualitative_grid()
    build_psnr_latency_plot()
    build_action_ratio_plot()
    write_status()

    print()
    print("[DONE] final paper figures generated.")
    print("[OUT]", OUT)
    print("[PAPER_FIG]", PAPER_FIG)


if __name__ == "__main__":
    main()
