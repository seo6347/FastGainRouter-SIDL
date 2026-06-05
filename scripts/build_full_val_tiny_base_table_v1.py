from pathlib import Path
import math
import csv
import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception as e:
    raise ImportError("scikit-image is required. Try: pip install scikit-image") from e

ROOT = Path("/data/donggyun/SIDL_project")
DATA_ROOT = ROOT / "data_full"
RESULTS_ROOT = ROOT / "baseline" / "NAFNet" / "results"
OUT_DIR = ROOT / "my_experiment" / "skip_restore_v2_tiny" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["clean", "dust", "finger", "mixed", "scratch", "water"]
DIFFS = ["easy", "medium", "hard"]


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


def result_dir(branch: str, task: str, diff: str) -> Path:
    if branch == "tiny":
        root = RESULTS_ROOT / f"NAFNet-SIDL-v2-tiny-lr1e3-{task}-{diff}-val"
    elif branch == "base":
        root = RESULTS_ROOT / f"NAFNet-SIDL-v2-base-width8-{task}-{diff}-val"
    else:
        raise ValueError(branch)

    p = root / "visualization" / f"sidl-{task}-{diff}-val"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def find_output(result_dir: Path, filename: str) -> Path:
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

    raise FileNotFoundError(f"Output not found: {filename} in {result_dir}")


rows = []
missing = []

for task in TASKS:
    for diff in DIFFS:
        input_dir = DATA_ROOT / "val" / task / diff / "input"
        gt_dir = DATA_ROOT / "val" / task / diff / "target"

        tiny_dir = result_dir("tiny", task, diff)
        base_dir = result_dir("base", task, diff)

        input_files = sorted(input_dir.glob("*.png"))

        print(f"[{task}/{diff}] n={len(input_files)}")
        print(f"  tiny_dir={tiny_dir}")
        print(f"  base_dir={base_dir}")

        for input_path in input_files:
            name = input_path.name
            gt_path = gt_dir / name

            if not gt_path.exists():
                missing.append((task, diff, name, "gt_missing"))
                continue

            try:
                tiny_path = find_output(tiny_dir, name)
                base_path = find_output(base_dir, name)
            except FileNotFoundError as e:
                missing.append((task, diff, name, str(e)))
                continue

            inp = read_img(input_path)
            gt = read_img(gt_path)
            tiny = read_img(tiny_path)
            base = read_img(base_path)

            psnr_input = calc_psnr(inp, gt)
            psnr_tiny = calc_psnr(tiny, gt)
            psnr_base = calc_psnr(base, gt)

            ssim_input = calc_ssim(inp, gt)
            ssim_tiny = calc_ssim(tiny, gt)
            ssim_base = calc_ssim(base, gt)

            psnr_values = {
                "input": psnr_input,
                "tiny": psnr_tiny,
                "base": psnr_base,
            }
            ssim_values = {
                "input": ssim_input,
                "tiny": ssim_tiny,
                "base": ssim_base,
            }

            oracle_psnr = max(psnr_values, key=psnr_values.get)
            oracle_ssim = max(ssim_values, key=ssim_values.get)

            rows.append({
                "task": task,
                "difficulty": diff,
                "filename": name,
                "input_path": str(input_path),
                "gt_path": str(gt_path),
                "tiny_path": str(tiny_path),
                "base_path": str(base_path),

                "psnr_input": psnr_input,
                "psnr_tiny": psnr_tiny,
                "psnr_base": psnr_base,
                "psnr_oracle_tiny_base": max(psnr_tiny, psnr_base),
                "psnr_oracle_input_tiny_base": max(psnr_input, psnr_tiny, psnr_base),
                "oracle_label_psnr": oracle_psnr,
                "gain_tiny_vs_input": psnr_tiny - psnr_input,
                "gain_base_vs_tiny": psnr_base - psnr_tiny,
                "gain_base_vs_input": psnr_base - psnr_input,

                "ssim_input": ssim_input,
                "ssim_tiny": ssim_tiny,
                "ssim_base": ssim_base,
                "ssim_oracle_tiny_base": max(ssim_tiny, ssim_base),
                "ssim_oracle_input_tiny_base": max(ssim_input, ssim_tiny, ssim_base),
                "oracle_label_ssim": oracle_ssim,
            })

out_csv = OUT_DIR / "full_val_tiny_base_policy_table_v1.csv"

fieldnames = [
    "task", "difficulty", "filename",
    "input_path", "gt_path", "tiny_path", "base_path",
    "psnr_input", "psnr_tiny", "psnr_base",
    "psnr_oracle_tiny_base", "psnr_oracle_input_tiny_base", "oracle_label_psnr",
    "gain_tiny_vs_input", "gain_base_vs_tiny", "gain_base_vs_input",
    "ssim_input", "ssim_tiny", "ssim_base",
    "ssim_oracle_tiny_base", "ssim_oracle_input_tiny_base", "oracle_label_ssim",
]

with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print()
print("[OK] saved:", out_csv)
print("[OK] rows:", len(rows))
print("[OK] missing:", len(missing))

if missing:
    missing_csv = OUT_DIR / "full_val_tiny_base_policy_table_missing_v1.csv"
    with missing_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "difficulty", "filename", "reason"])
        writer.writerows(missing)
    print("[WARN] missing saved:", missing_csv)
    print("[WARN] first missing:", missing[:10])


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


print()
print("[Overall PSNR Policy Comparison]")
print("Always Input:", mean([r["psnr_input"] for r in rows]))
print("Always Tiny: ", mean([r["psnr_tiny"] for r in rows]))
print("Always Base: ", mean([r["psnr_base"] for r in rows]))
print("Oracle Tiny/Base:", mean([r["psnr_oracle_tiny_base"] for r in rows]))
print("Oracle Input/Tiny/Base:", mean([r["psnr_oracle_input_tiny_base"] for r in rows]))

print()
print("[Overall SSIM Policy Comparison]")
print("Always Input:", mean([r["ssim_input"] for r in rows]))
print("Always Tiny: ", mean([r["ssim_tiny"] for r in rows]))
print("Always Base: ", mean([r["ssim_base"] for r in rows]))
print("Oracle Tiny/Base:", mean([r["ssim_oracle_tiny_base"] for r in rows]))
print("Oracle Input/Tiny/Base:", mean([r["ssim_oracle_input_tiny_base"] for r in rows]))

print()
print("[Oracle label ratio by PSNR]")
labels = [r["oracle_label_psnr"] for r in rows]
for label in ["input", "tiny", "base"]:
    print(label, labels.count(label), labels.count(label) / len(labels) if labels else 0)

print()
print("[Gain Summary]")
print("mean gain tiny vs input:", mean([r["gain_tiny_vs_input"] for r in rows]))
print("mean gain base vs tiny:", mean([r["gain_base_vs_tiny"] for r in rows]))
print("mean gain base vs input:", mean([r["gain_base_vs_input"] for r in rows]))

print()
print("[By task/difficulty]")
for task in TASKS:
    for diff in DIFFS:
        g = [r for r in rows if r["task"] == task and r["difficulty"] == diff]
        if not g:
            continue
        labels = [r["oracle_label_psnr"] for r in g]
        print(
            f"{task},{diff},n={len(g)},"
            f"input={mean([r['psnr_input'] for r in g]):.4f},"
            f"tiny={mean([r['psnr_tiny'] for r in g]):.4f},"
            f"base={mean([r['psnr_base'] for r in g]):.4f},"
            f"oracle={mean([r['psnr_oracle_input_tiny_base'] for r in g]):.4f},"
            f"ratio(input/tiny/base)="
            f"{labels.count('input')/len(g):.3f}/"
            f"{labels.count('tiny')/len(g):.3f}/"
            f"{labels.count('base')/len(g):.3f}"
        )
