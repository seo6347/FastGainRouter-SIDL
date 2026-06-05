from pathlib import Path
import sys
import time
import statistics
import torch
import pandas as pd

ROOT = Path("/data/donggyun/SIDL_project")
NAFNET_DIR = ROOT / "baseline" / "NAFNet"
sys.path.insert(0, str(NAFNET_DIR))

try:
    from basicsr.models.archs.NAFNet_arch import NAFNetLocal
except Exception:
    from basicsr.archs.nafnet_arch import NAFNetLocal

OUT = ROOT / "my_experiment" / "skip_restore_v2_tiny" / "results" / "pure_forward_latency_v1.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512
BATCH_SIZE = 1
WARMUP = 50
N_RUNS = 300

MODELS = {
    "tiny_width4": {
        "ckpt": ROOT / "baseline" / "NAFNet" / "experiments" / "NAFNet-SIDL-all-tiny-width4-lr1e3-es20000-v1" / "models" / "best_net_g.pth",
        "width": 4,
        "middle_blk_num": 1,
    },
    "base_width8": {
        "ckpt": ROOT / "baseline" / "NAFNet" / "experiments" / "NAFNet-SIDL-all-low-es20000-v1" / "models" / "best_net_g.pth",
        "width": 8,
        "middle_blk_num": 1,
    },
    "old_high_width16": {
        "ckpt": ROOT / "baseline" / "NAFNet" / "experiments" / "NAFNet-SIDL-all-high-es20000-v1" / "models" / "best_net_g.pth",
        "width": 16,
        "middle_blk_num": 4,
    },
}


def load_state_dict(path: Path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ["params", "params_ema", "state_dict", "net_g"]:
            if key in obj:
                return obj[key]
    return obj


def build_model(width: int, middle_blk_num: int, ckpt: Path):
    model = NAFNetLocal(
        width=width,
        enc_blk_nums=[1, 1, 1, 1],
        middle_blk_num=middle_blk_num,
        dec_blk_nums=[1, 1, 1, 1],
    )
    sd = load_state_dict(ckpt)
    model.load_state_dict(sd, strict=True)
    model = model.to(DEVICE).eval()
    return model


@torch.no_grad()
def measure_model(name, cfg):
    ckpt = cfg["ckpt"]
    if not ckpt.exists():
        print(f"[SKIP] missing checkpoint: {ckpt}")
        return None

    print()
    print("=" * 80)
    print("[MODEL]", name)
    print("[CKPT]", ckpt)

    model = build_model(cfg["width"], cfg["middle_blk_num"], ckpt)

    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    # Warmup
    for _ in range(WARMUP):
        _ = model(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(N_RUNS):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    row = {
        "model": name,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "warmup": WARMUP,
        "runs": N_RUNS,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p90_ms": sorted(times)[int(0.9 * len(times))],
        "checkpoint_mb": ckpt.stat().st_size / 1024 / 1024,
    }

    print(row)
    return row


rows = []
for name, cfg in MODELS.items():
    row = measure_model(name, cfg)
    if row is not None:
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print()
print("[DONE] saved:", OUT)
print(df)
