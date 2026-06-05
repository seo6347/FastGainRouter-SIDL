from pathlib import Path

ROOT = Path("/data/donggyun/SIDL_project")
DATA_ROOT = ROOT / "data_full"
CONFIG_DIR = ROOT / "my_experiment" / "skip_restore_v2_tiny" / "configs" / "eval"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["clean", "dust", "finger", "mixed", "scratch", "water"]
DIFFS = ["easy", "medium", "hard"]

MODELS = {
    "tiny": {
        "exp_name_prefix": "NAFNet-SIDL-v2-tiny-lr1e3",
        "ckpt": ROOT / "baseline" / "NAFNet" / "experiments" / "NAFNet-SIDL-all-tiny-width4-lr1e3-es20000-v1" / "models" / "best_net_g.pth",
        "width": 4,
        "enc_blk_nums": [1, 1, 1, 1],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
    },
    "base": {
        "exp_name_prefix": "NAFNet-SIDL-v2-base-width8",
        "ckpt": ROOT / "baseline" / "NAFNet" / "experiments" / "NAFNet-SIDL-all-low-es20000-v1" / "models" / "best_net_g.pth",
        "width": 8,
        "enc_blk_nums": [1, 1, 1, 1],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
    },
}

def list_to_yaml(xs):
    return "[" + ", ".join(str(x) for x in xs) + "]"

for branch, cfg in MODELS.items():
    if not cfg["ckpt"].exists():
        raise FileNotFoundError(cfg["ckpt"])

    for task in TASKS:
        for diff in DIFFS:
            lq = DATA_ROOT / "val" / task / diff / "input"
            gt = DATA_ROOT / "val" / task / diff / "target"

            if not lq.exists() or not gt.exists():
                print("[SKIP]", task, diff)
                continue

            exp_name = f"{cfg['exp_name_prefix']}-{task}-{diff}-val"
            dataset_name = f"sidl-{task}-{diff}-val"

            yml = f"""\
name: {exp_name}
model_type: ImageRestorationModel
scale: 1
num_gpu: 1
manual_seed: 42

datasets:
  test:
    name: {dataset_name}
    type: PairedImageDataset
    dataroot_gt: {gt}
    dataroot_lq: {lq}
    io_backend:
      type: disk

network_g:
  type: NAFNetLocal
  width: {cfg["width"]}
  enc_blk_nums: {list_to_yaml(cfg["enc_blk_nums"])}
  middle_blk_num: {cfg["middle_blk_num"]}
  dec_blk_nums: {list_to_yaml(cfg["dec_blk_nums"])}

path:
  pretrain_network_g: {cfg["ckpt"]}
  strict_load_g: true
  resume_state: ~

val:
  save_img: true
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 0
      test_y_channel: false

dist_params:
  backend: nccl
  port: 29500
"""
            out = CONFIG_DIR / f"{branch}_{task}_{diff}_val.yml"
            out.write_text(yml)
            print("[OK]", out)

print("[DONE] Tiny/Base official val configs generated.")
