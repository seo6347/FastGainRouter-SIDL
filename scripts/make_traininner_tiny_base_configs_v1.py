from pathlib import Path

ROOT = Path("/data/donggyun/SIDL_project")
DATA_ROOT = ROOT / "data_full"
CONFIG_DIR = ROOT / "my_experiment" / "skip_restore_v2_tiny" / "configs" / "traininner_full"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

SET = {
    "dataset_name": "sidl-v2-train-inner",
    "input": DATA_ROOT / "train" / "all_custom_train_inner" / "input",
    "target": DATA_ROOT / "train" / "all_custom_train_inner" / "target",
}

MODELS = {
    "tiny": {
        "prefix": "NAFNet-SIDL-v2-tiny-lr1e3",
        "ckpt": ROOT / "baseline" / "NAFNet" / "experiments" / "NAFNet-SIDL-all-tiny-width4-lr1e3-es20000-v1" / "models" / "best_net_g.pth",
        "width": 4,
        "enc_blk_nums": [1, 1, 1, 1],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
    },
    "base": {
        "prefix": "NAFNet-SIDL-v2-base-width8",
        "ckpt": ROOT / "baseline" / "NAFNet" / "experiments" / "NAFNet-SIDL-all-low-es20000-v1" / "models" / "best_net_g.pth",
        "width": 8,
        "enc_blk_nums": [1, 1, 1, 1],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
    },
}

def list_to_yaml(xs):
    return "[" + ", ".join(str(x) for x in xs) + "]"

if not SET["input"].exists() or not SET["target"].exists():
    raise FileNotFoundError(SET)

for branch, cfg in MODELS.items():
    if not cfg["ckpt"].exists():
        raise FileNotFoundError(cfg["ckpt"])

    exp_name = f"{cfg['prefix']}-traininner-full-v1"

    yml = f"""\
name: {exp_name}
model_type: ImageRestorationModel
scale: 1
num_gpu: 1
manual_seed: 42

datasets:
  test:
    name: {SET["dataset_name"]}
    type: PairedImageDataset
    dataroot_gt: {SET["target"]}
    dataroot_lq: {SET["input"]}
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
    out = CONFIG_DIR / f"{branch}_traininner_full_v1.yml"
    out.write_text(yml)
    print("[OK]", out)
