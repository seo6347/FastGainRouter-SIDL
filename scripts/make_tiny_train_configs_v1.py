from pathlib import Path

ROOT = Path("/data/donggyun/SIDL_project")
DATA_ROOT = ROOT / "data_full"
CONFIG_DIR = ROOT / "my_experiment" / "skip_restore_v2_tiny" / "configs" / "train"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "tiny_lr1e3": {
        "exp_name": "NAFNet-SIDL-all-tiny-width4-lr1e3-es20000-v1",
        "width": 4,
        "enc_blk_nums": [1, 1, 1, 1],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
        "batch_size": 16,
        "gt_size": 64,
        "total_iter": 20000,
        "val_freq": 1000,
        "save_checkpoint_freq": 1000,
        "lr": "1e-3",
    },
    "tiny_lr5e4": {
        "exp_name": "NAFNet-SIDL-all-tiny-width4-lr5e4-es20000-v1",
        "width": 4,
        "enc_blk_nums": [1, 1, 1, 1],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
        "batch_size": 16,
        "gt_size": 64,
        "total_iter": 20000,
        "val_freq": 1000,
        "save_checkpoint_freq": 1000,
        "lr": "5e-4",
    },
}

def list_to_yaml(xs):
    return "[" + ", ".join(str(x) for x in xs) + "]"

for key, cfg in MODELS.items():
    yml = f"""\
# Tiny NAFNet train config: {key}

name: {cfg["exp_name"]}
model_type: ImageRestorationModel
scale: 1
num_gpu: 1
manual_seed: 42

datasets:
  train:
    name: sidl-all-train-inner
    type: PairedImageDataset
    dataroot_gt: {DATA_ROOT}/train/all_custom_train_inner/target
    dataroot_lq: {DATA_ROOT}/train/all_custom_train_inner/input
    io_backend:
      type: disk
    gt_size: {cfg["gt_size"]}
    use_flip: true
    use_rot: true
    use_shuffle: true
    num_worker_per_gpu: 4
    batch_size_per_gpu: {cfg["batch_size"]}

  val:
    name: sidl-all-es-val-inner
    type: PairedImageDataset
    dataroot_gt: {DATA_ROOT}/train/all_custom_es_val_inner/target
    dataroot_lq: {DATA_ROOT}/train/all_custom_es_val_inner/input
    io_backend:
      type: disk

network_g:
  type: NAFNetLocal
  width: {cfg["width"]}
  enc_blk_nums: {list_to_yaml(cfg["enc_blk_nums"])}
  middle_blk_num: {cfg["middle_blk_num"]}
  dec_blk_nums: {list_to_yaml(cfg["dec_blk_nums"])}

path:
  pretrain_network_g: ~
  strict_load_g: true
  resume_state: ~

train:
  optim_g:
    type: AdamW
    lr: !!float {cfg["lr"]}
    weight_decay: !!float 1e-3
    betas: [0.9, 0.9]
  scheduler:
    type: TrueCosineAnnealingLR
    T_max: {cfg["total_iter"]}
    eta_min: !!float 1e-7
  total_iter: {cfg["total_iter"]}
  warmup_iter: -1
  pixel_opt:
    type: PSNRLoss
    loss_weight: 1
    reduction: mean

val:
  val_freq: {cfg["val_freq"]}
  save_img: false
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 0
      test_y_channel: false

logger:
  print_freq: 100
  save_checkpoint_freq: {cfg["save_checkpoint_freq"]}
  use_tb_logger: false
  wandb:
    project: ~
    resume_id: ~

dist_params:
  backend: nccl
  port: 29500
"""
    out = CONFIG_DIR / f"{key}.yml"
    out.write_text(yml)
    print("[OK]", out)
