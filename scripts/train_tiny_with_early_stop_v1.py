import argparse
import os
import re
import json
import time
import shutil
import signal
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--exp_name", required=True)
parser.add_argument("--max_bad", type=int, default=5)
parser.add_argument("--min_delta", type=float, default=0.01)
parser.add_argument("--port", type=int, default=4411)
args = parser.parse_args()

ROOT = Path("/data/donggyun/SIDL_project")
NAFNET_DIR = ROOT / "baseline" / "NAFNet"
LOG_DIR = ROOT / "my_experiment" / "skip_restore_v2_tiny" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

exp_dir = NAFNET_DIR / "experiments" / args.exp_name
model_dir = exp_dir / "models"

log_path = LOG_DIR / f"earlystop_train_{args.name}.log"
state_path = LOG_DIR / f"earlystop_state_{args.name}.json"

cmd = [
    "torchrun",
    "--standalone",
    "--nnodes=1",
    "--nproc_per_node=1",
    f"--master_port={args.port}",
    "basicsr/train.py",
    "-opt",
    args.config,
    "--launcher",
    "pytorch",
]

env = os.environ.copy()
env["PYTHONPATH"] = f"{NAFNET_DIR}:{env.get('PYTHONPATH', '')}"

print("[INFO] name:", args.name)
print("[INFO] config:", args.config)
print("[INFO] exp_name:", args.exp_name)
print("[INFO] log:", log_path)
print("[INFO] command:", " ".join(cmd))

val_re = re.compile(r"Validation .*?# psnr:\s*([-+]?\d+(?:\.\d+)?)")
iter_re = re.compile(r"iter:\s*([0-9,]+)")

best_psnr = float("-inf")
best_val_index = 0
bad_count = 0
val_count = 0
last_iter = None
early_stopped = False

def newest_checkpoint():
    if not model_dir.exists():
        return None
    latest = model_dir / "net_g_latest.pth"
    if latest.exists():
        return latest
    candidates = sorted(model_dir.glob("net_g_*.pth"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]

def save_best_checkpoint(psnr_value):
    ckpt = newest_checkpoint()
    if ckpt is None:
        print("[WARN] no checkpoint found yet")
        return None

    best_path = model_dir / "best_net_g.pth"
    shutil.copy2(ckpt, best_path)

    info = {
        "name": args.name,
        "best_psnr": psnr_value,
        "source_checkpoint": str(ckpt),
        "best_checkpoint": str(best_path),
        "best_val_index": best_val_index,
        "val_count": val_count,
        "bad_count": bad_count,
        "last_iter": last_iter,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    state_path.write_text(json.dumps(info, indent=2))
    print(f"[BEST] copied {ckpt.name} -> best_net_g.pth")
    return best_path

with log_path.open("w") as log_f:
    proc = subprocess.Popen(
        cmd,
        cwd=str(NAFNET_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    try:
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()

            m_iter = iter_re.search(line)
            if m_iter:
                try:
                    last_iter = int(m_iter.group(1).replace(",", ""))
                except Exception:
                    pass

            m_val = val_re.search(line)
            if m_val:
                val_count += 1
                psnr = float(m_val.group(1))

                if psnr > best_psnr + args.min_delta:
                    best_psnr = psnr
                    best_val_index = val_count
                    bad_count = 0
                    msg = f"[EARLYSTOP] val#{val_count}: psnr={psnr:.4f} improved. best={best_psnr:.4f}"
                    print(msg)
                    log_f.write(msg + "\n")
                    save_best_checkpoint(best_psnr)
                else:
                    bad_count += 1
                    msg = (
                        f"[EARLYSTOP] val#{val_count}: psnr={psnr:.4f} no improvement. "
                        f"bad_count={bad_count}/{args.max_bad}, best={best_psnr:.4f}"
                    )
                    print(msg)
                    log_f.write(msg + "\n")

                    if bad_count >= args.max_bad:
                        early_stopped = True
                        print("[EARLYSTOP] patience exhausted. Terminating training process...")
                        log_f.write("[EARLYSTOP] patience exhausted. Terminating training process...\n")
                        os.killpg(proc.pid, signal.SIGTERM)
                        break

        if early_stopped:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("[EARLYSTOP] SIGTERM timeout. Sending SIGKILL...")
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        else:
            proc.wait()

    finally:
        ret = proc.poll()
        print("[INFO] process return code:", ret)
        log_f.write(f"[INFO] process return code: {ret}\n")

summary = {
    "name": args.name,
    "best_psnr": best_psnr,
    "best_val_index": best_val_index,
    "bad_count": bad_count,
    "val_count": val_count,
    "last_iter": last_iter,
    "early_stopped": early_stopped,
    "exp_dir": str(exp_dir),
    "model_dir": str(model_dir),
    "state_path": str(state_path),
}

state_path.write_text(json.dumps(summary, indent=2))

print("[SUMMARY]")
print(json.dumps(summary, indent=2))

if early_stopped:
    raise SystemExit(0)

if best_psnr > float("-inf") and not (model_dir / "best_net_g.pth").exists():
    save_best_checkpoint(best_psnr)
