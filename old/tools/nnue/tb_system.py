#!/usr/bin/env python3
"""System health → TensorBoard, sampled every 15s (OOM early-warning included).

    uv run --no-project --with tensorboardX python tools/nnue/tb_system.py \
        data/nnue/tb/d47-v1/system

Emits: gpu/util_pct, gpu/mem_used_mb, gpu/temp_c, gpu/power_w,
       sys/ram_available_gb (the OOM canary — the D47 trainer dies when this
       hits ~0), sys/trainer_alive (1/0).
X-axis is wall-relative minutes since the sampler started.
"""
import subprocess
import sys
import time

from tensorboardX import SummaryWriter


def main():
    writer = SummaryWriter(sys.argv[1])
    t0 = time.time()
    step = 0
    while True:
        step_min = int((time.time() - t0) / 15)
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().split(", ")
            writer.add_scalar("gpu/util_pct", float(q[0]), step_min)
            writer.add_scalar("gpu/mem_used_mb", float(q[1]), step_min)
            writer.add_scalar("gpu/temp_c", float(q[2]), step_min)
            writer.add_scalar("gpu/power_w", float(q[3]), step_min)
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as fh:
                for ln in fh:
                    if ln.startswith("MemAvailable:"):
                        writer.add_scalar(
                            "sys/ram_available_gb", int(ln.split()[1]) / 1048576, step_min
                        )
                        break
            alive = subprocess.run(
                ["pgrep", "-f", "d40_features"], capture_output=True
            ).returncode == 0
            writer.add_scalar("sys/trainer_alive", 1.0 if alive else 0.0, step_min)
        except Exception:
            pass
        writer.flush()
        step += 1
        time.sleep(15)


if __name__ == "__main__":
    main()
