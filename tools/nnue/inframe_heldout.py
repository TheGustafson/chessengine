"""In-frame heldout — THE health instrument for any retargeted (target_offset/
scale changed) run. The trainer's built-in heldout deliberately uses
TrainConfig() DEFAULTS (transform-invariant warm gate), so it MUST drift UP
while a net retargets — three rising readings on D47S nearly tripped a false
alarm (2026-07-26 ~06:5x). This maps the same heldout bin through THE RUN'S
transform instead. Usage:
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
    /bin/sh -c 'cd jax && JAX_PLATFORMS=cpu uv run --no-project \
      --with numpy==2.5.1 --with jax==0.11.0 --with optax==0.2.8 \
      python ../tools/nnue/inframe_heldout.py 10 50 120'

In-frame heldout for D47S: same heldout bin, but targets mapped through
THE RUN'S transform (scale 76.9, offset 163) — the instrument the default
heldout is NOT (it uses the old frame and must rise under retargeting)."""
import sys, json
sys.path.insert(0, "/home/nick/projects/chessengine/jax")
import numpy as np
from pathlib import Path
from archon_jax import checkpoint as ck
from archon_jax.spec import ArchSpec, TrainConfig, RECORD_DTYPE
from archon_jax.features import map_batch
from archon_jax.model import forward
import jax, dataclasses

spec = json.load(open("/home/nick/projects/chessengine/tools/nnue/runs/v15_refine.json"))
arch = dataclasses.replace(ArchSpec.full(), factorize=True, psqt=True)
cfg = TrainConfig()
HELDOUT = Path("/home/nick/projects/chessengine/data/nnue/heldout/uho-band0-5.bin")
recs = np.memmap(HELDOUT, dtype=RECORD_DTYPE, mode="r")
bs = 2048
n = (len(recs)//bs)*bs
for sb in sys.argv[1:]:
    p = Path(f"/home/nick/projects/chessengine/data/nnue/experiments/d47s-v15/d47s-v15-sb{sb}-ema-raw.bin")
    params = ck.load_warm(p, arch)
    jp = {k: jax.numpy.asarray(v) for k, v in params.items()}
    tot_new = 0.0
    for off in range(0, n, bs):
        b = map_batch(np.array(recs[off:off+bs]), wdl=cfg.wdl,
                      score_multiplier=cfg.score_multiplier,
                      scale=float(spec["scale"]),
                      target_offset=float(spec["target_offset"]), policy=False)
        out = forward(jp, arch, {k: np.asarray(v) for k, v in b.items()}, "reference")[0]
        tot_new += float(np.sum((np.asarray(jax.nn.sigmoid(out)) - b["target"])**2))
    print(f"sb{sb}: in-frame heldout {tot_new/n:.6f}")
