"""Export a RAW checkpoint to a bullet quant.bin under a run-config's arch.

The training driver only writes quant checkpoints as a side effect of a
superbatch boundary, so a freshly assembled warm net (net2net expansion
output) has no deployable form until sb1 lands. That blocks the pre-launch
NPS smoke, which needs a packed net for the target architecture *before*
committing GPU-days to the run. This reconstructs the exact arch from the
run config (same keys pump_train reads) and writes the quant stream.

    uv run python jax/tools/export_warm_quant.py <run-config.json> <out.bin> \
        [--raw path]           # default: the config's "warm"

Weights are the warm's own — untrained, so the values are only as good as
the warm. Speed, however, is a property of the architecture and the engine
kernels, which is exactly what the smoke measures.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "jax"))

from archon_jax import checkpoint as ck  # noqa: E402
from archon_jax.spec import ArchSpec  # noqa: E402


def arch_from_spec(spec: dict) -> ArchSpec:
    """Mirror pump_train.main's arch construction (D47U spec-driven head)."""
    full = ArchSpec.full()
    return dataclasses.replace(
        full, factorize=True, psqt=True,
        hl=int(spec.get("arch_hl", full.hl)),
        l2=int(spec.get("arch_l2", full.l2)),
        l3=int(spec.get("arch_l3", full.l3)),
        out_skip=bool(spec.get("arch_out_skip", False)),
        pol_proj=int(spec.get("arch_pol_proj", 0)),
        dis_proj=int(spec.get("arch_dis_proj", 0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--raw", default=None)
    args = ap.parse_args()

    spec = json.loads(Path(args.config).read_text())
    arch = arch_from_spec(spec)
    raw = Path(args.raw) if args.raw else ROOT / spec["warm"]
    print(f"arch hl={arch.hl} l2={arch.l2} l3={arch.l3} "
          f"out_skip={arch.out_skip} pol={arch.pol_proj} dis={arch.dis_proj}")
    print(f"raw  {raw}")
    params = ck.load_warm(raw, arch)
    total = ck.write_quant(args.out, params, arch)
    print(f"wrote {args.out} ({total} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
