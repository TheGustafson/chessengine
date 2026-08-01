#!/usr/bin/env python3
"""Emit the SF-net-teacher probe run spec from the PLAIN run's own spec.

`tools/nnue/runs/` is gitignored, so the spec itself cannot be the artifact —
this generator is. It takes the live plain continuation's json and changes
exactly four things, so "same everything except the teacher" is enforced by
construction rather than by a careful hand-edit:

  net_id / name       a separate checkpoint namespace (the live run's files
                      must not be overwritten)
  distill_alpha       0.0 -> 0.4     the owner's 60/40
  teacher_warm/_arch  the converted Stockfish net
  teacher_kp          the per-corpus (k, p) table, read straight out of
                      docs/research/sf-teacher-calibration-2026-08-01.json
                      and ORDER-CHECKED against the mix by corpus name

plus the schedule window (superbatches, export_every) and an lr_ramp that is a
verbatim SLICE of the plain run's own ramp, so the two arms see identical
learning rates at identical absolute superbatches.

Launched with `resume_sb = <warm_sb>`, which additionally makes
`skip_batches = bps * warm_sb` — so the probe consumes the SAME batches, in
the same order, that the plain run consumed over the same superbatches. The
arm is paired at the batch level, not merely at the corpus level.

    python3 tools/nnue/make_sfblend_probe.py \
        --base tools/nnue/runs/d49_x500_1280.json \
        --out  tools/nnue/runs/d49_sfblend_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CALIB = ROOT / "docs/research/sf-teacher-calibration-2026-08-01.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="tools/nnue/runs/d49_x500_1280.json")
    ap.add_argument("--out", default="tools/nnue/runs/d49_sfblend_probe.json")
    ap.add_argument("--net-id", default="d49sfb")
    ap.add_argument("--warm-sb", type=int, default=1630)
    ap.add_argument("--probe-sb", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--export-every", type=int, default=2)
    ap.add_argument("--teacher",
                    default="data/nnue/teachers/sf-eca43a97-teacher-raw.bin")
    ap.add_argument("--warm",
                    default="data/nnue/experiments/d49-1280/"
                            "d49lrc3-sb1630-ema-raw.bin")
    a = ap.parse_args()

    spec = json.loads((ROOT / a.base).read_text())
    calib = json.loads(CALIB.read_text())
    rows = calib["deliverable_3run_mean"]

    # ORDER CHECK. per_corpus_target and teacher_kp are both POSITIONAL against
    # `mix`, and a silent off-by-one there produces a plausible-looking run that
    # is simply wrong — the exact failure mode that sank D48Y2. The calibration
    # rows carry their corpus name, so verify rather than trust the ordering.
    mix_names = [Path(p).name.replace(".shuf.bin", "") for p, _ in spec["mix"]]
    row_names = [r["corpus"] for r in sorted(rows, key=lambda r: r["idx"])]
    if mix_names != row_names:
        raise SystemExit("mix/calibration order mismatch:\n"
                         f"  mix   {mix_names}\n  calib {row_names}")
    kp = [[r["k"], r["p"]] for r in sorted(rows, key=lambda r: r["idx"])]

    # LR: a verbatim slice of the plain run's ramp covering the probe window,
    # plus one breakpoint on each side so jnp.interp never extrapolates.
    lo, hi = a.warm_sb, a.warm_sb + a.probe_sb
    ramp = [[s, v] for s, v in spec["lr_ramp"] if lo - 4 <= s <= hi + 4]
    if not ramp:
        raise SystemExit(f"no lr_ramp breakpoints in [{lo - 4}, {hi + 4}]")

    import importlib.util
    p = ROOT / "tools/nnue/sf_to_archon_warm.py"
    m = importlib.util.spec_from_file_location("sf_to_archon_warm", p)
    conv = importlib.util.module_from_spec(m)
    m.loader.exec_module(conv)

    spec.update({
        "net_id": a.net_id,
        "name": f"{a.net_id}-1280",
        "distill_alpha": a.alpha,
        "teacher_warm": a.teacher,
        "teacher_arch": conv.ARCH_KWARGS,
        "teacher_kp": kp,
        "lr_ramp": ramp,
        "superbatches": hi,
        "lr_end_superbatch": hi,
        "export_every": a.export_every,
        "heldout_expect": 0.0,
        "_probe": (
            f"SF-net teacher probe. alpha={a.alpha} (60/40), warm "
            f"{a.warm}, superbatches {lo + 1}..{hi}. Everything else is "
            f"{Path(a.base).name} verbatim, and resume_sb={lo} makes the batch "
            f"stream identical to the plain run's over the same window, so the "
            f"ONLY difference is the teacher. Compare heldout against the "
            f"plain run's own cyc.log over sb{lo + 1}..{hi}."),
    })
    spec.pop("warm", None)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  teacher_kp k range {min(k for k, _ in kp):.4f}-"
          f"{max(k for k, _ in kp):.4f}, p range {min(p for _, p in kp):+.2f}-"
          f"{max(p for _, p in kp):+.2f}")
    print(f"  lr {ramp[0][1]:.6e} @sb{ramp[0][0]} -> "
          f"{ramp[-1][1]:.6e} @sb{ramp[-1][0]}")
    print(f"\nlaunch (resume_sb={lo}; needs "
          f"{a.net_id}-sb{lo}-raw.bin / -ema-raw.bin in the out dir):\n"
          f"  python tools/pump_train.py ../{a.out} {lo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
