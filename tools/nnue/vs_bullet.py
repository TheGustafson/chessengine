#!/usr/bin/env python3
"""vs-bullet cross-check: CENN packing is faithful to bullet's own weights.

P13 step 3, deliverable 5(b) — the oracle deferred from step 1. It answers a
single question: *does our CENN container (produced by netpack.py) carry
bullet's ``quantised.bin`` weights faithfully — including the i16->i32 output
bias sign-extension (design/11 §3.4.5 packer note)?*

Method (two independent readers of the SAME weights):
  * Python reads bullet's ``quantised.bin`` DIRECTLY (its own offset arithmetic,
    bypassing CENN entirely) and runs the §3.4.3 integer forward pass
    (imported from nnue_ref.py — the step-1 reference already proven
    bit-identical to the Rust engine over 100+ positions by verify_forward.sh).
  * The Rust engine reads the CENN file (packed from that same quantised.bin)
    and runs its own forward (`nnue-eval`).

Because the two forward *implementations* are already known bit-identical, any
disagreement here isolates to the PACKING (wrong offsets, byte order, a
mis-sign-extended bias, checksum over the wrong bytes). Agreement over a broad
FEN set is positive evidence the CENN packing is faithful.

Run under uv (design/11 §11.1):
    uv run python tools/nnue/vs_bullet.py <quantised.bin> <net.cenn> <engine_bin> [fens.txt]
    # FENs from the file arg, else one per line on stdin.

Discardable scaffold (D15/D28): reads gitignored artifacts, no `src/` touched.
"""

import subprocess
import sys

import numpy as np

import nnue_ref  # sibling module (tools/nnue/); the §3.4.3 forward reference.

INPUT, L1 = 768, 256
N_FT_W = INPUT * L1
N_FT_B = L1
N_OUT_W = 2 * L1


def die(msg: str):
    print(f"vs_bullet: {msg}", file=sys.stderr)
    sys.exit(2)


def load_bullet_quantised(path: str) -> "nnue_ref.Net":
    """Parse bullet's raw quantised.bin into a Net (independent of netpack.py).

    Layout (bullet save-format order, tools/nnue/train_net0.rs):
        l0w i16[768*256], l0b i16[256], l1w i16[512], l1b i16[1], then pad.
    The i16 l1b is sign-extended to the i32 out_bias (this is exactly the
    packing step under test)."""
    raw = open(path, "rb").read()
    n_i16 = N_FT_W + N_FT_B + N_OUT_W + 1
    if len(raw) < n_i16 * 2:
        die(f"{path}: {len(raw)} bytes < {n_i16 * 2} needed")
    a = np.frombuffer(raw, dtype="<i2", count=n_i16).astype(np.int64)
    off = 0
    ft_w = a[off:off + N_FT_W].tolist(); off += N_FT_W
    ft_b = a[off:off + N_FT_B].tolist(); off += N_FT_B
    out_w = a[off:off + N_OUT_W].tolist(); off += N_OUT_W
    out_bias = int(a[off])  # signed i16 value -> i32 out_bias (sign-extended)
    # SCReLU (activation id 1) — net-0's fixed activation (design/11 §3.4.1).
    return nnue_ref.Net(ft_w, ft_b, out_w, out_bias, 1)


def main():
    if len(sys.argv) < 4:
        die("usage: vs_bullet.py <quantised.bin> <net.cenn> <engine_bin> [fens.txt]")
    quantised, cenn, engine = sys.argv[1], sys.argv[2], sys.argv[3]

    if len(sys.argv) >= 5:
        with open(sys.argv[4]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()
    fens = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    if not fens:
        die("no FENs provided")

    net = load_bullet_quantised(quantised)

    # Rust engine forward over the CENN-packed net (batch mode: one FEN/line).
    proc = subprocess.run(
        [engine, "nnue-eval", cenn],
        input="\n".join(fens) + "\n",
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        die(f"engine nnue-eval failed: {proc.stderr.strip()}")
    rust_vals = [int(x) for x in proc.stdout.split()]
    if len(rust_vals) != len(fens):
        die(f"engine returned {len(rust_vals)} values for {len(fens)} FENs")

    max_abs = 0
    mismatches = []
    for fen, rust_v in zip(fens, rust_vals):
        py_v = nnue_ref.evaluate(net, fen)
        d = abs(py_v - rust_v)
        max_abs = max(max_abs, d)
        if d != 0:
            mismatches.append((fen, py_v, rust_v))

    n = len(fens)
    print(f"vs_bullet: {n} positions")
    print(f"  quantised.bin : {quantised}")
    print(f"  cenn          : {cenn}")
    print(f"  matches       : {n - len(mismatches)}/{n}")
    print(f"  max |py - rust|: {max_abs}")
    if mismatches:
        print(f"  MISMATCH on {len(mismatches)} position(s):")
        for fen, py_v, rust_v in mismatches[:10]:
            print(f"    py={py_v} rust={rust_v}  {fen}")
        print("==> FAIL: CENN packing does NOT faithfully reproduce bullet's weights")
        sys.exit(1)
    print("==> PASS: CENN packing (incl. i16->i32 out_bias sign-extension) is "
          "bit-faithful to bullet's quantised.bin over all positions")


if __name__ == "__main__":
    main()
