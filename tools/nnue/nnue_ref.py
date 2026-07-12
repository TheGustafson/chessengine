#!/usr/bin/env python3
"""Independent CENN forward-pass reference (design/11 §3.4.3, §3.4.5).

This is the step-1 forward-pass **bit-oracle**: an implementation of the
quantized integer forward pass written from the design spec, entirely
separate from the Rust engine, so that agreement between the two is real
evidence the math is right (not a tautology). It reads a CENN net + a list of
FENs and prints, per FEN, the exact integer centipawn value — the same number
the engine's `nnue-eval` subcommand prints.

Accumulation is in Python `int` (arbitrary precision), so this is a true
reference; the only place we deliberately model a fixed width is the
feature-transformer accumulator, which the spec pins to i16 — we wrap to i16
to mirror the engine exactly (a no-op for real / small-weight nets, but it
keeps the two bit-identical even on a synthetic net that would overflow).
Integer division truncates toward zero, matching Rust's `/`.

Run under uv (design/11 §11.1): `uv run python tools/nnue/nnue_ref.py`.
The vs-`bullet` cross-check is a separate, later oracle (step 3): it folds in
once a real trained net exists; this script never fabricates one.

Usage:
    uv run python tools/nnue/nnue_ref.py <net.cenn> [fens.txt]
    # FENs from the file arg, or one per line on stdin if omitted.
    # '#' comments and blank lines are skipped.
"""

import sys

import numpy as np

# Architecture + quantization (design/11 §3.4.1, §3.4.3) — must match the
# engine constants in src/eval/nnue/mod.rs.
INPUT = 768
L1 = 256
QA = 255
QB = 64
SCALE = 400
HEADER_LEN = 48
MAGIC = b"CENN"
CENN_VERSION = 1

# PieceType order matches the Rust enum: Pawn, Knight, Bishop, Rook, Queen, King.
PIECE_TYPE = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}

WHITE, BLACK = 0, 1


class Net:
    """A parsed, validated CENN network."""

    __slots__ = ("ft_weights", "ft_bias", "out_weights", "out_bias", "activation")

    def __init__(self, ft_weights, ft_bias, out_weights, out_bias, activation):
        self.ft_weights = ft_weights  # int list, feature-major (INPUT*L1)
        self.ft_bias = ft_bias  # int list (L1)
        self.out_weights = out_weights  # int list (2*L1), stm half first
        self.out_bias = out_bias  # int
        self.activation = activation  # 0 = CReLU, 1 = SCReLU


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def load_net(path: str) -> Net:
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < HEADER_LEN:
        die(f"CENN too short: {len(raw)} bytes")
    if raw[0:4] != MAGIC:
        die(f"CENN bad magic: {raw[0:4]!r}")
    version = int.from_bytes(raw[4:8], "little")
    if version != CENN_VERSION:
        die(f"CENN unsupported version {version}")
    input_dims = int.from_bytes(raw[8:12], "little")
    l1 = int.from_bytes(raw[12:16], "little")
    if input_dims != INPUT or l1 != L1:
        die(f"CENN arch {input_dims}->{l1} != {INPUT}->{L1}")
    activation = raw[16]
    if activation not in (0, 1):
        die(f"CENN bad activation id {activation}")
    qa = int.from_bytes(raw[20:22], "little")
    qb = int.from_bytes(raw[22:24], "little")
    scale = int.from_bytes(raw[24:26], "little")
    if (qa, qb, scale) != (QA, QB, SCALE):
        die(f"CENN quant {(qa, qb, scale)} != {(QA, QB, SCALE)}")
    payload_len = int.from_bytes(raw[32:40], "little")
    checksum = int.from_bytes(raw[40:48], "little")

    expected_payload = (INPUT * L1 + L1 + 2 * L1) * 2 + 4
    if payload_len != expected_payload:
        die(f"CENN payload_len {payload_len} != {expected_payload}")
    if len(raw) < HEADER_LEN + expected_payload:
        die(f"CENN truncated payload: {len(raw)} bytes")
    payload = raw[HEADER_LEN : HEADER_LEN + expected_payload]
    if fnv1a64(payload) != checksum:
        die(f"CENN checksum mismatch: header {checksum:#018x}, computed {fnv1a64(payload):#018x}")

    # int16 little-endian arrays -> Python int lists (exact arithmetic).
    off = 0
    n_ft = INPUT * L1
    ft_weights = np.frombuffer(payload, dtype="<i2", count=n_ft, offset=off).astype(np.int64).tolist()
    off += n_ft * 2
    ft_bias = np.frombuffer(payload, dtype="<i2", count=L1, offset=off).astype(np.int64).tolist()
    off += L1 * 2
    out_weights = np.frombuffer(payload, dtype="<i2", count=2 * L1, offset=off).astype(np.int64).tolist()
    off += 2 * L1 * 2
    out_bias = int(np.frombuffer(payload, dtype="<i4", count=1, offset=off)[0])
    return Net(ft_weights, ft_bias, out_weights, out_bias, activation)


def parse_placement(fen: str):
    """Return (pieces, stm): pieces is a list of (color, piece_type, square)
    with square in little-endian rank-file (a1 = 0); stm is 0/1."""
    parts = fen.split()
    if not parts:
        die(f"empty FEN: {fen!r}")
    placement = parts[0]
    stm = WHITE if (len(parts) < 2 or parts[1] == "w") else BLACK

    pieces = []
    rank = 7  # FEN lists rank 8 first
    for row in placement.split("/"):
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                color = WHITE if ch.isupper() else BLACK
                pt = PIECE_TYPE.get(ch.lower())
                if pt is None:
                    die(f"bad piece {ch!r} in FEN {fen!r}")
                sq = rank * 8 + file
                pieces.append((color, pt, sq))
                file += 1
        rank -= 1
    return pieces, stm


def to_i16(x: int) -> int:
    """Wrap an int into the i16 range (mirrors the engine's i16 accumulator)."""
    x &= 0xFFFF
    return x - 0x10000 if x >= 0x8000 else x


def feature_index(perspective: int, color: int, pt: int, sq: int) -> int:
    opponent = 1 if color != perspective else 0
    rel = sq if perspective == WHITE else sq ^ 56
    return (opponent * 6 + pt) * 64 + rel


def refresh(net: Net, pieces, perspective: int):
    """From-scratch accumulator: bias + sum of active feature rows, wrapped to
    i16 (design/11 §3.4.4)."""
    acc = np.array(net.ft_bias, dtype=np.int64)
    w = net.ft_weights
    for (color, pt, sq) in pieces:
        f = feature_index(perspective, color, pt, sq)
        base = f * L1
        acc += np.array(w[base : base + L1], dtype=np.int64)
    # Wrap to i16 elementwise (no-op for small-weight nets; faithful either way).
    acc = ((acc + 0x8000) & 0xFFFF).astype(np.int64) - 0x8000
    return acc.tolist()


def trunc_div(a: int, b: int) -> int:
    """Integer division truncating toward zero (Rust `/` on i64)."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def forward(net: Net, us, them) -> int:
    """The exact §3.4.3 integer forward pass, accumulating in Python int.
    `us`/`them` are the stm/nstm accumulators; out_weights stm half first."""
    w = net.out_weights
    total = 0
    if net.activation == 1:  # SCReLU
        for i in range(L1):
            a = min(max(us[i], 0), QA)
            total += a * a * w[i]
        for i in range(L1):
            a = min(max(them[i], 0), QA)
            total += a * a * w[L1 + i]
        total = trunc_div(total, QA)
    else:  # CReLU
        for i in range(L1):
            a = min(max(us[i], 0), QA)
            total += a * w[i]
        for i in range(L1):
            a = min(max(them[i], 0), QA)
            total += a * w[L1 + i]
    return trunc_div((total + net.out_bias) * SCALE, QA * QB)


def evaluate(net: Net, fen: str) -> int:
    pieces, stm = parse_placement(fen)
    acc_w = refresh(net, pieces, WHITE)
    acc_b = refresh(net, pieces, BLACK)
    us, them = (acc_w, acc_b) if stm == WHITE else (acc_b, acc_w)
    return forward(net, us, them)


def die(msg: str):
    print(f"nnue_ref: {msg}", file=sys.stderr)
    sys.exit(2)


def main():
    if len(sys.argv) < 2:
        die("usage: nnue_ref.py <net.cenn> [fens.txt]")
    net = load_net(sys.argv[1])

    if len(sys.argv) >= 3:
        with open(sys.argv[2]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    out = []
    for line in lines:
        fen = line.strip()
        if not fen or fen.startswith("#"):
            continue
        out.append(str(evaluate(net, fen)))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
