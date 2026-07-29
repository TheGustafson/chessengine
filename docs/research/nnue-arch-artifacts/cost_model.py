#!/usr/bin/env python3
"""ARCHON NNUE architecture cost model — instructions/node as a function of
(accumulator width L1, neck width L2, tail width L3, threats on/off).

CALIBRATION (all measured, none assumed):
  * base point = D48X (L1=1280, L2=16, L3=32, threats on), 7116.6 instructions
    /node on the matched 15.0M-node wl50 panel — perf study 8 §2, `instructions:u`
    spread 0.0000% over six interleaved rounds.
  * phase split of `forward_into` and the width-dependent block — perf study 8
    §3.2 and §1.1.
  * l1_dot's loop body is 8 static instructions post-§4.1, of which exactly ONE
    (`vpdpbusd`) scales with L2 in 16-lane steps; the other 7 are per-group
    overhead. That is read off the disassembly quoted in perf study 8 §4.1.
  * threat removal factor from perf study 7 §5 (1.317x wl50 / 1.389x middlegame
    at 1280, deterministic instruction counts, node totals verified equal).

VALIDATION: the model is checked against the ONLY independent 4-point speed
ladder we have — exp-0156's idle-box PGO NPS smoke (V-768/32 1051k, X-1280/16
893k, W-1536/16 804k, S-1280/128 724k). See `validate()`.

Run:  python3 docs/research/nnue-arch-artifacts/cost_model.py
"""

from __future__ import annotations

BASE_IPN = 7116.6          # perf study 8 §2, D48X PGO
BASE = dict(L1=1280, L2=16, L3=32, threats=True)

# --- perf study 8 §3.2 / §1.1, as instructions/node at the base point --------
# forward_into phases (share of engine x BASE_IPN)
P = {
    "activate":   0.0494 * BASE_IPN,   # pairwise FT merge, ~ L1
    "find_nnz":   0.0075 * BASE_IPN,   # ~ L1
    "l1_dot":     0.0617 * BASE_IPN,   # ~ L1 (groups) x (7 + L2/16)/8
    "l1_dequant": 0.0004 * BASE_IPN,   # ~ L2
    "l2_l3":      0.0216 * BASE_IPN,   # ~ (L2*L3 + L3) MACs
    "skip_dot":   0.0250 * BASE_IPN,   # ~ L1
}
# accumulator-maintenance block (per node, not per eval), all ~ L1
ACC_SHARE = 0.0664 + 0.1064 + 0.0534 + 0.0345 + 0.0186 + 0.0091 + 0.0200
#            ^at_full plumbing+fold split as 5.73/10.64; current_full 5.34;
#             i16_deltas 3.45; finny 1.86; refresh_threat 0.91; memmove 2.00
ACC = ACC_SHARE * BASE_IPN
# threat EVENT generation (record_square_touch + record_targets): O(events),
# independent of L1
THREAT_EVENTS = 0.0355 * BASE_IPN
# everything else: search shell, movegen, movepick, make/unmake, misc
REST = BASE_IPN - sum(P.values()) - ACC - THREAT_EVENTS

# Threat-dependent share of the width-scaling block. At 1280/wl50 removing
# threats measured 1.317x  =>  it removes 24.06% of all instructions.
# We attribute that removal to: THREAT_EVENTS (3.55%), refresh_threat (0.91%),
# the at_full stage (16.37% = 10.64 fold + 5.73 plumbing), half the accumulator
# memmove (1.00%), and ~40% of `activate` (the second-accumulator load+rescale).
THREAT_BLOCK = {"events": THREAT_EVENTS,
                "refresh": 0.0091 * BASE_IPN,
                "at_full": 0.1637 * BASE_IPN,
                "memmove_half": 0.0100 * BASE_IPN,
                "activate_frac": 0.40 * P["activate"]}


def ipn(L1=1280, L2=16, L3=32, threats=True, verbose=False):
    w = L1 / BASE["L1"]
    acc = ACC * w
    fw = {
        "activate":   P["activate"] * w * (1.0 if threats else 0.60),
        "find_nnz":   P["find_nnz"] * w,
        "l1_dot":     P["l1_dot"] * w * (7 + L2 / 16) / 8,
        "l1_dequant": P["l1_dequant"] * (L2 / 16),
        "l2_l3":      P["l2_l3"] * ((L2 * L3 + L3) / (16 * 32 + 32)),
        "skip_dot":   P["skip_dot"] * w,
    }
    ev = THREAT_EVENTS if threats else 0.0
    if not threats:
        # at_full, refresh_threat, half the memmove and the threat half of the
        # accumulator block all vanish
        acc -= (THREAT_BLOCK["at_full"] + THREAT_BLOCK["refresh"]
                + THREAT_BLOCK["memmove_half"]) * w
    total = REST + acc + ev + sum(fw.values())
    if verbose:
        print(f"  L1={L1} L2={L2} L3={L3} threats={threats}: "
              f"rest {REST:6.0f} acc {acc:6.0f} ev {ev:5.0f} fw {sum(fw.values()):6.0f} "
              f"=> {total:7.1f} i/n")
    return total


def elo_from_speed(ratio, per_doubling=120.0):
    import math
    return per_doubling * math.log2(ratio)


def validate():
    """Against exp-0156's 4-point idle-box PGO NPS ladder (the only one we have)."""
    ladder = [("V  768/32/32", 768, 32, 1051), ("X 1280/16/32", 1280, 16, 893),
              ("W 1536/16/32", 1536, 16, 804), ("S 1280/128/32", 1280, 128, 724)]
    print("VALIDATION vs exp-0156 (NPS ratios relative to X-1280/16):")
    x_ipn = ipn(1280, 16)
    print(f"  {'arch':16s} {'nps':>6s} {'measured':>9s} {'model':>7s} {'err':>7s}")
    for name, l1, l2, nps in ladder:
        meas = nps / 893.0
        pred = x_ipn / ipn(l1, l2)
        print(f"  {name:16s} {nps:6d} {meas:9.3f} {pred:7.3f} {100*(pred/meas-1):+6.1f}%")


def table():
    rows = [
        ("D48X (canonical)",            1280, 16, 32, True),
        ("D47V (previous)",              768, 32, 32, True),
        ("owner's '512 -> 8 -> 8'",      512,  8,  8, True),
        ("512/8/8, threats OFF",         512,  8,  8, False),
        ("512/16/32, threats ON",        512, 16, 32, True),
        ("neck 16 -> 8",                1280,  8, 32, True),
        ("neck 16 -> 32",               1280, 32, 32, True),
        ("neck 16 -> 128",              1280,128, 32, True),
        ("width 1280 -> 3072",          3072, 16, 32, True),
        ("width 1280 -> 2048",          2048, 16, 32, True),
        ("threats OFF @1280",           1280, 16, 32, False),
        ("threats OFF @2048",           2048, 16, 32, False),
        ("threats OFF @2560",           2560, 16, 32, False),
        ("threats OFF @3072",           3072, 16, 32, False),
        ("threats OFF @4096",           4096, 16, 32, False),
    ]
    base = ipn(**BASE)
    print(f"\n{'arch':28s} {'i/n':>8s} {'vs D48X':>8s} {'speed':>7s} "
          f"{'Elo@120':>8s} {'Elo@70':>7s}")
    for name, l1, l2, l3, th in rows:
        v = ipn(l1, l2, l3, th)
        sp = base / v
        print(f"{name:28s} {v:8.0f} {v/base:8.3f} {sp:7.3f} "
              f"{elo_from_speed(sp,120):+8.1f} {elo_from_speed(sp,70):+7.1f}")


if __name__ == "__main__":
    validate()
    table()
