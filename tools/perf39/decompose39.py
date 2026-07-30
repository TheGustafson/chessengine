#!/usr/bin/env python3
"""perf study 39 — decompose the additive/subtractive gap into its two factors.

The clean subtractive arm is `dstpin − actpin0`, and it is clean for a reason
that took the whole study to see: **both arms pin `activate` to the same
all-zero buffer, so both compute the same (wrong) evaluation, search the same
tree, and retire the same instructions** — measured at +1.05 i/n out of 5474,
0.019%. Their difference therefore relocates the FOLD accumulator stream from
cold stack memory to a hot 2.5 KB buffer and changes nothing else. That is the
tree-clean, instruction-clean subtractive measurement neither study 29 nor
study 31 had, and it exists only as a difference of two individually dirty
arms.

Line geometry is read off `tools/perf39/census39.sh` and the `pin_src!` /
`pin_dst!` sites in `sota_simd.rs`, not assumed.
"""

# --- census, tools/perf39/census39.sh, fc-base, 5,001,562 nodes -------------
HKA_INTO = 0.9784
HKA_DUAL = 0.1647
THR_INTO = 0.8538
THR_COPY = 0.0220
EVALS = 0.4311
ACC_LINES = 40  # [i16; 1280] = 2560 B

# --- what `dstpin` pins that `actpin0` does not -----------------------------
# sota_simd.rs: pin_src!+pin_dst! on apply_threat_deltas_into (597/598) and
# apply_halfka_deltas_into (644/645); pin_dst! only on apply_halfka_deltas_dual
# (617, one of its two destinations); sota_threats.rs:1495 the empty-delta copy.
SITES = {
    "HalfKA fold, src + dst": HKA_INTO * 2 * ACC_LINES,
    "threat fold, src + dst": THR_INTO * 2 * ACC_LINES,
    "Finny warm refresh, one dst": HKA_DUAL * ACC_LINES,
    "threat empty-delta copy": THR_COPY * 2 * ACC_LINES,
}
FOLD_LINES = sum(SITES.values())

# --- measured, panel-main, n = 24 medians, paired within round --------------
SUB = {"miss": -133.46, "stall": -173.29, "cyc": -187.98, "ins": +1.05}
ADD = {"miss": +87.74, "stall": +219.63, "cyc": +294.75, "ins": +78.90}
ADD_LINES = 160 * EVALS
# `actpin0` vs `d0`: relocating activate's OWN accumulator reads.
ACT = {"miss": -0.22, "stall": -47.80, "lines": 160 * EVALS}

print("\n  LINES RELOCATED BY (dstpin − actpin0)")
for k, v in SITES.items():
    print(f"    {k:32s} {v:7.2f} lines/node")
print(f"    {'TOTAL':32s} {FOLD_LINES:7.2f} lines/node\n")

print(f"  ADDITIVE arm (d1 − d0), {ADD_LINES:.2f} lines/node added, tree identical:")
for k in ("miss", "stall", "cyc"):
    print(f"    {k:6s} {ADD[k]:+8.2f}/node  =>  {ADD[k] / ADD_LINES:6.3f} per line")
print(f"    stall per miss = {ADD['stall'] / ADD['miss']:.3f}\n")

print(f"  SUBTRACTIVE arm (dstpin − actpin0), {FOLD_LINES:.2f} lines/node relocated,")
print(f"  tree identical AND instructions identical ({SUB['ins']:+.2f} i/n):")
for k in ("miss", "stall", "cyc"):
    print(f"    {k:6s} {SUB[k]:+8.2f}/node  =>  {abs(SUB[k]) / FOLD_LINES:6.3f} per line")
print(f"    stall per miss = {SUB['stall'] / SUB['miss']:.3f}\n")

cold_a = ADD["miss"] / ADD_LINES
cold_s = abs(SUB["miss"]) / FOLD_LINES
spm_a = ADD["stall"] / ADD["miss"]
spm_s = SUB["stall"] / SUB["miss"]
gap = (ADD["stall"] / ADD_LINES) / (abs(SUB["stall"]) / FOLD_LINES)
print("  THE GAP, FACTORISED")
print(f"    stall per LINE, additive / subtractive          = {gap:.2f}x")
print(f"    of which COLDNESS  ({cold_a:.3f} vs {cold_s:.3f} miss/line) = {cold_a / cold_s:.2f}x")
print(f"    of which DIRECTION ({spm_a:.2f} vs {spm_s:.2f} stall/miss)    = {spm_a / spm_s:.2f}x")
print(f"    product                                         = "
      f"{(cold_a / cold_s) * (spm_a / spm_s):.2f}x\n")

print("  ACTIVATE'S OWN ACCUMULATOR READS (actpin0 − d0), "
      f"{ACT['lines']:.2f} lines/node relocated:")
print(f"    miss  {ACT['miss']:+8.2f}/node  =>  {abs(ACT['miss']) / ACT['lines']:.4f} per line")
print("    i.e. the four arrays `activate` reads DO NOT MISS — they were written")
print("    by the fold at this ply moments before. Study 28 and study 31 both")
print("    counted that merge as a third of fusion's prize.\n")
