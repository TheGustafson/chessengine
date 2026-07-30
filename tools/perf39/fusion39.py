#!/usr/bin/env python3
"""perf study 39 — price accumulator fusion on TODAY's engine, both sides, in
ONE currency.

Studies 29 and 31 priced fusion against a census taken before perf study 32
(the i8 HalfKA transformer) and before study 38 (the threat fold's parent
alias, which took the empty-delta copy from 0.3339 calls/node to 0.0220), so
rule 8 says re-derive.  Every rate below is measured on the current tree by
`tools/perf39/census39.sh`; the accumulator line price is the direction- and
coldness-matched one from `tools/perf39/decompose39.py`.

**One currency.** Study 31 kept two ledgers — an instruction ledger it called
"instruction-neutral" and a cycles ledger holding only the memory side — and
then quoted the cycles ledger as the answer.  Instructions cost cycles: its own
instruction ledger topped out at +199 i/n, which is +137 cycles at this
engine's IPC.  Everything here is converted to cycles/node at the measured
IPC and added up once.
"""

# --- measured, tools/perf39/census39.sh, fc-base, 5,001,562 nodes -----------
EVALS = 0.4311
THR_FOLD = 0.8538
HKA_REFRESH = 0.1647  # `hka_dual`, the Finny WARM refresh
THR_REFRESH = 0.0196  # from-scratch FullThreats rebuilds  (study 31: 0.01955)
REBUILD_ROWS = 4.156  # rows per rebuild                   (study 31: 4.18)

# --- measured, panel-main n = 24 -------------------------------------------
IPC = 5928.71 / 4080.71  # 1.453
ACC_LINES = 40  # [i16; 1280] = 2560 B
THR_ROW_LINES = 20  # i8 threat weight row, 1280 B

# accumulator line, direction- and coldness-matched (dstpin − actpin0)
LINE_CYC = 1.213
# `activate`'s own accumulator reads, measured at 0.0032 misses/line — they do
# not miss, so halving them buys issue work only.
ACT_LINE_CYC = 0.0
# probe-derived issue cost of an accumulator line: 78.90 i/n per 68.98 lines
LINE_INS = 78.90 / (160 * EVALS)

fold_lines = THR_FOLD * 2 * ACC_LINES
act_lines = EVALS * 2 * ACC_LINES
extra = HKA_REFRESH - THR_REFRESH

print(f"\n  IPC = {IPC:.3f}   (base 5928.71 i/n, 4080.71 cyc/n)\n")
print("  SAVING")
loc = fold_lines * LINE_CYC + act_lines * ACT_LINE_CYC
print(f"    fold pass, {fold_lines:6.2f} lines/node x {LINE_CYC:.3f} cyc/line "
      f"= {-loc + act_lines * ACT_LINE_CYC:8.1f}")
print(f"    activate merge, {act_lines:6.2f} lines/node -- MEASURED not to miss "
      f"= {-act_lines * ACT_LINE_CYC:8.1f}")
# issue: probe-derived at the low end, studies 22 + 12's direct kernel counts high
for label, ins in (("probe-derived", (fold_lines + act_lines) * LINE_INS),
                   ("studies 22+12", 226.0)):
    print(f"    issue, {label:14s} {-ins:7.1f} i/n = {-ins / IPC:8.1f}")
save_lo = -(loc + (fold_lines + act_lines) * LINE_INS / IPC)
save_hi = -(loc + 226.0 / IPC)
print(f"    => saving {save_hi:+8.1f} .. {save_lo:+8.1f} cycles/node\n")

print("  COST -- fusion makes the threat channel inherit HalfKA's refresh trigger")
print(f"    extra rebuilds: {HKA_REFRESH:.4f} - {THR_REFRESH:.4f} = {extra:.4f}/node "
      f"({HKA_REFRESH / THR_REFRESH:.1f}x more threat rebuilds)")
rows = extra * REBUILD_ROWS * THR_ROW_LINES
fill = extra * ACC_LINES * LINE_CYC
print(f"    threat weight rows: {rows:6.2f} cold lines/node out of a 77.7 MB table")
for w in (6.5, 13.0):
    print(f"      at {w:5.1f} cycles/line = {rows * w:+8.1f}")
print(f"    dst.fill(0): {extra * ACC_LINES:.2f} lines = {fill:+8.1f}")
for label, ins in (("study 31 derived, 1300 i/rebuild", 1300.0),
                   ("study 18/19 measured symbol, 2808", 2808.0)):
    print(f"    rebuild issue, {label:34s} {extra * ins:7.1f} i/n "
          f"= {extra * ins / IPC:+8.1f}")
cost_lo = rows * 6.5 + fill + extra * 1300.0 / IPC
cost_hi = rows * 13.0 + fill + extra * 2808.0 / IPC
print(f"    => cost {cost_lo:+8.1f} .. {cost_hi:+8.1f} cycles/node\n")

print(f"  NET: {save_hi + cost_lo:+8.1f} (best case for fusion) .. "
      f"{save_lo + cost_hi:+8.1f} (worst)  cycles/node")
print(f"       base 4080.71 cyc/node  =>  {100 * (save_hi + cost_lo) / 4080.71:+.2f}% .. "
      f"{100 * (save_lo + cost_hi) / 4080.71:+.2f}%")
import math
for net in (save_hi + cost_lo, save_lo + cost_hi):
    ratio = 4080.71 / (4080.71 + net)
    print(f"       net {net:+7.1f} => {ratio:.4f}x throughput => "
          f"{120 * math.log2(ratio):+5.1f} Elo at 120 Elo/doubling")
print()
