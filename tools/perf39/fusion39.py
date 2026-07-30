#!/usr/bin/env python3
"""perf study 39 — price accumulator fusion on TODAY's engine, both sides.

Studies 29 and 31 priced fusion against a census taken before perf study 32
(the i8 HalfKA feature transformer, −6.8% cycles) and perf study 38 (the threat
fold's parent alias, which took the empty-delta copy from 0.3339 calls/node to
0.0220).  Rule 8 says re-derive any number before building on it, so every rate
below comes from `tools/perf39/census39.sh` on the current tree and every price
from `tools/perf39/dose39.py`.

    python3 tools/perf39/fusion39.py <stall-per-line-cold> <stall-per-line-warm>
                                     <rebuild-rows> [--json]
"""

import sys

# --- measured rates, tools/perf39/census39.sh, fc-base, 5,001,562 nodes ------
EVALS = 0.4311  # `activate` calls per node
THR_FOLD = 0.8538  # FullThreats parent->child folds per node
HKA_REFRESH = 0.1647  # `hka_dual`, the Finny WARM refresh, per node
THR_REFRESH = 0.0196  # from-scratch FullThreats rebuilds per node (study 31: 0.01955)

# --- geometry at ACC = 1280 --------------------------------------------------
ACC_LINES = 40  # one [i16; ACC] accumulator = 2560 B = 40 lines
THR_ROW_LINES = 20  # one i8 threat weight row = 1280 B = 20 lines


def main() -> None:
    cold, warm, rebuild_rows = (float(x) for x in sys.argv[1:4])

    # ---- SAVING ------------------------------------------------------------
    # A fused accumulator does ONE parent->child fold pass where we do two, so
    # the whole accumulator stream of one pass disappears: a 2560 B read plus a
    # 2560 B write per call. That traffic is COLD -- the parent entry is a
    # different ply's slot 5312 B up the stack, the child is a fresh one.
    fold_lines = THR_FOLD * 2 * ACC_LINES
    # `activate` reads four arrays instead of two. These are WARMER: the fold
    # wrote them at this ply moments ago.
    act_lines = EVALS * 2 * ACC_LINES

    save_cold = fold_lines * cold + act_lines * warm
    save_warm = (fold_lines + act_lines) * warm

    # ---- COST --------------------------------------------------------------
    # Fusion makes the threat channel inherit HalfKA's refresh trigger: every
    # HalfKA refresh drags a from-scratch threat rebuild with it. The rebuild
    # folds `rebuild_rows` weight rows out of the 77.7 MB threat table -- the
    # engine's most expensive load class -- and zeroes the destination first.
    extra = HKA_REFRESH - THR_REFRESH
    row_lines = extra * rebuild_rows * THR_ROW_LINES
    zero_lines = extra * ACC_LINES
    # Weight rows are priced on study 31's residual, a BAND not a point.
    for wcost in (5.0, 7.4, 13.0):
        cost = row_lines * wcost + zero_lines * warm
        print(
            f"  weight-row line at {wcost:5.1f} cycles: cost = {cost:+7.1f} cycles/node"
            f"   =>  net {save_cold * -1 + cost:+8.1f} .. {save_warm * -1 + cost:+8.1f}"
        )

    print()
    print(f"  saving, fold pass      : {fold_lines:6.2f} lines/node")
    print(f"  saving, activate merge : {act_lines:6.2f} lines/node")
    print(f"  cost, extra rebuilds   : {extra:6.4f}/node x {rebuild_rows:.2f} rows "
          f"= {row_lines:6.2f} weight-row lines/node + {zero_lines:.2f} zeroed")
    print(f"  saving at cold={cold:.2f}/warm={warm:.2f} : "
          f"{-save_cold:+8.1f} .. {-save_warm:+8.1f} cycles/node")


if __name__ == "__main__":
    main()
