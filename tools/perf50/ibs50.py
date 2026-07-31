#!/usr/bin/env python3
"""perf study 50 — per-symbol census from AMD IBS op samples.

Why IBS and not a sampled PMU counter: IBS tags a micro-op at issue and records
that op's OWN rip (IbsOpRip) plus, for a load that misses the data cache, the
measured miss latency (IbsDcMissLat).  There is no skid: the sample is the op.
A sampled `ex_no_retire.load_not_complete` overflow, by contrast, is delivered
tens of cycles after the counter overflows, so its rip is wherever retirement
had got to by then -- which is exactly the quantity under test.

Input: `perf script -F ip,sym,weight,data_src`.  Layout after the final '|' is
    BLK  N/A   <weight> <ip> <symbol...>
"""
import sys
import collections

STARTUP = ("init_magics", "load_v2_bytes", "quantise_halfka", "sota_loader")


def main() -> int:
    ops = collections.Counter()       # all tagged ops
    loads = collections.Counter()     # tagged ops that are loads
    misses = collections.Counter()    # loads with a DC miss (weight > 0)
    lat = collections.Counter()       # summed DC-miss latency
    lvl = collections.defaultdict(collections.Counter)

    for line in sys.stdin:
        if "|" not in line:
            continue
        head, _, tail = line.rpartition("|")
        tok = tail.split()
        if len(tok) < 4:
            continue
        # tok[0:2] == ['BLK','N/A'] (or 'BLK','Yes'); then weight, ip, symbol
        try:
            weight = int(tok[2])
        except ValueError:
            continue
        sym = " ".join(tok[4:]) or "[unknown]"
        # `perf script -F symoff` appends "+0x..."; strip it so this
        # stays a per-SYMBOL census (join49.py is the per-offset view).
        sym = sym.split("+0x")[0]
        ops[sym] += 1
        is_load = "OP LOAD" in head
        if is_load:
            loads[sym] += 1
            for name in ("L1 hit", "L2 hit", "L3 hit", "RAM hit", "Remote"):
                if "LVL " + name in head:
                    lvl[sym][name] += 1
                    break
            else:
                lvl[sym]["other"] += 1
        if weight > 0:
            misses[sym] += 1
            lat[sym] += weight

    tot_ops = sum(ops.values())
    tot_lat = sum(lat.values())
    tot_loads = sum(loads.values())
    tot_miss = sum(misses.values())
    # startup symbols are excluded from the shares, per study 48's warning that
    # loading the 138 MB net is ~3 % of a naive profile
    def is_startup(s):
        return any(k in s for k in STARTUP)

    live_lat = sum(v for k, v in lat.items() if not is_startup(k))
    live_ops = sum(v for k, v in ops.items() if not is_startup(k))
    live_loads = sum(v for k, v in loads.items() if not is_startup(k))
    live_miss = sum(v for k, v in misses.items() if not is_startup(k))

    print(f"# tagged ops {tot_ops}  loads {tot_loads}  dc-miss loads {tot_miss} "
          f" summed miss latency {tot_lat}")
    print(f"# excluding startup: ops {live_ops} loads {live_loads} "
          f"miss {live_miss} lat {live_lat}")
    print(f"{'symbol':62s} {'%ops':>7s} {'%loads':>7s} {'%miss':>7s} "
          f"{'%LAT':>7s} {'lat/miss':>8s} {'miss/load':>9s}  L1/L2/L3/RAM")
    for sym, _ in sorted(lat.items(), key=lambda kv: -kv[1])[:28]:
        if is_startup(sym):
            continue
        d = lvl[sym]
        mix = "/".join(str(d[k]) for k in ("L1 hit", "L2 hit", "L3 hit", "RAM hit"))
        short = sym.replace("chessengine::", "").replace("eval::nnue::", "")
        short = short.replace("<", "").replace(">", "")
        print(f"{short[:62]:62s} {100*ops[sym]/live_ops:7.2f} "
              f"{100*loads[sym]/live_loads:7.2f} {100*misses[sym]/live_miss:7.2f} "
              f"{100*lat[sym]/live_lat:7.2f} {lat[sym]/max(misses[sym],1):8.1f} "
              f"{100*misses[sym]/max(loads[sym],1):8.1f}%  {mix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
