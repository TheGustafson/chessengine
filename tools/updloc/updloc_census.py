#!/usr/bin/env python3
"""Learned-representation concept study, Tier-0: the update-locality census.

Turns a FOLDCOUNT dump (the instrument build's `ARCHON_FOLDCOUNT` file) into
(a) the incumbent's measured update-locality — rows changed per made move,
rows actually folded per node, active rows per evaluated position, refresh
interaction — and (b) the BUDGET LINE: what one unit of each cost dimension
prices at, in cycles/node and Elo, both as an ADDITION to the incumbent and
as a REPLACEMENT for part of the threat channel.

Denominators and prices, stated explicitly (the 2x cycles/node lesson):
  * cycles/node comes from THIS study's perf-stat battery on the pinned frame
    (canonical PGO binary, startup-subtracted): median 4513.9 loaded
    (trainer + studies live, load ~6/16), ~4100 quiet (study 40's implied
    4104, confirmed by the coordinator's 4489.8 under contention, and by our
    min-round 4393.7). Study 36's per-node total 8262.9 is REFUTED and never
    used here; its within-profile symbol SHARES are used, which are
    self-normalising and load-insensitive.
  * Elo at ~120 per speed doubling (the project's measured constant):
    dElo = 120 * log2(1 + dCycles / C).

Usage: python3 tools/updloc/updloc_census.py <foldcount-dump> [cycles/node]
"""

import math
import re
import sys

# ---- prices (documented in the study doc; re-derive before reusing) --------
C_LOADED = 4513.9  # cycles/node, this study's battery, median, trainer live
C_QUIET = 4100.0   # cycles/node, quiet estimate (study 40 / coordinator tick)
ELO_PER_DOUBLING = 120.0

# perf-study-36 within-profile cycle shares (self-normalising; its per-node
# TOTALS line is refuted, the shares are not).
SHARE = {
    "hka_fold": 0.1377,       # avx512::i8_deltas_into
    "thr_fold": 0.1238,       # avx512::i8_deltas_into_generic
    "finny": 0.0875,          # HalfkaContext::finny_refresh_into
    "head": 0.0615,           # SotaHead::forward_into
    "at_full": 0.0550,        # ThreatContext::at_full
    "hka_walk": 0.0295,       # HalfkaContext::current_full
    "touch": 0.0173,          # sota_threats::record_square_touch
    "psqt_fold": 0.0117,      # sota_accumulator::psqt_fold (both channels)
    "apply_at": 0.0103,       # ThreatContext::apply_at (make-time snapshot)
    "ensure": 0.0092,         # Board::ensure_threats
}
THREAT_CHANNEL_NAMED = ["thr_fold", "at_full", "touch", "apply_at", "ensure"]


def elo(dcycles, c):
    return ELO_PER_DOUBLING * math.log2(1.0 + dcycles / c)


def parse(path):
    kv = {}
    for line in open(path):
        if line.startswith("FOLDCOUNT "):
            for k, v in re.findall(r"([a-zA-Z_0-9.]+)=(\d+)", line):
                kv[k] = int(v)
    return kv


def hist(kv, prefix, size):
    h = [0] * size
    for k, v in kv.items():
        m = re.fullmatch(re.escape(prefix) + r"\.(\d+)", k)
        if m:
            h[int(m.group(1))] = v
    return h


def pct(h, q):
    tot = sum(h)
    if tot == 0:
        return 0
    acc = 0
    for i, n in enumerate(h):
        acc += n
        if acc >= q * tot:
            return i
    return len(h) - 1


def ktrans(kv):
    t = [[0] * 64 for _ in range(64)]
    for k, v in kv.items():
        m = re.fullmatch(r"ktrans\.(\d+)\.(\d+)", k)
        if m:
            t[int(m.group(1))][int(m.group(2))] = v
    return t


def flip_rate(t, part, nodes):
    """Crossing rate per node of a 64-square partition `part` (sq -> cell)."""
    return sum(t[f][g] for f in range(64) for g in range(64)
               if part[f] != part[g]) / nodes


def anneal_bipartition(t, nodes, mass, min_mass_share, seed=1, iters=200_000):
    """Min-flip 2-partition of the 64 king squares subject to each side
    holding >= min_mass_share of king-arrival mass. Simulated annealing with
    incremental O(64) move evaluation; near-exhaustive at this size."""
    import random
    rng = random.Random(seed)
    part = [rng.randint(0, 1) for _ in range(64)]
    tot_mass = sum(mass)
    sym = [[t[a][b] + t[b][a] for b in range(64)] for a in range(64)]

    def cross(p):
        return sum(sym[a][b] for a in range(64) for b in range(a + 1, 64)
                   if p[a] != p[b])

    def pen(m1):
        bal = min(m1, tot_mass - m1) / tot_mass
        return 0.0 if bal >= min_mass_share else (min_mass_share - bal) * 1e12

    cur_f = cross(part)
    cur_m = sum(mass[s] for s in range(64) if part[s])
    cur = cur_f + pen(cur_m)
    best, best_c = part[:], cur
    for i in range(iters):
        temp = max(1e-3, 1.0 - i / iters) * 0.05 * max(cur_f, 1.0)
        s = rng.randrange(64)
        # flipping s: pairs (s,o) with p[o]==p[s] become crossings, crossings
        # with p[o]!=p[s] stop being crossings.
        df = sum(sym[s][o] if part[o] == part[s] else -sym[s][o]
                 for o in range(64) if o != s)
        dm = -mass[s] if part[s] else mass[s]
        new = cur_f + df + pen(cur_m + dm)
        if new <= cur or rng.random() < math.exp((cur - new) / temp):
            part[s] ^= 1
            cur_f += df
            cur_m += dm
            cur = new
            if cur < best_c:
                best, best_c = part[:], cur
    return best, flip_rate(t, best, nodes)


def main():
    kv = parse(sys.argv[1])
    c_loaded = float(sys.argv[2]) if len(sys.argv) > 2 else C_LOADED
    n = kv["nodes"]
    r = lambda k: kv.get(k, 0) / n

    print(f"nodes={n}  (= made moves: make.calls={kv['make.calls']})")
    assert kv["make.calls"] == n, "made-move total must equal node total"

    print("\n== make-time locality (unconditional, per made move) ==")
    print(f"halfka delta rows/move : {kv['make.rows']/n:8.4f}")
    hm = hist(kv, "hka_make_shape", 16)  # printed by dump as a*4+s key form
    # dump emits hka_make_shape.{a}a{s}s; re-parse those:
    shapes = {k: v for k, v in kv.items() if k.startswith("hka_make_shape")}
    tot = sum(shapes.values())
    for k in sorted(shapes):
        print(f"  {k.split('.')[1]:6s} {shapes[k]:>12,}  ({shapes[k]/tot:6.2%})")
    print(f"threat events/move     : {kv['thr_make.rows']/n:8.4f}")
    th = hist(kv, "thr_make_shape", 96)
    print(f"  p50={pct(th,0.5)} p90={pct(th,0.9)} p99={pct(th,0.99)} "
          f"max_bin={max(i for i,v in enumerate(th) if v)}")
    print(f"king moves/move        : {r('kmove.total'):8.4f}")

    print("\n== fold-time (what the engine actually pays), per node ==")
    for site in ["hka_into", "hka_dual", "hka_inplace", "thr_into",
                 "thr_copy", "thr_alias", "thr_refresh", "activate", "eval"]:
        c, w = kv.get(f"{site}.calls", 0), kv.get(f"{site}.rows", 0)
        rows_per_call = w / c if c else 0.0
        print(f"  {site:12s} calls/n={c/n:8.4f}  rows/n={w/n:8.4f}  "
              f"rows/call={rows_per_call:7.3f}")

    print("\n== deferral / laziness interaction ==")
    ev = kv["eval.calls"] / n
    print(f"eval rate              : {ev:8.4f} evals/node")
    print(f"halfka rows folded / recorded : "
          f"{kv['hka_into.rows']/kv['make.rows']:.4f}  (chain-fold pays ~all)")
    thr_paid = kv["thr_into.rows"]
    thr_potential = 2 * kv["thr_make.rows"]  # 2 perspectives
    print(f"threat rows folded / (2x events recorded) : "
          f"{thr_paid/thr_potential:.4f}   <- fold-time survival")
    mirror = kv["kmove.mirror"]
    refr = kv["thr_refresh.calls"]
    print(f"mirror crossings/node  : {mirror/n:8.5f}")
    print(f"thr refreshes/node     : {refr/n:8.5f}   "
          f"survival s = {refr/mirror:.3f}")

    print("\n== active rows per evaluated position ==")
    ah = hist(kv, "active_hka", 33)
    mean_pieces = sum(i * v for i, v in enumerate(ah)) / max(sum(ah), 1)
    print(f"halfka active/persp    : mean {mean_pieces:6.2f}  "
          f"p50={pct(ah,0.5)} p90={pct(ah,0.9)}")
    at = kv["active_thr.rows"] / kv["active_thr.calls"]
    ath = hist(kv, "active_thr_shape", 128)
    print(f"threat active/persp    : mean {at:6.2f}  "
          f"p50={pct(ath,0.5)} p90={pct(ath,0.9)} p99={pct(ath,0.99)}")
    print(f"threat active at REFRESH points : "
          f"{kv['thr_refresh.rows']/kv['thr_refresh.calls']:6.2f} "
          f"(king-move-conditioned, endgame-weighted)")

    # ---- KTRANS -------------------------------------------------------------
    t = ktrans(kv)
    tot_trans = sum(map(sum, t))
    print(f"\n== king transition matrix ==  total={tot_trans:,} "
          f"(kmove.total={kv['kmove.total']:,})")
    parts = {
        "mirror(file a-d)": [0 if (s & 7) <= 3 else 1 for s in range(64)],
        "rank1(back rank)": [0 if (s >> 3) == 0 else 1 for s in range(64)],
        "rank12(back two)": [0 if (s >> 3) <= 1 else 1 for s in range(64)],
        "fpair(4 cells)": [(s & 7) >> 1 for s in range(64)],
        "rankhalf(own half)": [0 if (s >> 3) <= 3 else 1 for s in range(64)],
    }
    kmove_ref = {"mirror(file a-d)": "kmove.mirror", "rank1(back rank)":
                 "kmove.rank01", "rank12(back two)": "kmove.rank12",
                 "fpair(4 cells)": "kmove.fpair",
                 "rankhalf(own half)": "kmove.rankhalf"}
    for name, p in parts.items():
        fr = flip_rate(t, p, n)
        ref = kv[kmove_ref[name]] / n
        ok = "OK" if abs(fr - ref) < 1e-9 else f"MISMATCH ref={ref:.5f}"
        print(f"  {name:20s} flips/node={fr:8.5f}  [{ok}]")

    # exact addition/replacement pricing for candidate context bits, from the
    # matrix: ADDITION = union(mirror, b) - mirror (the exact extra rebuilds a
    # second bit causes, overlap included — the richer study priced b's
    # standalone rate, which overprices where a move crosses both boundaries);
    # REPLACEMENT = b - mirror.
    def cross_rate(pred):
        return sum(t[f][g] for f in range(64) for g in range(64)
                   if pred(f, g)) / n
    mir = lambda f, g: ((f & 7) <= 3) != ((g & 7) <= 3)
    print("\n== context-bit pricing, exact (crossings/node) ==")
    print(f"  {'bit':20s} {'alone':>8s} {'add(union-mir)':>14s} "
          f"{'replace(b-mir)':>14s}")
    for name, p in parts.items():
        b = lambda f, g, p=p: p[f] != p[g]
        alone = cross_rate(b)
        union = cross_rate(lambda f, g: b(f, g) or mir(f, g))
        print(f"  {name:20s} {alone:8.5f} {union - kv['kmove.mirror']/n:14.5f} "
              f"{alone - kv['kmove.mirror']/n:+14.5f}")

    mass = [sum(t[x][s] for x in range(64)) for s in range(64)]  # arrivals
    mir_mass = sum(mass[s] for s in range(64) if (s & 7) <= 3) / max(sum(mass), 1)
    print(f"  (mirror partition arrival-mass split: {mir_mass:.2%} on a-d)")
    for share in (0.10, 0.25, 0.40):
        best = None
        for seed in (1, 2, 3):
            p, fr = anneal_bipartition(t, n, mass, share, seed=seed)
            if best is None or fr < best[1]:
                best = (p, fr)
        p, fr = best
        # orient so side 1 is the smaller-mass side
        m1 = sum(mass[s] for s in range(64) if p[s]) / max(sum(mass), 1)
        if m1 > 0.5:
            p = [1 - x for x in p]
            m1 = 1 - m1
        print(f"  min-flip 2-part (>= {share:.0%} arrival mass/side): "
              f"flips/node={fr:8.5f}  minority mass={m1:.2%} "
              f"({sum(p)} squares; best of 3 seeds)")
        for rank in range(7, -1, -1):
            row = "".join(".#"[p[rank * 8 + f]] for f in range(8))
            print(f"      {row}   rank {rank + 1}")

    # ---- budget line --------------------------------------------------------
    # Loads-model decomposition of a fold kernel's cost into per-call
    # accumulator-stream and per-row weight-stream parts (census29 shapes,
    # post-i8: i8 row = 1280 B = 40 L1d events, i16 acc read = 2560 B = 80).
    ACC_EV, ROW_EV = 80.0, 40.0
    print("\n== BUDGET LINE ==")
    print("(additions COST Elo; displacements EARN it; 'blended' includes "
          "per-call overhead,\n 'marginal' is the loads-model row share)")
    for label, c in (("loaded", c_loaded), ("quiet", C_QUIET)):
        print(f"-- denominator: {label} C={c:.0f} cycles/node --")
        for name, share, calls, rows in (
            ("threat fold", SHARE["thr_fold"], kv["thr_into.calls"],
             kv["thr_into.rows"]),
            ("halfka fold", SHARE["hka_fold"], kv["hka_into.calls"],
             kv["hka_into.rows"]),
            ("finny refresh", SHARE["finny"], kv["hka_dual.calls"],
             kv["hka_dual.rows"] + kv["hka_inplace.rows"]),
        ):
            k_cyc = share * c                       # kernel cycles/node
            rpn = rows / n
            row_share = rows * ROW_EV / (calls * ACC_EV + rows * ROW_EV)
            blended = k_cyc / rpn
            marginal = k_cyc * row_share / rpn
            print(f"  {name:14s} blended {blended:6.1f} c/row "
                  f"(= {-elo(blended, c):5.2f} Elo per extra row/node)   "
                  f"marginal {marginal:6.1f} c/row   "
                  f"per-call {k_cyc*(1-row_share)/(calls/n):6.1f} c")
        reb = SHARE["at_full"] * c * (0.01 / (kv["thr_refresh.calls"] / n))
        print(f"  threat rebuild rate: +0.01/node costs {reb:6.1f} cyc "
              f"= {-elo(reb, c):5.2f} Elo")
        named = sum(SHARE[k] for k in THREAT_CHANNEL_NAMED)
        gain = ELO_PER_DOUBLING * math.log2(1.0 / (1.0 - named))
        print(f"  threat channel, named symbols = {named:.2%} of cycles; "
              f"full displacement EARNS +{gain:.1f} Elo of speed")


if __name__ == "__main__":
    main()
