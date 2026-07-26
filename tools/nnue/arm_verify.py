#!/usr/bin/env python3
"""arm_verify — double-verify a fixed-node A/B arm by analyzing its GAMES,
not just its Elo line (protocol: user directive 2026-07-26 ~08:0x).

For a knob-vs-base self-play PGN (same net both sides):
  1. Elo + Ptnml recap from the stdout log.
  2. DEPTH DIFFERENTIAL: at fixed nodes the knob side should reach DEEPER
     (that is the entire theory of the change). Mean completed depth per
     move, knob vs base — if the knob prunes harder but does NOT get deeper,
     the saved nodes evaporated into re-searches: reject regardless of Elo.
  3. LOSS TAXONOMY of the knob side (loss_forensics, same slope both sides):
     a TACTICAL share spike vs the base side's own losses = unsound cuts —
     route to the guards branch (wip/nmp-guards), do not just retune.
  4. Termination sanity: adjudication/maxmoves/repetition histogram, both
     colors — a knob that wins only via adjudication drift is suspect.

Usage: arm_verify.py data/nnue/experiments/nmpsweep-<tag> <tag>
"""
import re, subprocess, sys, statistics as st
from collections import Counter

d, tag = sys.argv[1], sys.argv[2]
pgn = f"{d}/games.pgn"

print(f"══ {tag} ══")
for line in open(f"{d}/stdout.log"):
    if line.startswith(("Elo:", "Games:", "Ptnml")):
        print("  " + line.rstrip())

# ---- depth differential ----
raw = open(pgn, errors="ignore").read()
chunks = [c for c in re.split(r"\n\n(?=\[Event )", raw) if "{" in c]
TOK = re.compile(r"\{[-+]?\d+(?:\.\d+)?/(\d+)\s+[\d.]+s\}")
dep = {tag: [], "base": []}
for c in chunks:
    hdr = dict(re.findall(r'\[(\w+) "([^"]*)"\]', c))
    fen = hdr.get("FEN", "")
    white_first = (" w " in fen) if fen else True
    wt = hdr.get("White", "")
    depths = [int(x) for x in TOK.findall(c)]
    # interleaved: index parity 0 = first mover = White iff white_first
    first = wt if white_first else hdr.get("Black", "")
    other = hdr.get("Black", "") if white_first else wt
    for i, dp in enumerate(depths):
        who = first if i % 2 == 0 else other
        key = tag if who == tag else "base"
        dep[key].append(dp)
if dep[tag] and dep["base"]:
    a, b = st.mean(dep[tag]), st.mean(dep["base"])
    print(f"  depth@50k: {tag} {a:.2f}  base {b:.2f}  diff {a-b:+.2f} "
          f"({'DEEPER ✓' if a > b + 0.05 else 'NOT DEEPER — reject theory'})")

# ---- terminations ----
term = Counter(re.search(r'\[Termination "([^"]*)"\]', c).group(1)
               if '[Termination' in c else "checkmate/other" for c in chunks)
print("  terminations:", dict(term.most_common(5)))

# ---- loss taxonomy, both sides ----
for side in (tag, "base"):
    r = subprocess.run([sys.executable, "tools/nnue/loss_forensics.py", pgn,
                        "--us", side, "--winner-slope", "0.548",
                        "--base-ms", "10000"],
                       capture_output=True, text=True)
    tax = [l for l in r.stdout.splitlines()
           if re.match(r"\s+(misjudgment|tactical|grind|timepressure|collapse|unclassified)", l)]
    n = re.search(r"L (\d+)", r.stdout)
    print(f"  {side} losses ({n.group(1) if n else '?'}): "
          + "; ".join(t.split()[0] + " " + t.split()[1] for t in tax[:4]))
