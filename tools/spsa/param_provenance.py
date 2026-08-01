#!/usr/bin/env python3
"""Provenance + coverage audit for ARCHON's tunables (D20/D45).

Answers three questions that kept getting answered wrong by hand:

1. **Which Stockfish commit are our constants actually from?**  `src/search/
   params.rs`'s header names one (`ebcea3ef`).  This checks it, because
   reading the WRONG SF revision has now cost us twice: once when the clone
   sat at `sf_16` and a bare grep returned SF16 constants, and once when it
   moved to `sf_18` — which is OLDER than our source, so `sf_18` is just as
   wrong a reference, in the other direction.
   Razoring is the canary: 456/252 at sf_16, 485/281 at sf_18, 465/300 at
   eca43a97, 483/318 at ebcea3ef — and 483/318 is what we ship (the
   2026-08-01 reference refresh).

2. **Did we transcribe faithfully?**  Screens every tunable's value against
   every integer literal in SF's search/movepick/evaluate/timeman sources at
   the reference commit.  A value >= MIN_INTERESTING that appears NOWHERE in
   SF is either an ARCHON original or a transcription drift; small values are
   skipped because they collide trivially.  This is a high-recall screen, not
   a proof: it does not check that a value sits in the RIGHT formula.  Spot-
   check in place before acting on a hit.

3. **Which tunables has SPSA never touched?**  Cross-references
   `tools/spsa/campaigns/*.json`.  An uncovered tunable is not automatically
   suspect (many are unit-defining denominators that are deliberately pinned),
   but it is where a bad value would survive undetected.

Stdlib only.  Run from the repo root:  python3 tools/spsa/param_provenance.py
"""

import argparse
import json
import os
import re
import subprocess
import sys

SF_DIR = "stockfish_reference/Stockfish"
REF_DEFAULT = "ebcea3ef"
SF_FILES = [
    "src/search.cpp",
    "src/movepick.cpp",
    "src/evaluate.cpp",
    "src/timeman.cpp",
    "src/history.h",
    "src/search.h",
    "src/movepick.h",
]
PARAM_FILES = ["src/search/params.rs", "src/eval/params.rs", "src/timeman.rs"]
CAMPAIGN_GLOB = "tools/spsa/campaigns"

# Below this, an integer collides with SF by accident (loop bounds, small
# multipliers, ply counts). 20 is where the screen starts discriminating.
MIN_INTERESTING = 20

# The canary: (name, our value, expected value at the reference commit).
# Razoring moved at sf_16->sf_18, sf_18->eca43a97 AND eca43a97->ebcea3ef, so
# it detects a reference slip in any direction.
CANARIES = [("RAZOR_BASE", 483), ("RAZOR_DEPTH_MULT", 318)]


def sf_show(ref, path):
    r = subprocess.run(
        ["git", "-C", SF_DIR, "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
    )
    return r.stdout if r.returncode == 0 else ""


def parse_tunables():
    """(name, type, raw-default, file) for every `tunables!` entry."""
    rows = []
    for f in PARAM_FILES:
        if not os.path.exists(f):
            continue
        src = open(f).read()
        for name, ty, val in re.findall(
            r"^\s*pub ([A-Z_0-9]+)\s*:\s*(\w+)\s*=\s*([^;]+);", src, re.M
        ):
            rows.append((name, ty, val.strip(), f))
    return rows


def as_int(raw):
    """Evaluate a simple integer default (`465`, `1 << 28`, `-14000`)."""
    try:
        v = eval(raw, {"__builtins__": {}}, {})
    except Exception:
        return None
    return v if isinstance(v, int) else None


def campaign_coverage():
    covered = {}
    if not os.path.isdir(CAMPAIGN_GLOB):
        return covered
    for fn in sorted(os.listdir(CAMPAIGN_GLOB)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(CAMPAIGN_GLOB, fn)))
        for p in d.get("params", []):
            covered.setdefault(p["name"], []).append(fn[:-5])
    return covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF_DEFAULT, help="SF commit our params came from")
    ap.add_argument("--verbose", action="store_true", help="list every uncovered tunable")
    args = ap.parse_args()

    rows = parse_tunables()
    by_name = {n: (ty, val, f) for n, ty, val, f in rows}
    fail = False

    # --- 1. reference commit -------------------------------------------------
    print(f"== reference: SF {args.ref}")
    sf_src = "".join(sf_show(args.ref, f) for f in SF_FILES)
    if not sf_src:
        print(f"  ERROR: cannot read SF sources at {args.ref} (clone missing?)")
        return 2
    desc = subprocess.run(
        ["git", "-C", SF_DIR, "log", "-1", "--format=%h %ad %s", args.ref],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"  {desc}")
    head = subprocess.run(
        ["git", "-C", SF_DIR, "describe", "--tags", "--always", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"  clone HEAD is at: {head}")
    if head and args.ref not in head:
        print(
            f"  NOTE: HEAD != reference. A bare grep of the checkout returns "
            f"{head}'s constants, NOT the ones we transcribed. Read the "
            f"reference explicitly: git -C {SF_DIR} show {args.ref}:src/search.cpp"
        )

    for name, expect in CANARIES:
        got = as_int(by_name[name][1]) if name in by_name else None
        ok = got == expect
        fail |= not ok
        print(f"  canary {name:20s} ours={got} expected@{args.ref}={expect}  {'ok' if ok else 'MISMATCH'}")

    # --- 2. transcription screen --------------------------------------------
    sf_ints = set(int(x) for x in re.findall(r"(?<![\w.])(\d+)(?![\w.])", sf_src))
    print(f"\n== transcription screen ({len(rows)} tunables, |value| >= {MIN_INTERESTING})")
    hits = []
    for name, ty, val, f in rows:
        if ty == "f64":
            continue
        n = as_int(val)
        if n is None or abs(n) < MIN_INTERESTING:
            continue
        # A shift-written constant is compared both ways: SF may write the
        # shift too, in which case the product never appears as a literal.
        if abs(n) in sf_ints or (("<<" in val) and all(
            int(t.strip()) in sf_ints for t in val.split("<<")
        )):
            continue
        hits.append((name, val, f))
    print(f"  values with NO occurrence anywhere in SF@{args.ref}: {len(hits)}")
    for name, val, f in hits:
        print(f"    {name:34s} = {val:>12s}   [{os.path.basename(f)}]")
    if not hits:
        print("    (none — every value traces to the reference commit)")

    # --- 3. SPSA coverage ----------------------------------------------------
    covered = campaign_coverage()
    stale = sorted(n for n in covered if n not in by_name)
    uncovered = [n for n, _, _, _ in rows if n not in covered]
    print(f"\n== SPSA campaign coverage")
    print(f"  covered   : {len(rows) - len(uncovered)} / {len(rows)}")
    print(f"  UNCOVERED : {len(uncovered)}")
    if stale:
        fail = True
        print(f"  STALE campaign entries (named param does not exist): {len(stale)}")
        for n in stale:
            print(f"    {n}  <- in {', '.join(covered[n])}")
    if args.verbose:
        for n in uncovered:
            print(f"    uncovered: {n}")

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
