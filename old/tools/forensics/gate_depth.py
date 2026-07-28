"""How many PLIES does each pruning gate cost on a position we got wrong?

WHY a depth-of-discovery instrument instead of the usual on/off flip: "gate X
off changes the move at depth 12" is a binary, and a binary cannot be ranked or
compared across positions.  What we actually want to know is how much SEARCH
RESOLUTION each gate is buying or destroying here -- so measure, for every gate
configuration, the first depth at which the engine picks the right move AND
HOLDS it to the end of the ladder.  The difference from control is the gate's
cost in plies, which is additive-ish, comparable between positions, and
convertible to Elo through the ~120 Elo per doubling constant.

Holding matters: a move that is briefly touched at depth 9 and abandoned was
never found.  The in-game search would not have played it.

Arms: control, each gate off alone (what does removing it buy), each gate on
alone (is it sufficient by itself to cause the miss), and all off.

    .venv/bin/python tools/forensics/gate_depth.py FEN TARGET_MOVE [--max-depth 26]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uciprobe import Engine  # noqa: E402

REPO = "/home/nick/projects/chessengine"
OURS = REPO + "/engines/bin/archon-d47v-tune-pgo"
OFF = 30000

# name -> (knob, off-value).  Defaults are NOT written here: they are read from
# the binary with `<engine> tunables`.  WHY: hardcoding them once produced a
# table in which three of eight "defaults" were invented (NMP_R_BASE 4658 for a
# real 7, RED_TABLE_NUM 1155 for 2834, LMP_BASE 3387 for 3), so the CONTROL arm
# was a wildly perturbed engine and every delta in the table was meaningless --
# while looking perfectly quantitative.  A control you typed is not a control.
#
# RFP is switched by its depth cap, not by inflating its margin: the margin form
# keeps firing through the improving/opp_worsening path even with a huge base,
# which is what made an early game-18 probe read "RFP is innocent".
GATES = {
    "razor": ("RAZOR_BASE", OFF),
    "rfp": ("RFP_MAX_DEPTH", 0),
    "pfut": ("PFUT_BASE", OFF),
    "cfut": ("CFUT_BASE", OFF),
    "qsfut": ("QS_FUTILITY_MARGIN", OFF),
    "lmp": ("LMP_BASE", 10_000_000),
    "lmr": ("RED_TABLE_NUM", 1),
    "nmp": ("NMP_R_BASE", 0),
}


def engine_defaults(binary: str) -> dict[str, str]:
    """Ground truth for every tunable, straight out of the binary."""
    import subprocess

    out = subprocess.run([binary, "tunables"], capture_output=True, text=True,
                         cwd=REPO).stdout
    d = {}
    for line in out.splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            d[k.strip()] = v.strip()
    if not d:
        raise SystemExit(f"{binary}: no `tunables` output -- is it a --features tune build?")
    return d


def d_find(eng: Engine, fen: str, target: str, max_depth: int) -> tuple[int | None, dict]:
    """First depth from which `target` is chosen and never abandoned."""
    eng.newgame()
    r = eng.go(fen=fen, depth=max_depth)
    lad = r["ladder"]
    first = None
    for d in sorted(lad):
        if lad[d]["move"] == target:
            if first is None:
                first = d
        else:
            first = None
    return first, lad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fen")
    ap.add_argument("target")
    ap.add_argument("--max-depth", type=int, default=26)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--gates", default="", help="comma-separated subset")
    ap.add_argument("--arm", action="append", default=[],
                    help="extra arm, NAME:KNOB=VAL[,KNOB=VAL...]; repeatable. "
                         "Use with --gates '' to run ONLY the custom arms.")
    ap.add_argument("--only-arms", action="store_true",
                    help="skip the standard gate arms, run control + --arm only")
    args = ap.parse_args()

    # a copy: the reset loop below adds entries, and GATES is imported by
    # gate_census -- mutating a shared module global is how probes get haunted
    gates = dict(GATES) if not args.gates else {k: GATES[k] for k in args.gates.split(",")}
    dflts = engine_defaults(OURS)
    missing = [k for _n, (k, _o) in gates.items() if k not in dflts]
    if missing:
        raise SystemExit(f"not tunables in this binary: {missing}")
    print("defaults from binary: "
          + ", ".join(f"{k}={dflts[k]}" for _n, (k, _o) in gates.items()), flush=True)

    allon = {k: off for _n, (k, off) in gates.items()}
    arms = [("control", {})]
    if not args.only_arms:
        arms += [(f"-{n}", {k: off}) for n, (k, off) in gates.items()]
        arms += [("ALL-off", allon)]
        # leave-one-in: everything off except this gate, which stays at default
        for n, (k, _off) in gates.items():
            o = dict(allon)
            o[k] = dflts[k]
            arms.append((f"only+{n}", o))
    for spec in args.arm:
        label, _, body = spec.partition(":")
        opts = {}
        for kv in body.split(","):
            k, _, v = kv.partition("=")
            k = k.strip()
            if k not in dflts:
                raise SystemExit(f"--arm {label}: '{k}' is not a tunable")
            opts[k] = v.strip()
        arms.append((label, opts))
    # every custom knob must also be reset to its default between arms
    for _l, o in arms:
        for k in o:
            if k not in [g[0] for g in gates.values()]:
                gates.setdefault("_x" + k, (k, dflts[k]))

    e = Engine(OURS, name="archon", hash_mb=args.hash)
    e.assert_net_live()

    base = None
    print(f"{'arm':<12}{'d_find':>7}{'delta':>7}{'nodes@find':>12}  ladder tail")
    for name, opts in arms:
        for _n, (k, _off) in gates.items():
            e.setopt(k, dflts[k])
        for k, v in opts.items():
            e.setopt(k, v)
        d, lad = d_find(e, args.fen, args.target, args.max_depth)
        if name == "control":
            base = d
        nd = lad[d]["nodes"] if d else 0
        delta = "" if (d is None or base is None) else f"{d - base:+d}"
        tail = " ".join(f"{lad[x]['move']}" for x in sorted(lad)[-6:])
        print(f"{name:<12}{str(d) if d else 'never':>7}{delta:>7}{nd:>12}  {tail}",
              flush=True)
    e.close()


if __name__ == "__main__":
    main()
