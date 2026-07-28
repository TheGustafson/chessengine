"""Persistent-UCI probe harness.

WHY a persistent process rather than the `printf | sleep | engine` idiom used
so far: our engine treats stdin EOF as a STOP, so the sleep-based form is a
race between the sleep and the search, and it silently truncates a search that
runs long.  Holding stdin open and reading until `bestmove` is exact, and it is
also ~10x faster because one `uci`/net-load is amortised over many probes.

Everything here is deterministic (fixed depth / fixed nodes), so it is safe to
run at nice 19 beside anything (playbook 1.3).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = "/home/nick/projects/chessengine"
NET = REPO + "/engines/canonical/net.cenn"

# ---------------------------------------------------------------------------
# FRAME.  Our engine prints RAW INTERNAL values; Stockfish's `to_cp` divides its
# internal value by `a = p_a(material)` ~ 385 before printing
# (eca43a97:src/uci.cpp:554).  So OUR printed score is ~2.07x SF's printed cp,
# and a raw comparison of the two is meaningless.
#
# On 2026-07-28 this was diagnosed, written into a design doc -- and then the
# SAME mistake was made three more times in the same session, manufacturing a
# "266 cp gap at depth 1" and a "400 cp PV disagreement" that were both entirely
# this factor.  A fact in a doc gets forgotten; a helper at the tool boundary
# cannot.  CLAUDE.md rule R2: never hand-compare cross-engine scores.
ARCHON_PER_SF_PRINTED = 2.07


def to_sf_frame(score, engine="archon"):
    """Put a printed score into Stockfish's PRINTED-cp frame.

    engine="archon" divides by 2.07; engine="sf" is already there.  Use this on
    BOTH sides of every cross-engine comparison -- including the ones that look
    obviously safe, which is exactly where it went wrong before.
    """
    if score is None:
        return None
    if engine in ("sf", "stockfish", "sf18"):
        return float(score)
    if engine in ("archon", "ours"):
        return float(score) / ARCHON_PER_SF_PRINTED
    raise ValueError(f"unknown engine frame {engine!r}")


def to_sf_internal(printed_cp, fen):
    """SF's PRINTED cp -> SF's INTERNAL units, the frame its search constants
    (razoring 465, LMR 52/9, ...) actually live in.  internal = printed * a/100.
    Needed whenever comparing a transcribed constant against an eval value."""
    w = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}
    mat = sum(w.get(c.lower(), 0) for c in fen.split()[0] if c.isalpha())
    m = max(17, min(78, mat)) / 58.0
    a = ((-72.32565836 * m + 185.93832038) * m - 144.58862193) * m + 416.44950446
    return printed_cp * a / 100.0

INFO = re.compile(
    r"^info depth (\d+) .*?score (cp|mate) (-?\d+).*? nodes (\d+) .*? pv (.+)$"
)

# WHY this filter matters more than it looks.  An aspiration re-search emits
# `score cp X upperbound` / `lowerbound` lines, which are BOUNDS, not values.
# Recording them as exact scores is harmless for a depth-limited ladder (the
# last line at a completed depth is exact) but catastrophic for a NODE-limited
# search, whose final iteration is usually cut off mid-window: the last line is
# then a fail-low bound.  That is how a 100k-node probe of game 18 read "-40,
# only 24 cp lost" when the true value was about -228 -- and it nearly got the
# whole gift corpus thrown out as blind.
BOUND = re.compile(r"\b(lowerbound|upperbound)\b")


class Engine:
    def __init__(self, path, net=NET, hash_mb=16, options=None, name=""):
        self.name = name or os.path.basename(path)
        self.p = subprocess.Popen(
            ["nice", "-n", "19", path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=REPO,
        )
        self.send("uci")
        # Keep the banner: it is the only place an engine states a tunable's
        # DEFAULT, and a null arm has to nudge the real default rather than a
        # value read out of a source file that may have moved on.
        self.uci_lines = self.wait_for("uciok")
        self.net_line = None
        if net:
            self.send(f"setoption name EvalFile value {net}")
        self.send(f"setoption name Hash value {hash_mb}")
        for k, v in (options or {}).items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.net_line = self.wait_for("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait_for(self, tok):
        buf = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name}: engine died waiting for {tok}")
            buf.append(line.rstrip())
            if line.startswith(tok):
                return buf

    def setopt(self, k, v):
        self.send(f"setoption name {k} value {v}")

    def assert_net_live(self):
        """Playbook 1.1 -- presence of a success line, not absence of an error."""
        self.send("isready")
        # the eval line is printed at uci time; re-ask via 'd' which prints it
        blob = "\n".join(self.wait_for("readyok"))
        if "ERROR" in blob:
            raise SystemExit(f"{self.name}: EvalFile load ERROR:\n{blob}")
        return blob

    def newgame(self):
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok")

    def go(self, fen=None, moves=None, depth=None, nodes=None, movetime=None,
           searchmoves=None, startpos=False):
        pos = "position " + ("startpos" if startpos else f"fen {fen}")
        if moves:
            pos += " moves " + " ".join(moves)
        self.send(pos)
        cmd = "go"
        if depth:
            cmd += f" depth {depth}"
        if nodes:
            cmd += f" nodes {nodes}"
        if movetime:
            cmd += f" movetime {movetime}"
        if searchmoves:
            cmd += " searchmoves " + " ".join(searchmoves)
        self.send(cmd)
        ladder = {}
        best = None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name}: died during go")
            line = line.rstrip()
            if line.startswith("bestmove"):
                best = line.split()[1]
                break
            m = None if BOUND.search(line) else INFO.match(line)
            if m:
                d, kind, sc, nd, pv = m.groups()
                d = int(d)
                v = int(sc) if kind == "cp" else (
                    (30000 - abs(int(sc))) * (1 if int(sc) > 0 else -1)
                )
                # keep the LAST info at each depth (post-research value)
                ladder[d] = dict(depth=d, score=v, mate=(kind == "mate"),
                                 nodes=int(nd), pv=pv.split(), move=pv.split()[0])
        return dict(best=best, ladder=ladder)

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


# The five eval gates, DISABLED.
#
# WHY RFP is switched off with a depth cap and not with a huge margin: the
# obvious recipe `RFP_MULT_BASE = RFP_MULT_CAP = 30000` is NOT a disable.  The
# margin is
#
#     futility_mult*depth - (2934*improving + 343*opp_worsening)*futility_mult/1024
#
# so the trend discount is ~2.87 * futility_mult.  Blow futility_mult up to
# 30000 and at depth 1-2 the discount (~86k) dwarfs `futility_mult*depth`
# (30-60k) and the margin goes hugely NEGATIVE -- which turns RFP into an
# UNCONDITIONAL prune at every shallow improving node instead of turning it
# off.  Measured on the game-18 position (2026-07-28): the mult=30000 arm and
# the two genuine disables disagree on the chosen move AND on node count
# (152k vs 119k at depth 15), and the mult arm "recovered" the correct move
# only as a side effect of pruning MORE somewhere else.  Every gate attribution
# made with the old recipe has to be re-measured.
#
# `RFP_MAX_DEPTH = 0` makes the `depth < RFP_MAX_DEPTH` guard unsatisfiable and
# is a true disable.  It is bit-identical (move, score AND node count at d12 /
# d15 / d16) to the independent disable that raises the mult while zeroing the
# two trend weights, which is how it was verified.
#
# The other four gates are additive margins compared against alpha with no
# subtractive companion term, so raising them really does disable them.
GATES_OFF = {
    "RAZOR_BASE": 30000,
    "RFP_MAX_DEPTH": 0,
    "PFUT_BASE": 30000,
    "CFUT_BASE": 30000,
    "QS_FUTILITY_MARGIN": 30000,
}

# The defaults for everything GATES_OFF (and the per-family arms below) touch,
# so a driver that reuses one engine process can restore a clean baseline
# between arms instead of trusting the previous arm to have been additive.
GATES_DEFAULT = {
    "RAZOR_BASE": 465,
    "RFP_MAX_DEPTH": 17,
    "RFP_MULT_BASE": 40,
    "RFP_MULT_CAP": 80,
    "PFUT_BASE": 40,
    "CFUT_BASE": 231,
    "QS_FUTILITY_MARGIN": 335,
}

# Single-family disables, for the bisect rung.  Same caveat as above: prefer a
# structural guard (a depth/lmr-depth cap) over a huge margin wherever the
# margin has a subtractive term.
GATE_FAMILIES = {
    "razor": {"RAZOR_BASE": 30000},
    "rfp": {"RFP_MAX_DEPTH": 0},
    "pfut": {"PFUT_BASE": 30000},
    "cfut": {"CFUT_BASE": 30000},
    "qsfut": {"QS_FUTILITY_MARGIN": 30000},
}

# A semantically NULL perturbation: one unit on a constant that cannot matter.
# WHY it belongs in the harness: on the game-18 position, `RAZOR_BASE 465->464`
# changes the move the engine plays at depth 17.  The argmax of a single search
# is a chaotic function of the pruning constants, so "arm X flips this position"
# is worth nothing without the null band beside it.  Run these, see what the
# baseline flip rate is, and only believe an arm that beats it.
NULL_ARMS = {
    "null_razor": {"RAZOR_BASE": 464},
    "null_cfut": {"CFUT_BASE": 232},
    "null_qsfut": {"QS_FUTILITY_MARGIN": 336},
}
