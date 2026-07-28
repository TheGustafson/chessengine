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
NET = REPO + "/data/nnue/nets/screens/d47v-a4b2-cal400.cenn"

INFO = re.compile(
    r"^info depth (\d+) .*?score (cp|mate) (-?\d+).*? nodes (\d+) .*? pv (.+)$"
)


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
        self.wait_for("uciok")
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
            m = INFO.match(line)
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


GATES_OFF = {
    "RAZOR_BASE": 30000,
    "RFP_MULT_BASE": 30000,
    "RFP_MULT_CAP": 30000,
    "PFUT_BASE": 30000,
    "CFUT_BASE": 30000,
    "QS_FUTILITY_MARGIN": 30000,
}
