#!/usr/bin/env python3
"""Static (net-only) eval of a position, from BOTH engines, in one frame.

TWO CONVENTIONS THAT DO NOT MATCH — this is the whole reason this file exists:

  ARCHON : "Final evaluation: 37 (internal, from <side-to-move>'s perspective)"
           -> SIDE-TO-MOVE POV, and the LABEL CHANGES with the side to move.
  SF18   : "Final evaluation  +0.15 (white side)"
           -> ALWAYS WHITE POV, in pawns.

On 2026-07-28 a bench hardcoded white's POV for both. That sign-flipped ARCHON on
every black-to-move position and produced ARCHON corr = 0.032 (vs SF 0.905) --
i.e. "our eval is noise", a false catastrophe. Sign-corrected, the same data gives
MAE 97.1 vs 97.2 and corr 0.899 vs 0.905: a dead heat, with OUR p90 error LOWER
(149.5 vs 176). This is the second time the perspective trap has been hit here
(see LEDGER REFUTED, "Our eval prints the wrong perspective"), so it is now a
function instead of a comment.

FRAME: ARCHON prints raw internal ~2.07x SF's printed cp (uciprobe R2).
Everything returned here is SF PRINTED CP, WHITE'S POV.
"""
import re, subprocess

ARCHON_PER_SF_PRINTED = 2.07
_A = re.compile(r'Final evaluation:\s*(-?\d+)')
_S = re.compile(r'Final evaluation\s*([+-]?\d+\.\d+)')


def _drive(cmd, opts, fens, pat):
    """Persistent stdin — `printf | engine` closes the pipe and aborts (trap #1)."""
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    send = lambda x: (p.stdin.write(x + '\n'), p.stdin.flush())
    def wait(tok):
        buf = []
        while True:
            l = p.stdout.readline()
            if not l:
                return buf
            buf.append(l.rstrip())
            if l.startswith(tok):
                return buf
    send('uci'); wait('uciok')
    for o in opts:
        send(o)
    send('isready'); wait('readyok')
    out = []
    for fen in fens:
        send(f'position fen {fen}'); send('eval'); send('isready')
        m = pat.search('\n'.join(wait('readyok')))
        out.append(float(m.group(1)) if m else None)
    send('quit')
    return out


def archon(fens, net='engines/canonical/net.cenn', binary='engines/canonical/archon'):
    """-> SF printed cp, WHITE POV. Flips side-to-move POV; divides the 2.07 frame."""
    raw = _drive([binary], [f'setoption name EvalFile value {net}',
                            'setoption name Hash value 64'], fens, _A)
    out = []
    for fen, v in zip(fens, raw):
        if v is None:
            out.append(None); continue
        v = v / ARCHON_PER_SF_PRINTED
        out.append(v if fen.split()[1] == 'w' else -v)   # <- THE FIX
    return out


def sf(fens, binary='engines/bin/stockfish18'):
    """-> SF printed cp, WHITE POV. Already white-POV; pawns -> cp."""
    raw = _drive([binary], ['setoption name Hash value 64'], fens, _S)
    return [None if v is None else v * 100 for v in raw]
