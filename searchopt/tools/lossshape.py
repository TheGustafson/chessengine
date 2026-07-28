#!/usr/bin/env python3
"""WHY DID WE LOSE — per-game eval trajectories straight out of the match PGN.

The blunder frame answered itself on the D48X corpus: 0.46% of our moves in
still-playable positions clear the wp>=0.06 bar, 44% of THOSE are oracle
artifacts, and the whole 404-game match yields ONE actionable position. So the
question is no longer "which move was bad" but "what shape does a loss have".

This reads the per-move eval annotations fastchess already writes
(`{+1.19/20 0.485s}`), so it costs no engine time and covers ALL 404 games --
draws included, which the miner never touched (it only scored LOST games).

FRAME (CLAUDE.md R2). Each engine reports its OWN eval in its OWN units, and
ARCHON prints ~2.07x Stockfish's cp. A raw comparison of the two columns is
meaningless -- exactly the mistake that produced a phantom "266 cp gap" three
times in one session. `to_sf` normalises ARCHON's side at the parse boundary,
and once it does the two engines agree closely (archon -2.61 -> -1.26 against
SF's -1.19 on the first game's move 8).

Signs are normalised to ARCHON'S POINT OF VIEW throughout: positive = good for
us, whichever colour we had.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys

ARCHON_PER_SF_PRINTED = 2.07          # uciprobe.ARCHON_PER_SF_PRINTED
MOVE = re.compile(r'\{([+-]?(?:M?\d+\.?\d*|M\d+))/(\d+)\s+([\d.]+)s\}')
MATE = re.compile(r'^([+-])M(\d+)$')


def to_pawns(tok: str, engine: str) -> float:
    """One annotation token -> pawns in SF's PRINTED frame, mover's POV."""
    m = MATE.match(tok)
    if m:
        return 100.0 if m.group(1) == '+' else -100.0
    v = float(tok)
    return v / ARCHON_PER_SF_PRINTED if engine == 'archon' else v


def games(path: str):
    txt = open(path, errors='ignore').read()
    for b in re.findall(r'(\[Event[\s\S]*?)(?=\[Event|\Z)', txt):
        h = dict(re.findall(r'\[(\w+) "([^"]*)"\]', b))
        if 'White' not in h:
            continue
        body = b[b.rfind(']') + 1:]
        ours_white = h.get('White') == 'archon'
        res = h.get('Result', '*')
        score = 0.5 if res == '1/2-1/2' else (1.0 if (res == '1-0') == ours_white else 0.0)
        # Annotations alternate White, Black, White, ... from the book position.
        # The book FEN's side-to-move sets who moves first.
        stm_white = h.get('FEN', 'w').split()[1] == 'w' if 'FEN' in h else True
        traj = []
        for i, m in enumerate(MOVE.finditer(body)):
            mover_white = stm_white if i % 2 == 0 else not stm_white
            engine = 'archon' if mover_white == ours_white else 'sf18'
            cp = to_pawns(m.group(1), engine)
            # mover's POV -> OUR POV
            ours = cp if engine == 'archon' else -cp
            traj.append({'i': i, 'engine': engine, 'ours_pov': ours,
                         'depth': int(m.group(2)), 'time': float(m.group(3))})
        yield {'headers': h, 'ours_white': ours_white, 'result': res,
               'score': score, 'traj': traj}


def summarise(g):
    """Shape of one game, from OUR point of view, in SF-frame pawns."""
    ours = [t['ours_pov'] for t in g['traj'] if t['engine'] == 'archon']
    sf = [t['ours_pov'] for t in g['traj'] if t['engine'] == 'sf18']
    if not ours or not sf:
        return None
    n = min(len(ours), len(sf))
    # Peak advantage we ever held, per SF's own opinion (the honest judge).
    peak_sf, trough_sf = max(sf[:n]), min(sf[:n])
    # Where did WE think we stood vs where SF did (delusion, mean over the game).
    delusion = sum(ours[i] - sf[i] for i in range(n)) / n
    return {'plies': n, 'start_sf': sf[0], 'end_sf': sf[-1],
            'peak_sf': peak_sf, 'trough_sf': trough_sf,
            'delusion': delusion, 'our_start': ours[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pgn', default='engines/runs/corpus-d48x/games.pgn')
    ap.add_argument('--result', choices=['loss', 'draw', 'win', 'all'], default='all')
    ap.add_argument('--list', action='store_true', help='one line per game')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    rows = []
    for g in games(a.pgn):
        s = summarise(g)
        if not s:
            continue
        tag = {1.0: 'win', 0.5: 'draw', 0.0: 'loss'}[g['score']]
        if a.result != 'all' and tag != a.result:
            continue
        rows.append((tag, s, g))
    print(f'{len(rows)} games ({a.result})')

    if a.list:
        print(f"{'#':>4} {'res':<5}{'col':<6}{'ply':>5}{'startSF':>9}{'peakSF':>8}"
              f"{'troughSF':>10}{'endSF':>8}{'delusion':>10}")
        for i, (tag, s, g) in enumerate(rows[:a.limit or len(rows)]):
            print(f"{i:>4} {tag:<5}{'white' if g['ours_white'] else 'black':<6}"
                  f"{s['plies']:>5}{s['start_sf']:>9.2f}{s['peak_sf']:>8.2f}"
                  f"{s['trough_sf']:>10.2f}{s['end_sf']:>8.2f}{s['delusion']:>10.2f}")
        return 0

    # Aggregate shape -- corpus CHARACTERISATION only (README §6: fine once,
    # never as a diagnosis instrument).
    for tag in ('win', 'draw', 'loss'):
        sub = [s for t, s, _ in rows if t == tag]
        if not sub:
            continue
        med = lambda k: sorted(x[k] for x in sub)[len(sub) // 2]
        print(f'\n{tag.upper()} (n={len(sub)}):')
        print(f'  median start {med("start_sf"):+.2f}  peak {med("peak_sf"):+.2f}  '
              f'trough {med("trough_sf"):+.2f}  end {med("end_sf"):+.2f}')
        print(f'  median delusion (ours - SF, our POV): {med("delusion"):+.2f} pawns')
        held = sum(1 for x in sub if x['peak_sf'] >= 1.0)
        print(f'  games where SF says we once stood >= +1.00: {held} '
              f'({100*held/len(sub):.0f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
