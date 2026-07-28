#!/usr/bin/env python3
"""Persistent oracle cache. Deep searches are the expensive part of every
measurement in this folder (a 1M-node probe is ~1s; a 700-position bench is
~15 min of pure oracle). Three tests this session each ran their own oracle
pass and two of them discarded the raw scores afterwards, keeping only a
derived statistic -- so the work evaporated.

Key: (fen, nodes, searchmoves) -> {"score": cp, "move": uci}. Scores are the
ORACLE ENGINE'S OWN printed cp, from the side-to-move's POV; callers normalise.
"""
import json, os, hashlib

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', 'oracle.json')

def _key(fen, nodes, searchmoves=None):
    sm = ','.join(sorted(searchmoves)) if searchmoves else ''
    return hashlib.sha1(f'{fen}|{nodes}|{sm}'.encode()).hexdigest()[:20]

def load():
    try:
        return json.load(open(PATH))
    except Exception:
        return {}

def save(cache):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    json.dump(cache, open(PATH, 'w'))

def get_or_run(cache, eng, fen, nodes, searchmoves=None):
    k = _key(fen, nodes, searchmoves)
    if k in cache:
        return cache[k]
    eng.newgame()
    g = eng.go(fen=fen, nodes=nodes, searchmoves=searchmoves) if searchmoves \
        else eng.go(fen=fen, nodes=nodes)
    if not g['ladder']:
        return None
    d = max(g['ladder'])
    cache[k] = {'score': g['ladder'][d]['score'], 'move': g['ladder'][d]['move'], 'depth': d}
    return cache[k]
