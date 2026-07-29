#!/usr/bin/env python3
"""Offline anatomy of a CENN-v2 net: parameter accounting, neck-rank analysis,
and the HalfKA-vs-threat contribution split.

Pure numpy over the net file — NO engine, NO search, NO GPU. Runs in seconds and
does not contend with a perf study.

Usage:  .venv/bin/python docs/research/nnue-arch-artifacts/net_anatomy.py <net.cenn> [...]
"""

from __future__ import annotations

import struct
import sys

import numpy as np

HDR = 64


def parse(path):
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:4] == b"CEN2", path
    (ver,) = struct.unpack_from("<H", raw, 4)
    (flags,) = struct.unpack_from("<H", raw, 6)
    hk, th, acc, l2, l3, ob = struct.unpack_from("<IIIIII", raw, 8)
    ftq, l1q, fts, scale = struct.unpack_from("<iiii", raw, 32)
    (plen,) = struct.unpack_from("<Q", raw, 48)
    o = HDR
    def take(n, dt):
        nonlocal o
        a = np.frombuffer(raw, dtype=dt, count=n, offset=o)
        o += n * np.dtype(dt).itemsize
        return a
    halfka_w = take(hk * acc, "<i2").reshape(hk, acc)
    ft_bias = take(acc, "<i2")
    threat_w = take(th * acc, "i1").reshape(th, acc) if (flags & 1) else None
    l1_w = take(ob * l2 * acc, "i1").reshape(ob, l2, acc)
    l1_b = take(ob * l2, "<f4").reshape(ob, l2)
    l2_w = take(ob * l2 * l3, "<f4").reshape(ob, l2, l3)
    l2_b = take(ob * l3, "<f4").reshape(ob, l3)
    l3_w = take(ob * l3, "<f4").reshape(ob, l3)
    l3_b = take(ob, "<f4")
    assert o - HDR == plen, (o - HDR, plen)
    # optional trailers
    psqt = skip = None
    tail = raw[o:]
    if len(tail) >= 12 and tail[:4] == b"PSQ1":
        nb, rows = struct.unpack_from("<II", tail, 4)
        psqt = np.frombuffer(tail, dtype="<i4", count=rows * nb, offset=12).reshape(rows, nb)
        tail = tail[12 + rows * nb * 4:]
    if len(tail) >= 12 and tail[:4] == b"SKP1":
        nb, hl = struct.unpack_from("<II", tail, 4)
        skip = np.frombuffer(tail, dtype="<f4", count=hl * nb, offset=12).reshape(nb, hl)
    return dict(path=path, ver=ver, flags=flags, halfka_dims=hk, threat_dims=th,
                acc=acc, l2=l2, l3=l3, buckets=ob, ftq=ftq, l1q=l1q, ftshift=fts,
                scale=scale, halfka_w=halfka_w, ft_bias=ft_bias, threat_w=threat_w,
                l1_w=l1_w, l1_b=l1_b, l2_w=l2_w, l2_b=l2_b, l3_w=l3_w, l3_b=l3_b,
                psqt=psqt, skip=skip, nbytes=len(raw))


def eff_rank(sv):
    """Participation ratio of the squared spectrum: exp(entropy of sigma^2)."""
    p = sv ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def report(n):
    print("=" * 78)
    print(f"{n['path']}")
    print(f"  arch  acc={n['acc']} L2={n['l2']} L3={n['l3']} buckets={n['buckets']} "
          f"halfka={n['halfka_dims']} threats={n['threat_dims']} scale={n['scale']}")
    hk, th, acc, l2, l3, ob = (n['halfka_dims'], n['threat_dims'], n['acc'],
                               n['l2'], n['l3'], n['buckets'])
    ft_params = hk * acc + (th * acc if n['threat_w'] is not None else 0) + acc
    head_params = ob * (l2 * acc + l2 + l2 * l3 + l3 + l3 + 1)
    psqt_params = n['psqt'].size if n['psqt'] is not None else 0
    skip_params = n['skip'].size if n['skip'] is not None else 0
    tot = ft_params + head_params + psqt_params + skip_params
    print(f"  params  FT {ft_params/1e6:8.2f} M | head {head_params/1e3:8.1f} k | "
          f"PSQ1 {psqt_params/1e3:8.1f} k | skip {skip_params/1e3:6.1f} k | "
          f"TOTAL {tot/1e6:.2f} M  ({n['nbytes']/1e6:.1f} MB)")
    print(f"  head share of params: {100*head_params/tot:.4f} %")
    # per-eval touched params (one bucket)
    touched = l2 * acc + l2 + l2 * l3 + l3 + l3 + 1 + (acc if n['skip'] is not None else 0)
    print(f"  params TOUCHED per eval (1 bucket, dense view): {touched:,}")

    # ---- neck rank -------------------------------------------------------
    print("  -- L1 weight spectrum per bucket (l2 x acc, dequantised) --")
    ers = []
    for b in range(ob):
        w = n['l1_w'][b].astype(np.float64) / n['l1q']
        sv = np.linalg.svd(w, compute_uv=False)
        er = eff_rank(sv)
        ers.append(er)
        frac = (sv[:min(8, len(sv))] ** 2).sum() / (sv ** 2).sum()
        print(f"     b{b}: eff_rank {er:6.2f} / {l2:3d}  ({100*er/l2:5.1f}%)   "
              f"top-8 energy {100*frac:5.1f}%   sigma1/sigmaN {sv[0]/sv[-1]:7.1f}")
    print(f"     mean eff_rank {np.mean(ers):.2f} / {l2} = {100*np.mean(ers)/l2:.1f}%")

    # L2 (l2 x l3) spectrum -- the neck as seen by the next layer
    ers2 = []
    for b in range(ob):
        w = n['l2_w'][b].astype(np.float64)
        sv = np.linalg.svd(w, compute_uv=False)
        ers2.append(eff_rank(sv))
    print(f"  -- L2 weight (l2 x l3) mean eff_rank {np.mean(ers2):.2f} / {min(l2, l3)} "
          f"= {100*np.mean(ers2)/min(l2, l3):.1f}%")

    # ---- FT channel contribution ----------------------------------------
    hkw = n['halfka_w'].astype(np.float32)
    print("  -- feature-transformer channel weight scale (quantised units) --")
    print(f"     HalfKA  |w| mean {np.abs(hkw).mean():7.3f}  p99 {np.percentile(np.abs(hkw),99):7.2f}  "
          f"max {np.abs(hkw).max():7.0f}   (QA={n['ftq']})")
    if n['threat_w'] is not None:
        thw = n['threat_w'].astype(np.float32)
        print(f"     threat  |w| mean {np.abs(thw).mean():7.3f}  p99 {np.percentile(np.abs(thw),99):7.2f}  "
              f"max {np.abs(thw).max():7.0f}   (QT=64)")
        # put both in the HalfKA frame: threat weights are quantised at 64,
        # HalfKA at 255, so a threat weight is worth 255/64 = 3.984 HalfKA units.
        r = 255.0 / 64.0
        # expected accumulator contribution = mean|w| * expected active count
        n_hk, n_th = 32.0, 128.0   # design/11 12.1: max active per perspective
        c_hk = np.abs(hkw).mean() * n_hk
        c_th = np.abs(thw).mean() * r * n_th
        print(f"     E[|contribution|] per accumulator lane, upper-bound active counts "
              f"(HalfKA<=32, threats<=128):")
        print(f"       HalfKA {c_hk:8.2f}   threats {c_th:8.2f}   "
              f"threat share {100*c_th/(c_hk+c_th):.1f}%   (random-sign sqrt model: "
              f"{100*(np.abs(thw).mean()*r*np.sqrt(n_th))/(np.abs(thw).mean()*r*np.sqrt(n_th)+np.abs(hkw).mean()*np.sqrt(n_hk)):.1f}%)")
        # per-lane norms
        hk_lane = np.abs(hkw).mean(axis=0)
        th_lane = np.abs(thw).mean(axis=0) * r
        dom = (th_lane * np.sqrt(n_th) > hk_lane * np.sqrt(n_hk)).mean()
        print(f"       lanes where the threat channel dominates (sqrt model): {100*dom:.1f}%")
        # dead lanes
        print(f"       accumulator lanes with ~zero HalfKA mass: "
              f"{(hk_lane < 0.01*hk_lane.mean()).sum()}; ~zero threat mass: "
              f"{(th_lane < 0.01*th_lane.mean()).sum()}")

    # ---- skip lane --------------------------------------------------------
    if n['skip'] is not None:
        sk = n['skip'].astype(np.float64)
        print(f"  -- output-skip lane: |w| mean {np.abs(sk).mean():.5f}, "
              f"per-bucket L2 norm {np.round(np.linalg.norm(sk, axis=1), 3).tolist()}")
    if n['psqt'] is not None:
        ps = n['psqt'].astype(np.float64) / 256.0   # 8 fractional bits -> cp
        print(f"  -- PSQ1 rows: {ps.shape}, |v| mean {np.abs(ps).mean():.3f} cp, "
              f"p99 {np.percentile(np.abs(ps),99):.2f} cp, max {np.abs(ps).max():.1f} cp")
        hk_rows = ps[:n['halfka_dims']]
        th_rows = ps[n['halfka_dims']:]
        print(f"     HalfKA rows |v| mean {np.abs(hk_rows).mean():.3f} cp; "
              f"threat rows |v| mean {np.abs(th_rows).mean():.4f} cp")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(parse(p))
