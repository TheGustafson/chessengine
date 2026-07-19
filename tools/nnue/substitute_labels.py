#!/usr/bin/env python3
"""Stream-substitute deep (50k) labels into the full binpack for a finetune.

Reads fast_fens output on stdin (`FEN\\tresult\\tply\\tscore5k`, white-relative)
and emits `FEN | cp | wdl` on stdout for `nnue_convert convert-stdin`. For each
position: wdl = the game result; cp = the SF18@50k label (UCI cp) if we have one
for that FEN, else the binpack's 5k score converted to the trainer's scale
(score5k / SFBINPACK_SCORE_DIV). Relabeled positions can be UPSAMPLED (emitted N
times) so their signal isn't diluted in the 188M-position corpus.

This makes a LARGE (no-overfit), SAME-distribution corpus that differs from the
incumbent's own training data only in the deep-relabeled positions — the clean
substrate for a finetune that actually tests whether the 50k labels help.

    fast_fens BINPACK 200000000 | substitute_labels.py <deep_corpus.txt> <upsample> [score_div]
                                                        | nnue_convert convert-stdin out.bin
"""
import sys

def main():
    deep_path = sys.argv[1]
    upsample = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    score_div = float(sys.argv[3]) if len(sys.argv) > 3 else 1.9

    # FEN -> deep cp50k (white-relative UCI centipawns)
    deep = {}
    for line in open(deep_path):
        p = [x.strip() for x in line.split("|")]
        if len(p) >= 2:
            try:
                deep[p[0]] = int(round(float(p[1])))
            except ValueError:
                pass
    print(f"deep labels: {len(deep)}  upsample={upsample}  score_div={score_div}", file=sys.stderr)

    RES2WDL = {"1.0": "1.0", "0.5": "0.5", "0.0": "0.0"}
    out = sys.stdout
    n = subs = 0
    w = out.write
    for line in sys.stdin:
        f = line.rstrip("\n").split("\t")
        if len(f) < 4:
            continue
        fen, res, cp5 = f[0], f[1], f[3]
        wdl = RES2WDL.get(res)
        if wdl is None:
            continue
        d = deep.get(fen)
        if d is not None:
            row = f"{fen} | {d} | {wdl}\n"
            for _ in range(upsample):
                w(row)
            subs += 1
            n += upsample
        else:
            try:
                cp = int(round(int(cp5) / score_div))
            except ValueError:
                continue
            w(f"{fen} | {cp} | {wdl}\n")
            n += 1
    print(f"emitted {n} rows ({subs} substituted x{upsample})", file=sys.stderr)

if __name__ == "__main__":
    main()
