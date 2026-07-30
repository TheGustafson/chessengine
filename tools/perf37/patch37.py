#!/usr/bin/env python3
"""perf study 37 — apply one of the attribute variants to `sota_simd.rs`.

    python3 tools/perf37/patch37.py <base|nocold|noinl|noboth>

The names are the study's arm labels, describing which of the two attributes
`i8_deltas_into_generic` carries:

    base    #[inline(never)] + #[cold]     the text as of `5e42622`
    nocold  #[inline(never)]               ** THE SHIPPED TEXT since study 37 **
    noinl                     #[cold]      (byte-identical to `base`)
    noboth  neither

**`base` is no longer "restore the tree".** Study 37 landed `nocold`, so to get
back to the shipped state use `git checkout -- src/eval/nnue/sota_simd.rs`,
which is what `build37.sh` does and cannot go stale.

The point of driving this from a script at all is that the binaries differ by
EXACTLY the attribute lines and nothing else: a hand-edit between builds is how
a study ends up comparing two trees that also disagree somewhere it forgot
about.
"""

import pathlib
import sys

SRC = pathlib.Path("src/eval/nnue/sota_simd.rs")
ANCHOR = "    unsafe fn i8_deltas_into_generic("
TF = '    #[target_feature(enable = "avx512f,avx512bw")]\n'
VARIANTS = {
    "base": TF + "    #[inline(never)]\n    #[cold]\n",
    "nocold": TF + "    #[inline(never)]\n",
    "noinl": TF + "    #[cold]\n",
    "noboth": TF,
}


def main() -> None:
    mode = sys.argv[1]
    want = VARIANTS[mode]
    text = SRC.read_text()
    i = text.index(ANCHOR)
    # Walk back over the attribute block that precedes the fn signature.
    j = i
    for _ in range(4):
        k = text.rindex("\n", 0, j - 1) + 1
        if not text[k:j].lstrip().startswith("#["):
            break
        j = k
    have = text[j:i]
    assert have in VARIANTS.values(), f"unexpected attribute block:\n{have!r}"
    if have == want:
        print(f"[patch37] {mode}: already applied")
        return
    SRC.write_text(text[:j] + want + text[i:])
    print(f"[patch37] {mode}: {have.strip()!r} -> {want.strip()!r}")


if __name__ == "__main__":
    main()
