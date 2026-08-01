#!/usr/bin/env python3
"""perf study 64 — split a symbol's IBS latency by WHAT IT WAS LOADING.

    python3 tools/perf64/foldsrc64.py <ibs-script.txt> <binary> [symbol-substr]

Study 61 ranked symbols by tag-to-retire latency and study 63 ruled that the
question is "what fraction of TRAFFIC does a rung convert".  Neither could be
answered *inside* the fold, because a symbol share cannot say whether the
stalled load was a WEIGHT ROW (1280 bytes out of a 77.7 MB table — the thing
study 61's capacity model is about) or the ACCUMULATOR STREAM (the parent
entry, 2560 bytes out of a 1.3 MB stack, which study 61 §1 measured as
L2-resident and therefore cheap).  Those two have completely different levers
and they live in the same symbol.

This tool disassembles the binary, classifies every instruction *offset* inside
the matching symbols by the memory stream it touches, and then charges IBS's
per-op tag-to-retire weight to that class.  IBS tags at ISSUE and reports its
own op's rip, so there is no skid and no attribution guess.

Classification, for the i8 fold kernels:

  WEIGHT ROW     `vpmovsxbw <mem>`  — the only instruction that reads a row
  SRC acc load   `vpaddw`/`vpsubw`/`vmovdqu64` reading memory
  DST acc store  a `vmovdqu64 %zmmN,(...)` into the destination accumulator

IBS charges a nonzero tag-to-retire weight only to ops that MISS L1, so the
latency columns are an L1-miss service profile and DST stores read exactly
zero — the write side of the fold is invisible to this instrument.  Say so when
quoting it: "the stores are free" is NOT what this measures.

Input is whatever `tools/perf59/ibs59.sh` writes (`-F ip,sym,symoff,weight,
data_src`).
"""

import collections
import re
import subprocess
import sys

SAMPLE = re.compile(r"\|OP (\w+)\|LVL ([^|]*)\|.*?BLK\s+\S+\s+(\d+)\s+([0-9a-f]+)\s+(.*)$")
FUNC = re.compile(r"^([0-9a-f]+) <(.*)>:$")
INSN = re.compile(r"\s*([0-9a-f]+):\t(.*)")


def classify(ins: str) -> str:
    mnemonic = ins.split()[0] if ins else "?"
    if "(" not in ins:
        return "other/" + mnemonic
    if mnemonic == "vpmovsxbw":
        return "WEIGHT ROW"
    if mnemonic in ("vpaddw", "vpsubw"):
        return "SRC acc load"
    if mnemonic in ("vmovdqu64", "vmovdqa64", "vmovntdq"):
        # A store writes a register INTO memory: `%zmmN,<disp>(<base>)`.
        return "DST acc store" if re.search(r"%zmm\d+,\S*\(", ins) else "SRC acc load"
    return "other/" + mnemonic


def disassemble(binary: str, want: str) -> dict[tuple[str, int], str]:
    text = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", "-C", binary],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    at: dict[tuple[str, int], str] = {}
    i = 0
    while i < len(text):
        m = FUNC.match(text[i])
        if not m or want not in m.group(2):
            i += 1
            continue
        base, name = int(m.group(1), 16), m.group(2)
        j = i + 1
        while j < len(text) and text[j].strip():
            mm = INSN.match(text[j])
            if mm:
                at[(name, int(mm.group(1), 16) - base)] = mm.group(2)
            j += 1
        i = j
    return at


def main() -> None:
    ibs, binary = sys.argv[1], sys.argv[2]
    want = sys.argv[3] if len(sys.argv) > 3 else "i8_into_shaped"
    at = disassemble(binary, want)
    lat: collections.Counter[str] = collections.Counter()
    ops: collections.Counter[str] = collections.Counter()
    lvl: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    engine = 0
    for line in open(ibs):
        m = SAMPLE.search(line)
        if not m:
            continue
        weight, sym = int(m.group(3)), m.group(5).strip()
        engine += weight
        if want not in sym:
            continue
        mo = re.search(r"^(.*)\+0x([0-9a-f]+)$", sym)
        cls = classify(at.get((mo.group(1), int(mo.group(2), 16)), "")) if mo else "UNMAPPED"
        lat[cls] += weight
        ops[cls] += 1
        lvl[cls][m.group(2).strip()] += weight
    total = sum(lat.values())
    if not total:
        print(f"# no samples matched {want!r}")
        return
    print(f"# {want}: {total} tag-to-retire cycles = {100 * total / engine:.2f}% of the engine")
    print(f"{'class':16s} {'sampled':>8s} {'latency':>10s} {'%sym':>7s} {'%engine':>8s}   by level")
    for cls, value in lat.most_common():
        share = (
            ", ".join(f"{k} {100 * x / value:.0f}%" for k, x in lvl[cls].most_common(3))
            if value
            else ""
        )
        print(
            f"{cls:16s} {ops[cls]:8d} {value:10d} {100 * value / total:6.2f}% "
            f"{100 * value / engine:7.2f}%   {share}"
        )


if __name__ == "__main__":
    main()
