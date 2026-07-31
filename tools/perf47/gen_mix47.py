#!/usr/bin/env python3
"""perf study 47 -- the ISSUE cost of the fold kernel's ACTUAL op sequence.

Per-op reciprocal throughputs (sweep47) are measured one instruction at a time.
They do NOT tell you what a MIX costs: `vpmovsxbw` is 1.00 c/op (one effective
512-bit pipe) and `vpaddw` is 0.55 (two), so if the two classes issue to
disjoint pipes a 1:1 mix could cost 1.00 per PAIR rather than 1.55.  That
difference decides whether deleting the widening is worth anything, so it is
measured rather than assumed.

Variants, all L1-resident so that MEMORY is not the constraint and what is left
is issue:

  shipped   vpmovsxbw zmm_t, [w]        ; vpaddw acc, acc, zmm_t   (2 ops/chunk)
  i16w      vpaddw acc, acc, [w]                                   (1 op/chunk)
  alu       vpaddw acc, acc, zmm31                                 (1 op, no mem)
  cvtonly   vpmovsxbw zmm_t, [w]                                   (1 op)

`shipped` vs `i16w` is exactly the arm this study is pricing: same arithmetic,
same number of accumulator updates, one fewer 512-bit op per chunk.

    python3 gen_mix47.py out.c ; ./mix47 <variant> <regs> <rows> <iters>
"""
import sys

REGSMAX = 8
ROWSMAX = 6
# 32 KiB L1d; keep the whole walked window well inside it.
BUFBYTES = 8192

VARIANTS = {
    # (ops per (row,chunk), template list)
    "shipped": (2, ["vpmovsxbw {off}(%%rbx), %%zmm{t}",
                    "vpaddw %%zmm{t}, %%zmm{a}, %%zmm{a}"]),
    "i16w":    (1, ["vpaddw {off}(%%rbx), %%zmm{a}, %%zmm{a}"]),
    "alu":     (1, ["vpaddw %%zmm31, %%zmm{a}, %%zmm{a}"]),
    "cvtonly": (1, ["vpmovsxbw {off}(%%rbx), %%zmm{t}"]),
    # `activate`'s ACTUAL emitted lane-iteration, transcribed from the shipped
    # disassembly at 0x70ab0 (NOT from the Rust source): two 6-op merges plus a
    # 3-op tail = 15 512-bit ops.  Measuring this replaces the assumption that
    # the fold pair's 0.873 overlap factor generalises to the head.
    # "rows" is reused as a repeat count; one unit = one lane-iteration.
    "act": (15, [
        # merge L
        "vmovdqu64 {off}(%%rbx), %%zmm{t}",
        "vpsllw $1, {off}(%%rbx), %%zmm{a}",
        "vpmullw {off}(%%rbx), %%zmm{t}, %%zmm{t}",
        "vpaddsw %%zmm{a}, %%zmm{t}, %%zmm{t}",
        "vpmaxsw %%zmm31, %%zmm{t}, %%zmm{t}",
        "vpminsw %%zmm31, %%zmm{t}, %%zmm{t}",
        # merge R
        "vmovdqu64 {off}(%%rbx), %%zmm{a}",
        "vpsllw $1, {off}(%%rbx), %%zmm{b}",
        "vpmullw {off}(%%rbx), %%zmm{a}, %%zmm{a}",
        "vpaddsw %%zmm{b}, %%zmm{a}, %%zmm{a}",
        "vpmaxsw %%zmm31, %%zmm{a}, %%zmm{a}",
        "vpminsw %%zmm31, %%zmm{a}, %%zmm{a}",
        # tail
        "vpsllw $7, %%zmm{t}, %%zmm{t}",
        "vpmulhuw %%zmm{t}, %%zmm{a}, %%zmm{t}",
        "vpmovwb %%zmm{t}, {off}(%%rbx)",
    ]),
}

out = ["#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n",
       f"static char buf[{BUFBYTES}] __attribute__((aligned(64)));\n"]

for vname, (opsper, tmpls) in VARIANTS.items():
    for regs in range(1, REGSMAX + 1):
        for rows in range(1, ROWSMAX + 1):
            body, slot = [], 0
            for r in range(rows):
                for k in range(regs):
                    # acc registers zmm0..zmm(regs-1); temps above them
                    a = k
                    t = regs + (k % (16 - regs)) if regs < 16 else 0
                    b = regs + ((k + 1) % (16 - regs)) if regs < 16 else 1
                    off = (slot * 64) % (BUFBYTES - 64)
                    for tm in tmpls:
                        body.append('"' + tm.format(off=off, a=a, t=t, b=b) + '\\n\\t"')
                    slot += 1
            asm = "\n            ".join(body)
            clob = ",".join(f'"zmm{i}"' for i in range(16)) + ',"zmm31"'
            out.append(f'''
static void run_{vname}_{regs}_{rows}(long iters) {{
    __asm__ __volatile__ (
        "vpxorq %%zmm31, %%zmm31, %%zmm31\\n\\t"
        "1:\\n\\t"
            {asm}
        "dec %0\\n\\t"
        "jnz 1b\\n\\t"
        : "+r"(iters) : "b"(buf)
        : {clob},"cc","memory");
}}''')

out.append('\nint main(int argc, char **argv) {')
out.append('    if (argc < 5) { fprintf(stderr, "usage: %s <variant> <regs> <rows> <iters>\\n", argv[0]); return 2; }')
out.append('    const char *v = argv[1]; int g = atoi(argv[2]), r = atoi(argv[3]); long it = atol(argv[4]);')
out.append('    memset(buf, 1, sizeof buf);')
for vname, (opsper, _t) in VARIANTS.items():
    for regs in range(1, REGSMAX + 1):
        for rows in range(1, ROWSMAX + 1):
            # report CHUNK-UPDATES (row x chunk), not raw ops: that is the unit
            # of work the kernel has to do, and it is what the variants share.
            out.append(f'    if (!strcmp(v,"{vname}") && g=={regs} && r=={rows}) '
                       f'{{ run_{vname}_{regs}_{rows}(it); printf("%ld\\n", it*{regs}*{rows}); return 0; }}')
out.append('    fprintf(stderr, "no such config\\n"); return 2;\n}')

open(sys.argv[1], "w").write("\n".join(out))
