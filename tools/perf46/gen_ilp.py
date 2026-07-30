#!/usr/bin/env python3
# Generate a Zen4 ILP microbenchmark: for a given instruction and a given number
# of independent accumulator chains NCH, issue OPS ops round-robin over NCH
# accumulator registers.  Cycles/op then reads:
#     NCH = 1        -> the LATENCY of the instruction
#     NCH large      -> the RECIPROCAL THROUGHPUT (1/ports)
# The knee (smallest NCH at which cycles/op stops falling) is the number of
# independent chains the machine actually wants.  That is the number study 46
# needs and nobody in this repo has measured.
import sys

INSNS = {
    # name: (mnemonic, "dst-clobber init", uses zmm30/zmm31 as the two sources)
    "vpdpbusd":   "vpdpbusd %%zmm31, %%zmm30, %%zmm{d}",
    "vfmadd231ps":"vfmadd231ps %%zmm31, %%zmm30, %%zmm{d}",
    "vpaddd":     "vpaddd %%zmm31, %%zmm{d}, %%zmm{d}",
    "vaddps":     "vaddps %%zmm31, %%zmm{d}, %%zmm{d}",
    "vpdpbusd_y": "vpdpbusd %%ymm31, %%ymm30, %%ymm{d}",
    "vfmadd231ps_y":"vfmadd231ps %%ymm31, %%ymm30, %%ymm{d}",
}

MAXCH = 16
UNROLL = 8   # each accumulator gets UNROLL ops per loop iteration

out = []
out.append('#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n')
for iname, tmpl in INSNS.items():
    for nch in range(1, MAXCH + 1):
        body = []
        for u in range(UNROLL):
            for c in range(nch):
                body.append('"' + tmpl.format(d=c) + '\\n\\t"')
        asm = "\n            ".join(body)
        out.append(f'''
static void run_{iname}_{nch}(long iters) {{
    __asm__ __volatile__ (
        "vpxorq %%zmm30, %%zmm30, %%zmm30\\n\\t"
        "vpxorq %%zmm31, %%zmm31, %%zmm31\\n\\t"
        "1:\\n\\t"
            {asm}
        "dec %0\\n\\t"
        "jnz 1b\\n\\t"
        : "+r"(iters)
        :
        : "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7",
          "zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",
          "zmm30","zmm31","cc");
}}''')

out.append('\nint main(int argc, char **argv) {')
out.append('    if (argc < 4) { fprintf(stderr, "usage: %s <insn> <nch> <iters>\\n", argv[0]); return 2; }')
out.append('    const char *n = argv[1]; int c = atoi(argv[2]); long it = atol(argv[3]);')
for iname in INSNS:
    for nch in range(1, MAXCH + 1):
        out.append(f'    if (!strcmp(n,"{iname}") && c=={nch}) {{ run_{iname}_{nch}(it); printf("%ld\\n", it*{nch}*{UNROLL}); return 0; }}')
out.append('    fprintf(stderr, "no such config\\n"); return 2;\n}')
open(sys.argv[1], "w").write("\n".join(out))
