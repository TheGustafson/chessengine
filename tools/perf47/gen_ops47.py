#!/usr/bin/env python3
"""perf study 47 -- measure 512-bit reciprocal throughput for the ops ARCHON's
hot kernels ACTUALLY issue.

Study 46 measured `vpdpbusd` / `vfmadd231ps` / `vaddps` (1.00 c/op) and
`vpaddd` (0.55 c/op) and left the bracket 13 points wide because the word-wise
ops -- `vpmovsxbw`, `vpaddw`, `vpmullw`, `vpaddsw`, ... -- were never measured.
This generator is study 46's `gen_ilp.py` with (a) the real opcode inventory,
read off the shipped disassembly by `census47.py`, and (b) memory-source forms,
because `vpmovsxbw zmm, ymmword ptr [mem]` is how the fold kernels ship and a
register-form number would not be the one the engine pays.

Two of study 46's instructions are kept verbatim as CONTROLS: reproducing
1.00 c/op for `vfmadd231ps` and 0.55 c/op for `vpaddd` on this harness is what
makes the new rows believable.

    python3 gen_ops47.py out.c
    gcc -O2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mavx512vbmi2 \
        -o ops47 out.c
    ./ops47 <insn> <nchains> <iters>      # prints the op count
"""

import sys

# form:
#   "acc" -- op zmm31, zmm30, zmmD   (dst is also a source: a dependency chain)
#   "bin" -- op zmm31, zmmD, zmmD    (dst accumulates)
#   "un"  -- op zmm31, zmmD          (dst written fresh: no chain, pure throughput)
#   "mem" -- op (%rbx,off), zmmD     (memory source, dst written fresh)
INSNS = {
    # ---- study 46 controls, verbatim -------------------------------------
    "vfmadd231ps": ("acc", "vfmadd231ps %%zmm31, %%zmm30, %%zmm{d}"),
    "vpdpbusd": ("acc", "vpdpbusd %%zmm31, %%zmm30, %%zmm{d}"),
    "vpaddd": ("bin", "vpaddd %%zmm31, %%zmm{d}, %%zmm{d}"),
    # ---- the fold kernels: i8_deltas_into*, finny_refresh_into ------------
    # 450 vpmovsxbw vs 450 vpaddw/vpsubw in i8_deltas_into_generic.
    "vpmovsxbw": ("un", "vpmovsxbw %%ymm31, %%zmm{d}"),
    "vpmovsxbw_m": ("mem", "vpmovsxbw {off}(%%rbx), %%zmm{d}"),
    "vpaddw": ("bin", "vpaddw %%zmm31, %%zmm{d}, %%zmm{d}"),
    "vpaddw_m": ("mem2", "vpaddw {off}(%%rbx), %%zmm{d}, %%zmm{d}"),
    "vpsubw": ("bin", "vpsubw %%zmm31, %%zmm{d}, %%zmm{d}"),
    "vmovdqu64_m": ("mem", "vmovdqu64 {off}(%%rbx), %%zmm{d}"),
    # ---- forward_into: the head -------------------------------------------
    "vpsllw": ("un", "vpsllw $2, %%zmm31, %%zmm{d}"),
    "vpmullw": ("bin", "vpmullw %%zmm31, %%zmm{d}, %%zmm{d}"),
    "vpmulhuw": ("bin", "vpmulhuw %%zmm31, %%zmm{d}, %%zmm{d}"),
    "vpaddsw": ("bin", "vpaddsw %%zmm31, %%zmm{d}, %%zmm{d}"),
    "vpmaxsw": ("bin", "vpmaxsw %%zmm31, %%zmm{d}, %%zmm{d}"),
    "vpminsw": ("bin", "vpminsw %%zmm31, %%zmm{d}, %%zmm{d}"),
    "vpmovwb": ("un", "vpmovwb %%zmm31, %%ymm{d}"),
    "vpbroadcastd": ("un", "vpbroadcastd %%xmm31, %%zmm{d}"),
    # NB: `{`/`}` are GCC extended-asm dialect-alternative delimiters and must
    # be escaped `%{` / `%}`; unescaped they gave "junk `%k1' after register".
    "vpcompressw": ("un", "vpcompressw %%zmm31, %%zmm{d}%{{%%k1%}}"),
    "vptestmd": ("k", "vptestmd %%zmm31, %%zmm30, %%k{d}"),
    "vcvtdq2ps": ("un", "vcvtdq2ps %%zmm31, %%zmm{d}"),
    "vpermt2b": ("bin", "vpermt2b %%zmm31, %%zmm30, %%zmm{d}"),
    # ---- 256-bit controls: the double-pump signature ----------------------
    "vpmovsxbw_y": ("un", "vpmovsxbw %%xmm31, %%ymm{d}"),
    "vpaddw_y": ("bin", "vpaddw %%ymm31, %%ymm{d}, %%ymm{d}"),
}

MAXCH = 12
UNROLL = 8
# 1 KiB of L1-resident source data, walked at 64 B stride so no two ops in an
# unrolled group hit the same line's worth of forwarding.
BUFBYTES = 1024

out = ["#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n",
       f"static char buf[{BUFBYTES}] __attribute__((aligned(64)));\n"]

for iname, (form, tmpl) in INSNS.items():
    for nch in range(1, MAXCH + 1):
        # `k` form only has k1..k7 available
        maxreg = 7 if form == "k" else 16
        if nch > maxreg:
            continue
        body = []
        slot = 0
        for u in range(UNROLL):
            for c in range(nch):
                off = (slot * 64) % (BUFBYTES - 64)
                body.append('"' + tmpl.format(d=c, off=off) + '\\n\\t"')
                slot += 1
        asm = "\n            ".join(body)
        clob = ",".join(f'"zmm{i}"' for i in range(16)) + ',"zmm30","zmm31"'
        clob += "," + ",".join(f'"k{i}"' for i in range(1, 8))
        out.append(f'''
static void run_{iname}_{nch}(long iters) {{
    __asm__ __volatile__ (
        "vpxorq %%zmm30, %%zmm30, %%zmm30\\n\\t"
        "vpxorq %%zmm31, %%zmm31, %%zmm31\\n\\t"
        "kxnorw %%k1, %%k1, %%k1\\n\\t"
        "1:\\n\\t"
            {asm}
        "dec %0\\n\\t"
        "jnz 1b\\n\\t"
        : "+r"(iters)
        : "b"(buf)
        : {clob},"cc","memory");
}}''')

out.append('\nint main(int argc, char **argv) {')
out.append('    if (argc < 4) { fprintf(stderr, "usage: %s <insn> <nch> <iters>\\n", argv[0]); return 2; }')
out.append('    const char *n = argv[1]; int c = atoi(argv[2]); long it = atol(argv[3]);')
out.append('    memset(buf, 1, sizeof buf);')
for iname, (form, _t) in INSNS.items():
    maxreg = 7 if form == "k" else 16
    for nch in range(1, min(MAXCH, maxreg) + 1):
        out.append(f'    if (!strcmp(n,"{iname}") && c=={nch}) {{ run_{iname}_{nch}(it); printf("%ld\\n", it*{nch}*{UNROLL}); return 0; }}')
out.append('    fprintf(stderr, "no such config\\n"); return 2;\n}')

open(sys.argv[1], "w").write("\n".join(out))
