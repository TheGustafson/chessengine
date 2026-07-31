#!/usr/bin/env python3
"""perf study 48 -- STATIC load census of a symbol.

Counts, per symbol, the instructions that carry a MEMORY SOURCE operand
(i.e. that dispatch a load) and classifies them by width and by base register.
Static counts are exact for the code; they become dynamic when multiplied by a
call count, which is what makes this different from a sampled profile.

  python3 tools/perf48/static48.py <binary> <symbol-substring> [...]
"""
import re, subprocess, sys
from collections import Counter

STORE_ONLY = re.compile(r"^(mov[a-z0-9]*|vmov[a-z0-9]*)\s+%[a-z0-9]+,\s*[^%].*\(")
# an operand that is a memory reference
MEMOP = re.compile(r"(-?(0x[0-9a-f]+)?\(%[a-z0-9]+(,%[a-z0-9]+(,[1248])?)?\)|0x[0-9a-f]+\(%rip\))")

def disas(binary):
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", binary],
                         capture_output=True, text=True).stdout
    syms, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m:
            cur = m.group(1); syms[cur] = []
            continue
        if cur is None: continue
        m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
        if m:
            syms[cur].append((m.group(1), m.group(2), m.group(3).split("#")[0].strip()))
    return syms

def classify(mn, ops):
    """Return (n_load_ops, n_store_ops) for one instruction."""
    if not MEMOP.search(ops):
        return 0, 0
    if mn.startswith(("lea", "nop", "prefetch")):
        return 0, 0
    parts = [p.strip() for p in re.split(r",(?![^(]*\))", ops)]
    # AT&T: last operand is the destination
    dst = parts[-1] if parts else ""
    src = parts[:-1]
    st = 1 if MEMOP.search(dst) else 0
    ld = 1 if any(MEMOP.search(p) for p in src) else 0
    # read-modify-write (add %reg,(%mem)) both loads and stores
    if st and mn.startswith(("add", "sub", "or", "and", "xor", "inc", "dec", "lock")):
        ld = 1
    # push reads a register and stores; pop loads
    if mn.startswith("push"): return 0, 1
    if mn.startswith("pop"):  return 1, 0
    if mn.startswith("call"): return 0, 0
    return ld, st

def main():
    binary = sys.argv[1]
    wants = sys.argv[2:]
    syms = disas(binary)
    for want in wants:
        hits = [s for s in syms if want in s]
        for s in hits:
            ins = syms[s]
            nl = ns = 0
            widths = Counter(); bases = Counter()
            for _, mn, ops in ins:
                l, t = classify(mn, ops)
                nl += l; ns += t
                if l:
                    w = "zmm" if "%zmm" in ops else ("ymm" if "%ymm" in ops else
                         ("xmm" if "%xmm" in ops else "gpr"))
                    widths[w] += 1
                    m = re.search(r"\(%([a-z0-9]+)", ops)
                    if m: bases[m.group(1)] += 1
            print(f"{s}")
            print(f"    instructions {len(ins):5d}   LOADS {nl:5d}   stores {ns:5d}")
            print(f"    load widths: {dict(widths)}")
            print(f"    top bases:   {dict(bases.most_common(8))}")
            # spill detection: loads/stores based on %rsp or %rbp
            spl = sum(1 for _, mn, ops in ins
                      if classify(mn, ops)[0] and re.search(r"\(%r[sb]p", ops))
            sps = sum(1 for _, mn, ops in ins
                      if classify(mn, ops)[1] and re.search(r"\(%r[sb]p", ops))
            print(f"    STACK-slot (spill/reload) loads {spl}, stores {sps}")
            print()

main()
