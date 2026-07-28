perf study 9 — regenerating the session-local artifacts

`perf.data` files and the 7 MB `objdump-s9base.txt` were deleted before commit
(same rule as studies 5/6/8 — they are large and fully regenerable). Everything
below reproduces them.

  # flat instruction profile (also writes perf-<label>-instructions.data)
  bash scripts/flat8.sh s9base 500000 . \
    ./engines/bin/archon-s9-base bench 16 1 100000 \
    tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn

  # the disassembly the static counts and libcalls9.py read
  objdump -d -C --no-show-raw-insn engines/bin/archon-s9-base > objdump-s9base.txt
  python3 scripts/libcalls9.py objdump-s9base.txt   # 2560-byte mem* call sites

  # per-symbol static instruction counts (the "counting" instrument)
  # -> see the inline python in the study doc §2; it walks the objdump above.

  # rebuild the two binaries this study compares
  bash scripts/arm9.sh base     # study 8 HEAD  (git checkout 6b42e44 first)
  bash scripts/arm9.sh final    # this branch's HEAD

NOTE on `perf annotate`: it needs the matching perf.data, so regenerate the
flat profile first. Study 8's finding still holds and was re-confirmed here —
annotate percent on stores and loads is SKID on this box (it valued one
`vmovdqu64` at 58.30% of `make_move_with`). Use it to LOCATE a phase, then read
the instruction sequence and count; never price a change with it.

NOTE on the SMP scripts: use `smpuci9.py`. `smp9.sh` drives `bench`, whose
`threads` argument is parsed and never used (src/bench.rs), so it silently
measures one thread at every point. It is kept only as the record of that trap.
