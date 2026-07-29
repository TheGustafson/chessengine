#!/usr/bin/env bash
# perf study 18 — mutation verification.
#
#   bash mut18.sh <label> <python-patch-file>
#
# Applies a mutation to the tree, runs the FAST sota test tier, records which
# test (if any) caught it, and reverts. Study 17 method rule 6: COMMIT BEFORE
# YOU MUTATE — the revert is `git checkout -- src/` and it eats uncommitted
# work.
#
# The compile is checked by cargo's EXIT STATUS, not by grepping for "error" in
# the combined output: a FAILING TEST prints `error` lines of its own (a panic
# message, an `assertion failed`), so the grep form reported "DID-NOT-COMPILE"
# for the mutations that worked — i.e. it silently inverted the verdict on
# exactly the runs that matter. Same class as study 17 §6.2.1's parser bug.
set -uo pipefail
LABEL=${1:?label}
PATCHPY=${2:?python patch}
OUT=docs/research/perf-artifacts-18/mutations.txt
python3 "$PATCHPY" || { echo "$LABEL  PATCH-FAILED" | tee -a "$OUT"; exit 1; }

if ! systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 -- \
     tools/oomlast.sh nice -n 10 cargo test --features sota,arch-d48x \
     --target-dir target-chk18 --lib --no-run > /dev/null 2>&1; then
  git checkout -- src/
  echo "$LABEL  DID-NOT-COMPILE" | tee -a "$OUT"; exit 0
fi
res=$(systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 -- \
  tools/oomlast.sh nice -n 10 cargo test --features sota,arch-d48x \
  --target-dir target-chk18 --lib -- --test-threads=2 2>&1)
git checkout -- src/
failed=$(grep -oP '^test \K\S+(?= \.\.\. FAILED)' <<<"$res" | tr '\n' ' ')
if [ -z "$failed" ]; then
  echo "$LABEL  *** SURVIVED *** (no test failed)" | tee -a "$OUT"
else
  echo "$LABEL  caught by: $failed" | tee -a "$OUT"
fi
