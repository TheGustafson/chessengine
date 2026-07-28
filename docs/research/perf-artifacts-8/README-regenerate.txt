# perf.data files are session-local and were deleted before commit (studies
# 5/6 artifact dirs did not survive for the same reason). Regenerate with
#   bash scripts/flat8.sh <label> 500000 . <cmd...>
# and, for the srcline/addr2line work, a line-table PGO build:
#   CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
#   RUSTFLAGS="-C target-cpu=native -C profile-use=<merged.profdata>" \
#   cargo build --release --features sota,arch-d48x --target-dir target-pgo-dbg
