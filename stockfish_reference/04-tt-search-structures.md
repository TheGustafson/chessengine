# Transposition table and search-support structures

Source: `tt.{h,cpp}`, `search.h`, `thread.h`, `memory.{h,cpp}`.

## TTEntry: 10 bytes (tt.cpp:37-88)

| Field | Bits | Notes |
|---|---|---|
| key16 | 16 | low 16 bits of Zobrist key = in-cluster tag |
| depth8 | 8 | stored as `depth - DEPTH_NONE`, so 0 == empty slot |
| genBound8 | 8 | generation(5 bits) \| bound(2) \| is_pv(1) |
| move16 | 16 | best move |
| value16 | 16 | search score |
| eval16 | 16 | static eval |

- The full key is NOT stored: high bits are implied by the cluster index,
  low 16 are the tag. Rare collisions are accepted.
- Every field is a relaxed atomic (`memory_order_relaxed`). Per-field loads
  can't tear, but a read entry may be **self-inconsistent** across fields
  (value from one writer, move from another) — tolerated by design. The old
  key^data XOR trick is gone.
- Rust: `#[repr(C)]` struct of `AtomicU16/AtomicU8/AtomicI16`, all accesses
  `Ordering::Relaxed`. No locks, no seqlock. Saturate depth arithmetic
  (`max(0, ...)`) to survive racy read-modify-writes.

## Cluster: 3 entries + 2 pad = 32 bytes (tt.cpp:150-161)

Exactly half a cache line; a probe touches one line. `static_assert` pins the
size — do the same with a Rust `const _: () = assert!(size_of::<Cluster>() == 32)`.

**Replacement** (probe miss): evict the entry minimizing
`depth8 - 8 * relative_age`. Age = `(current_gen - entry_gen) & 0x1F`
(5-bit modular wraparound, "like clocks"). One generation of age outweighs
8 plies of depth.

**save() policy**: keep existing move unless new one supplied or key differs.
Overwrite if EXACT bound, key differs, `new_depth + 2*is_pv > old_depth - 4`,
or entry is from an older generation. Otherwise decrement stale decisive
(mate/TB) scores' depth so they age out.

## Indexing: multiply, not modulo (tt.cpp:277)

`cluster = mul_hi64(key, clusterCount)` = high 64 bits of the 128-bit
product. Maps the key uniformly onto [0, count) for ANY count (no
power-of-two requirement), one mul instead of a div.
Rust: `((key as u128 * count as u128) >> 64) as usize`.

## Prefetch

- `search.cpp:637`: prefetch child's TT line BEFORE do_move using
  `pos.prefetch_key(move)` (cheap approximate next-key; wrong for
  castling/ep/promo, harmlessly).
- `position.cpp:992`: prefetch inside do_move once the real key is known.
- Rust: `core::arch::x86_64::_mm_prefetch` (or
  `std::intrinsics` via a tiny wrapper), no-op fallback on other arches.

## Mate score adjustment (search.cpp:1852-1881)

- `value_to_tt`: mate scores stored relative to the CURRENT node
  (`v + ply` for wins) so a TT entry is position-relative, not root-relative.
- `value_from_tt`: inverse, PLUS downgrade mate scores that could be false
  under the 50-move rule (`VALUE_MATE - v > 100 - rule50`).

## Large pages (memory.cpp:103-174)

TT allocated with 2 MB alignment + `madvise(MADV_HUGEPAGE)` (Linux THP), or
explicit 1 GB hugetlb pages when the table is big enough. Rationale: TT
probes are random cache misses; huge pages slash TLB misses — measurable
Elo. TT clear is multithreaded and NUMA-aware (first-touch page placement).
Rust: `mmap` via libc + madvise on Linux; do this when the TT lands, it's
cheap to add.

## Search stack & root moves (search.h:104-156)

Per-ply `Stack`: pv pointer, continuation-history POINTERS (into the tables,
so updates to shallower plies are cheap), ply, currentMove, excludedMove,
staticEval, statScore, moveCount, inCheck, ttPv, ttHit, cutoffCnt, reduction.
**No killers** — removed from modern SF.

`RootMove`: pv + previousPV, score/previousScore/averageScore/uciScore,
bound flags, effort (node count), selDepth. `RootMoves = Vec<RootMove>`,
stable-sorted descending by (score, previousScore).

## Threading data layout

- Shared: the TT, an options map, NUMA-replicated NNUE network, and (newer)
  per-NUMA-node shared histories. Everything else per-thread.
- Per-thread `Worker` holds all histories, root state, node counters
  (relaxed atomic u64, aggregated by summation at report time).
- No intra-struct cache-line padding — isolation comes from each Worker
  being its own large-page allocation.
- Rust plan: single-threaded first; design so `Board` + search state is
  `Send`, TT is `&TT` with interior atomic mutability. LazySMP falls out.
