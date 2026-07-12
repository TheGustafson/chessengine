# Transposition table

Source: `src/transposition.rs` (primary, 417 lines, read in full), plus call
sites in `src/search.rs`, `src/thread.rs`, `src/threadpool.rs`, `src/uci.rs`,
`src/board.rs` (`hash`/`key_after`), and `src/types/score.rs` (score-band
constants). Reckless v0.10.0-dev, AGPLv3 — ideas and formulas only, no code
reproduced (D27-equivalent). Compare against `stockfish_reference/04-tt-search-structures.md`
(SF) and our `src/tt.rs` / `design/08-transposition-table.md` (us).

Headline framing: Reckless's TT is SF-lineage in almost every externally
visible number (score constants, generation width, cluster size, mul-hi
indexing, rule50 hash folding) but takes a **structurally different, more
aggressively racy** implementation path internally — it trades our/SF's
per-field atomics for whole-cluster raw-pointer aliasing under a manual
`unsafe impl Sync`, and it trades SF/our per-entry key storage for a single
packed 64-bit tag word shared by the whole cluster, decoded with a
branchless SWAR bit-trick. Both changes are genuinely clever; one of them
(the tag packing) is a good idea to study, the other (the raw-pointer
racing) is exactly the risk our design deliberately paid 2 extra bytes per
entry (10 vs Reckless's 8) to avoid.

## 1. Entry: 8 bytes, tags factored out (`transposition.rs:68-98`, size
assert `:16`)

`InternalEntry` is `{ mv: Move (2B), score: i16 (2B), raw_eval: i16 (2B),
offset_depth: u8 (1B), flags: Flags (1B) }` = 8 bytes, no padding. Contrast
with our/SF's 10-byte entry: Reckless's entry carries **no key tag at all**
— the verification tag lives one level up, in the cluster (§2). That's the
entire size difference (8 vs 10 bytes): the tag isn't dropped, it's
deduplicated across the cluster's three entries into one shared field.

- `flags: Flags` is a `u8` bitfield (`:28-57`): bits 0-1 bound
  (`None`/`Exact`/`Lower`/`Upper`, `Bound` enum `:60-66`, same four-way
  semantics as ours/SF), bit 2 `tt_pv`, bits 3-7 a 5-bit `age` (`AGE_CYCLE =
  1 << 5 = 32`, `AGE_MASK = 0x1F`, `:12-13`). Same three fields and same
  5-bit generation width as our `gen_bound8`, just packed in the opposite
  bit order (age high here, generation low in ours).
- `offset_depth: u8` stores `depth + 1` clamped to `u8` range
  (`TtDepth::to_tt`, `:95-97`); decoding is `offset_depth as i32 - 1`
  (`:91-93`). Two named sentinels: `TtDepth::NONE = 0` (external "no depth
  argument" marker, asserted via `debug_assert!(depth != TtDepth::NONE)` at
  every `write()` call, `:211`) and `TtDepth::SOME = -1` (external depth
  used for eval-only/qsearch writes that have no real search depth, e.g.
  `search.rs:465,1275,1356`). Worth flagging for a careful reader: a
  genuinely never-written slot has raw byte 0, which decodes to depth `-1`
  (`SOME`), **not** `0` (`NONE`) — so the "break immediately on an empty
  slot" fast path in the replacement scan (`candidate.depth() ==
  TtDepth::NONE`, `:230`) doesn't fire on truly blank memory; it only fires
  on a slot that was explicitly written with external depth `0`, which
  `write()`'s own assert forbids. Harmless in practice (a blank slot's
  quality score of `-1 - 4*0` is still very low and usually wins the
  min-scan anyway, see §5), just a latent dead branch, not a correctness
  bug. Cite: `transposition.rs:85-98,227-242`.

## 2. Cluster: 32 bytes, tags packed into one `u64`, SWAR lookup — the
headline distinctive idea (`transposition.rs:106-130`, size asserts `:15`)

`Cluster { entries: [InternalEntry; 3] (24B), keys: u64 (8B) }`, no
`repr(C)` (only a size assert), align implied by the surrounding allocator.
3×8 + 8 = 32 bytes — the same half-cache-line budget as our/SF's
"3 entries + 2 pad", but Reckless spends the padding bytes on function
instead of leaving them idle: `keys` packs **four** 16-bit lanes (only
three are ever written; the top lane, bits 48-63, stays permanently zero).

- `key(index)`/`set_key(index, key)` (`:113-121`) read/write one 16-bit
  lane by shifting `index*16`.
- `lookup_key(key)` (`:123-129`) finds which lane (if any) equals `key` in
  **O(1), branchless**, using the classic "find a matching/zero lane" SWAR
  trick (the same family as the glibc `strlen`/`memchr` word-at-a-time
  trick, applied to 16-bit lanes instead of bytes): replicate `key` into
  all four lanes (`needle = key * 0x0001_0001_0001_0001`), XOR against the
  packed tags so a matching lane becomes all-zero, then apply the standard
  "has a zero lane" mask (`(x - lane_ones) & !x & (lane_ones << 15)`) and
  read off the first matching lane via `trailing_zeros() / 16`. No loop, no
  branch — one multiply, two XORs/subtracts, two ANDs, a `trailing_zeros`.
  Compare to our/SF's `for entry in &cluster.entries { if entry.key16 ==
  tag { ... } }`, a 3-iteration linear scan (unrolled by the compiler, but
  still a real loop with 3 compares).
- The unused fourth lane is intentionally never `set_key`'d, so it's
  permanently `0x0000`. It can only spuriously "match" when the probed
  key's low 16 bits are literally `0`; every caller already gates on
  `index < cluster.entries.len()` (3), so a phantom fourth-lane match is
  silently treated the same as "no match" (`read()` `:187-202`, `write()`
  `:222-224`). Neat: the same branchless arithmetic serves a real 3-slot
  cluster and a would-be 4-slot layout, and the bounds check throws away
  the difference for free.

This is a genuinely cleverer packing than both SF's and our per-entry
`key16` field: same 32-byte cluster budget, but Reckless buys back 2
bytes/entry (6 bytes/cluster) by deduplicating tag storage, and buys a
branchless lookup as a side effect of how the dedup happens to be encoded.

## 3. Indexing: mul-hi64, identical to ours/SF (`transposition.rs:297-301`)

`index(hash, len) = ((hash as u128) * (len as u128)) >> 64` — the exact
same Lemire fast-alternative-to-modulo trick, same citation
(lemire.me/blog/2016/06/27), same "works for any `len`, no power-of-two
requirement" property as our `mul_hi64`. No difference to report here
beyond naming (`index()` free function vs our `TranspositionTable::cluster_for`
method) — worth noting as one more point of convergent design, not
divergence.

## 4. Concurrency model: raw pointers + `unsafe impl Sync`, not atomics —
the headline *risk* (`transposition.rs:1,133-139,178-267`)

This is the biggest structural difference from both SF-current and us.
`TranspositionTable` holds `ptr: AtomicPtr<Cluster>`, `len: AtomicUsize`,
`age: AtomicU8` — but **`Cluster`/`InternalEntry` themselves hold plain,
non-atomic fields** (`u64`, `u16`, `i16`, `u8`). `read()`/`write()`
dereference the raw pointer (`unsafe { &*self.ptr().add(index) }` /
`&mut *...`, `:181,215`) and then do ordinary, non-atomic field
reads/writes across however many search threads are live, with a bare
`unsafe impl Sync for TranspositionTable {}` (`:139`) as the only
acknowledgment that this is a data race by construction. Under the strict
Rust/C++ memory model this is a race on non-atomic memory with at least one
writer — technically UB — that "works" only because it matches how real
x86-64/ARM hardware actually behaves for naturally-aligned loads/stores
(the same bet SF itself made for over a decade before its 2023 rewrite to
`RelaxedAtomic<T>` fields, which is what our `stockfish_reference/04-*.md`
and our own `src/tt.rs` both describe and follow). Our module comment on
`TtEntry` (`src/tt.rs:150-153`) exists specifically to name this trade-off
and land on the safer side of it: every field is a real `AtomicU16/U8/I16`,
so concurrent probe/write is racy in *content* (a read may mix two
writers') but never racy in the *memory-model* sense — no UB, miri-clean.
Reckless accepts the UB in exchange for not having to route every field
through an atomic type. Given Reckless's demonstrated strength this
"benign race" bet clearly works out in practice on real hardware, but it is
not a technique to port — it's the one place this note recommends explicitly
staying with our current (safer) design.

A second consequence of the same raw-pointer design: `resize()`/`clear()`
take `&self`, not `&mut self` (`:141-157`), and `resize()` `deallocate`s the
old buffer and swaps in the new pointer/length via plain relaxed stores
while other threads may be mid-probe holding a copy of the old pointer —
there is no memory fence or thread-join between the old buffer's
`munmap`/`dealloc` and a concurrent reader's dereference of a pointer it
loaded a moment earlier. In practice this is presumably safe only because
UCI protocol discipline keeps `setoption Hash`/`Clear Hash` from arriving
mid-search — nothing in the type system enforces it. Compare to our
design: `resize`/`clear` take `&mut self`, obtained via `Arc::get_mut` only
once every worker thread has actually been joined/parked (`design/08
§3.13-3.14`, `design/13 §3.6`) — the exclusivity is load-bearing and
compiler-checked, not merely protocol-assumed. This is the second place
this note recommends keeping our current approach rather than Reckless's.

## 5. Generations and replacement (`transposition.rs:100-104,219-244`)

- `relative_age(tt_age) = (AGE_CYCLE + tt_age - entry.age) & AGE_MASK`
  (`:101-104`) — the same 32-tick modular clock as SF/ours
  (`(32 + cur - ent) & 31`), same 5-bit width, same "increment once per
  root search" cadence (`TranspositionTable::increment_age`, called from
  `threadpool.rs:86` inside `execute_searches`, i.e. once per `go` — exactly
  our `new_search()` call site convention).
- Victim selection (inside `write()`, not a separate probe step — see §7):
  `quality = candidate.depth() - 4 * candidate.relative_age(tt_age)`, first
  strict min wins (`:235-239`). **The age weight is 4, not 8.** SF and our
  own port both use "one generation costs 8 plies of depth"; Reckless
  halves that to 4, i.e. it is *more willing to keep an old, deep entry
  over a newer shallow one* than SF/our tuning. This is a concrete,
  cheaply-testable tuning knob (§ Adapt to us).
- Tie-break on equal quality keeps the earlier-scanned entry (strict `<`,
  not `<=`, `:236`) — same convention as ours.

## 6. Write / save policy (`transposition.rs:205-267`)

Reckless recomputes the cluster index and re-runs `lookup_key` from
scratch inside `write()` (`:213-224`), independently of whatever `read()`
did earlier in the same node — there is no persistent "probe decision"
object carried from probe-time to write-time the way our `Probe`/`TtWriter`
is (`src/tt.rs:353-400`; our module doc explicitly flags that a held
`TtWriter` "goes stale after deeper re-searches at the same node re-probe",
`src/tt.rs:355-356`). Reckless's two-call, fully-decoupled `read()`/`write()`
API sidesteps that staleness question entirely at the cost of doing the
cluster lookup twice per node that both probes and stores.

Rule-by-rule, in call order:

1. **Move preservation** runs unconditionally first, before any
   accept/reject decision: keep the old `mv` unless the new write supplies
   a real move or the key differs (`:249-251`) — same rule, same ordering,
   as our rule 1 (`src/tt.rs:239-244`).
2. **Reject-and-return test** (`:253-255`): skip the rest of the write
   (depth/value/bound/eval/key untouched) when **all** of: not `force`,
   same key, `depth + 4 + 2*is_pv <= old_depth`, and same generation.
   Algebraically this is our accept condition `new_d + 2*pv > old_d - 4`
   negated and rearranged — **the constants (margin 4, PV credit 2) are
   identical to SF/ours**, just expressed as Reckless's reject
   short-circuit vs our accept condition. Good independent confirmation
   that this specific tuning is converged/industry-standard, not a
   Reckless-only choice.
3. **No explicit `Bound::Exact` carve-out.** SF and our port special-case
   "the new bound is Exact" as an unconditional-accept branch alongside
   "different key" and "older generation" (`src/tt.rs:256-260`). Reckless
   has no such branch in the formula; instead it exposes a `force: bool`
   parameter on `write()` that is threaded through and set to `NODE::PV`
   (a compile-time const-generic bool) at exactly one call site — the main
   negamax exit (`search.rs:1161`). Every other write call site passes
   `force = false` (TB store `:432`, eval-only cache `:465`, probcut
   `:677`, a fail-high-quiet-history write `:1040`, both qsearch stores
   `:1275,1356`). Net effect: **any** PV-node write at the main negamax
   exit force-overwrites its slot regardless of stored depth/age/bound —
   including a PV node that failed low and is storing only an `Upper`
   bound. That is strictly broader than SF/our "only `Exact` forces
   through" rule: Reckless trusts "this was a PV node" as the trigger,
   we/SF trust "this result is exact" specifically.
4. **No secondary-aging / stale-decisive-score erosion.** Our `save()` has
   a third branch (`src/tt.rs:273-288`, "Rule 3"): a *rejected* write
   against a deep, non-Exact, decisive (mate/TB-band) entry erodes its
   stored depth by one ply per rejection, so a stale mate claim ages out
   over repeated probes instead of squatting until a generation flip. SF
   has the equivalent (`search.cpp` mate-depth decrement noted in
   `stockfish_reference/04-tt-search-structures.md`). Reckless's `write()`
   has exactly two outcomes — full overwrite or full no-op — with nothing
   in between. This is a technique we (and SF) have that Reckless's TT
   layer does not.
5. **Mate/decisive-score ply adjustment is inlined**, not a named helper:
   `if is_decisive(score) && is_valid(score) { score += score.signum() *
   ply }` sits directly in `write()` (`:257-260`), whereas the inverse
   (read-side) conversion is its own function, `score_from_tt` (§7). Our
   port names both directions (`value_to_tt`/`value_from_tt`,
   `design/08 §3.12`); this is a minor style asymmetry in Reckless, not a
   behavioral difference — the write side is a one-line transform in both
   engines, it just didn't earn a name in Reckless's source.

## 7. Read and mate-score conversion (`transposition.rs:178-203,308-345`)

`read(hash, halfmove_clock, ply) -> Option<Entry>` does the cluster lookup,
and on a tag match (`index < 3`) **always** returns `Some(Entry{...})` —
there is no separate "tag matched but slot unoccupied" signal analogous to
our `Probe.hit` (our `probe()` returns `hit: bool` alongside the decoded
data specifically so a same-tag-but-empty slot reads as a miss, tested
explicitly by `zero_tag_matches_empty_slot_and_writes_back_into_it`,
`src/tt.rs:718-731`). Combined with the §1 observation (a truly blank slot
decodes to `depth = SOME(-1)`, `bound = Bound::None`, `score = 0`), this
means: in the rare case a position's 16-bit tag equals `0x0000` **and**
lands on a cluster slot that has literally never been written, `read()`
reports a "hit" with `bound = Bound::None`, and both of Reckless's TT-cutoff
match expressions use a wildcard arm for `Bound::None` that evaluates to
`true` (`search.rs:392,1231`) gated only by `is_valid(tt_score)` — and
`0 != Score::NONE (32002)`, so `is_valid(0)` is `true`. In `search()` the
accompanying depth guard (`tt_depth > depth - ...`, with `tt_depth = -1`)
almost certainly prevents a real cutoff at realistic depths, but `qsearch`'s
early-cutoff check (`search.rs:1225-1234`) has no depth term at all, so this
path *could* return a spurious score-0 cutoff for a position that was never
actually analyzed. This is a roughly 1-in-65536-per-slot edge case, plausibly
never triggered in practice (or triggered so rarely it's invisible against
search noise) — noted here as a "double-check if re-deriving this" curiosity,
not a confirmed live bug; the point worth keeping is that our engine's
explicit `hit` flag plus its dedicated regression test already close this
exact class of edge case, which Reckless's leaner `Option<Entry>`-only API
does not structurally rule out.

`score_from_tt` (`:308-345`) is the read-side mate/TB downgrade, matching
SF's `value_from_tt` shape exactly: convert root-relative back to
node-relative (`score - ply` / `score + ply`), and additionally downgrade a
mate-band or TB-band score to `TB_WIN_IN_MAX ± 1` whenever the claimed mate
distance could be invalidated by the 50-move rule (`MATE - score > 100 -
halfmove_clock`, and the mirrored TB-band check). Same two-band structure
(`MATE_IN_MAX`/`TB_WIN`/`TB_WIN_IN_MAX`, `types/score.rs:10-17`) and same
"100 - rule50" margin constant as SF and our own `value_from_tt`.

Score constants themselves are essentially identical across all three
engines — `Score::MATE = 32000`, `INFINITE = 32001`, `NONE = 32002`
(`types/score.rs:8-12`) match our `VALUE_MATE/VALUE_INFINITE/VALUE_NONE`
bit-for-bit; `MAX_PLY` differs slightly (Reckless 240 vs our 246,
`types/mod.rs:26`). Read as confirmation these are converged, SF-lineage
numbers rather than anything to "adapt" — nothing to change here, just
worth knowing the family resemblance runs deep even though the entry/cluster
internals diverge.

## 8. Allocation, resize, clear (`transposition.rs:347-417`)

`allocate(threads, size_mb)` (`:364-386`): on Linux, a plain anonymous
`mmap(null, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)`
followed by `madvise(ptr, size, MADV_HUGEPAGE)`; on any other target,
`std::alloc::alloc_zeroed` with `Layout::from_size_align(size,
align_of::<Cluster>())`. Two things worth noting relative to our
`design/08 §3.13` "Phase 2" plan:

- **Simpler than our plan assumes.** Our doc's Phase 2 sketch reaches for
  a custom `Layout` at 2 MiB alignment plus a hand-rolled `Drop`/`Deref`
  wrapper specifically because `std::alloc::alloc_zeroed` needs the exact
  same `Layout` back at `dealloc` time (a real hazard: mismatched
  size/align on dealloc is UB). `mmap`/`munmap` don't have that hazard —
  `munmap` only needs `(ptr, len)`, no alignment bookkeeping, and the
  kernel already hands back zeroed, natively-page-aligned memory. Reckless
  doesn't even request 2 MiB alignment explicitly; it just advises
  `MADV_HUGEPAGE` on a plain anonymous mapping and lets the kernel promote
  what it can. This is a meaningfully simpler shape than our doc's option
  (a)/(b) split and worth adopting nearly as-is when Phase 2 lands.
- **No failure handling at all** — a real gap relative to our design.
  `mmap`'s return value is never checked against `MAP_FAILED`, and
  `alloc_zeroed`'s result is never checked for null; both paths hand the
  (possibly invalid) pointer straight to `parallel_clear`, which writes
  through it unconditionally. Compare to our `resize()`
  (`src/tt.rs:522-547`, `design/08 §3.13`): `try_reserve_exact` in a loop
  that halves the request on failure until the 1 MiB floor, guaranteeing a
  *usable* table after any mid-session `setoption Hash`, with only the
  initial startup allocation permitted to be fatal. Reckless's UCI `Hash`
  option caps at `262144` MB (256 GiB, `uci.rs:154`) vs our SF-matching
  32 TiB cap, which reduces exposure but doesn't remove it — an
  allocation that fails or an overcommitted mapping that gets OOM-killed
  on first touch has no graceful path in Reckless's code as read here.
  This is a technique we already have and should keep; nothing to port.

`clear()`/the shared `parallel_clear` helper (`:401-417`) split the
cluster slice into `threads` chunks and zero each on its own scoped
thread — this *is* the multithreaded, first-touch-NUMA-friendly clear our
`design/08 §3.14` explicitly defers ("not needed for the single-threaded
milestone... split with `chunks_mut` across scoped threads... first-touch
NUMA placement for free" — our doc's own words for a plan that Reckless
has simply already built). Good, concrete reference implementation for
when we land it.

## 9. Hashfull (`transposition.rs:159-172`)

Samples the first 1000 clusters (a bare literal `.take(1000)`,
`transposition.rs:165` — matching our named `HASHFULL_SAMPLE_CLUSTERS`),
counts entries where `bound != None && age == current_age`, divides by
`ENTRIES_PER_CLUSTER` (3). Slightly less flexible than
ours: Reckless hardcodes "exactly this generation" where our `hashfull(max_age)`
takes a threshold (`relative_age <= max_age`) so a caller could ask "how full
counting the last N generations" — a small, already-shipped generalization
on our side; nothing to change.

## 10. Prefetch (`transposition.rs:269-282`, call sites `search.rs:596,
1425,1438`, `board.rs:86-106`)

Same intrinsic (`_mm_prefetch::<_MM_HINT_T0>`), same no-op-off-x86_64
fallback, same double-prefetch pattern as SF and us: once with a *predicted*
key before the move is made (`board.key_after(mv)`, computed from the
from/to/captured-piece XORs plus the rule50-bucket fold, deliberately wrong
for castling's rook leg, en passant's removed pawn, and promotion's piece-type
change — "wrong but harmless", exactly SF's framing), and again with the
*real* post-move key once `make_move` has actually updated the board
(`search.rs:1425,1438`). A third prefetch fires right after `make_null_move`
in the NMP branch (`search.rs:596`). No divergence from SF/us here beyond
call-site placement.

## 11. GHI mitigation folded into every hash, not just high-rule50 ones
(`board.rs:75-84`)

`Board::hash()` unconditionally XORs a rule50-bucket key into the raw
Zobrist key: `bucket = clamp((rule50.saturating_sub(8)) / 8, 0..=15)`
(`board.rs:75-77`), i.e. buckets step every 8 halfmove-clock ticks starting
at clock 16, folded into *every* position's hash from clock 0 onward (bucket
0 covers clock 0-15). SF (and our port, `design/04 §"Rule50 folding"`,
`stockfish_reference/stockfish_search_notes.md:715-719`) instead only starts
folding once `rule50 >= 14`, leaving low-clock positions on the raw
unfolded key. Both approaches solve the same Graph-History-Interaction
problem (a position revisited at very different 50-move-clock values must
not alias in the TT); Reckless's version is very slightly more conservative
(it starts bucketing sooner) at the cost of a marginally higher chance of
splitting truly-identical near-game-start positions into different TT
slots. Not a difference worth changing anything over — both are already
converged on the same idea we already implement; noted for completeness
since it's directly adjacent to TT correctness.

## 12. Threading placement — parity, not a difference (`thread.rs:156-176`,
`threadpool.rs:86`)

`SharedContext.tt` is a single, non-NUMA-replicated `TranspositionTable`
shared by every worker via `Arc`, exactly like SF and our own plan
(`stockfish_reference/04-tt-search-structures.md`: "Shared: the TT... —
history/other tables get NUMA replication, the TT doesn't, because probes
are inherently random and replication would multiply the memory cost for
no locality win). `threadpool.rs:86` bumps the age once per `go`
(`execute_searches`), matching our `new_search()` call convention.

## Comparison summary

| Aspect | Stockfish | Us | Reckless |
|---|---|---|---|
| Entry size | 10 B (SF-current, atomics) | 10 B (`src/tt.rs`, D20) | 8 B (tag factored out to cluster) |
| Cluster size | 32 B (3 entries + 2 pad) | 32 B (3 entries + 2 pad) | 32 B (3×8B entries + 8B packed tags) |
| Tag storage | per-entry `key16` | per-entry `key16` | per-cluster packed `u64`, SWAR lookup |
| Indexing | mul-hi64 | mul-hi64 | mul-hi64 (identical) |
| Field atomicity | per-field relaxed atomics | per-field relaxed atomics | **none** — raw non-atomic fields, `unsafe impl Sync` |
| resize/clear exclusivity | (thread-pool coordinated) | `&mut self` via `Arc::get_mut` | `&self`, no compiler-enforced exclusivity |
| Generation width | 5 bits, mod 32 | 5 bits, mod 32 | 5 bits, mod 32 (identical) |
| Replacement weight | depth − 8·age | depth − 8·age | depth − **4**·age |
| Exact/force overwrite | bound==Exact special-cased | bound==Exact special-cased | `force` flag = "this is a PV node" (broader) |
| Secondary aging (stale mate erosion) | yes | yes (`penalize`/Rule 3) | **no** |
| Huge pages | 2 MiB-aligned alloc + madvise + 1 GiB hugetlb | planned (§3.13 Phase 2) | mmap + madvise only, **no failure handling** |
| Multithreaded clear | yes | planned (§3.14) | yes, already shipped |
| GHI hash folding | rule50 ≥ 14 threshold | rule50 ≥ 14 threshold (matches SF) | folds from clock 0 (bucket steps every 8, starting at 16) |

## Adapt to us

- **Branchless per-cluster tag lookup (SWAR).** Pack the cluster's 3
  `key16` tags into one `u64` (as a companion field, or by shrinking each
  entry to match Reckless's 8-byte shape) and replace the linear
  `for entry in &cluster.entries` scan in `TranspositionTable::probe`
  (`src/tt.rs:472-484`) with the replicate-XOR-mask-trailing_zeros trick.
  Touches: `design/08 §3.2-3.5`, `src/tt.rs` entry/cluster layout. This
  interacts with our atomics: the tags would need to be read from 3
  separate `AtomicU16::load`s and reassembled into a `u64` locally (or
  stored as one `AtomicU64` covering all 3 tags, sacrificing independent
  per-entry tag writes — needs a design decision on which of the two you
  want, since our per-field-atomics contract is the thing we're not
  abandoning per this note's §4 recommendation). Effort: **M** (touches
  the entry layout, all the layout-size const-asserts, and the roundtrip
  tests in `src/tt.rs`'s `#[cfg(test)]` module). Impact: probe is already
  cheap (one cache line, ~3 compares the compiler likely unrolls/vectorizes
  already); expect a small constant-factor win on the hot probe path,
  worth a bench-nps measurement before committing, not an Elo-bearing
  change by itself. Best paired with whatever P9/P10 hardening pass next
  touches `src/tt.rs`.

- **Try halving the replacement age weight (8 → 4).** One-line change to
  `REPLACEMENT_AGE_WEIGHT` (`src/tt.rs:69`) mirroring Reckless's tuning.
  Touches: `design/08 §3.6`. Effort: **XS** (a constant). Impact:
  unknown sign — this is exactly the kind of single-constant tuning
  question the fixed-node proxy SPRT (`design/14 §9.4`) exists for; run it
  there before spending a real-TC SPRT slot. Cheap experiment, don't skip
  it just because Reckless's number differs from SF's — top engines
  disagreeing on a tuned constant is itself a useful hint that this knob
  is worth a real measurement here rather than assuming SF's 8 is settled.

- **Consider a broader "trust this node" force-overwrite,** as an
  alternative or a supplement to the current Exact-only carve-out in
  `TtEntry::save` (`src/tt.rs:256-260`): force through any `is_pv` write at
  the negamax exit, not just `Exact` ones, matching Reckless's
  `force = NODE::PV` pattern (`search.rs:1161`). Touches: `design/08 §3.7`,
  `design/07` (the search-side call site). Effort: **S** (one extra bool
  threaded through `TtWriter::write`'s call sites, `src/tt.rs:368-381`).
  Impact: speculative — broader forced overwrites could either help
  (fresher PV information propagates faster) or hurt (a fail-low PV write
  evicting a genuinely deeper Lower/Upper result from a sibling line more
  often). Gate behind the fixed-node proxy SPRT before a real-TC one.

- **When Phase 2 huge pages land, prefer the `mmap`/`madvise` shape over
  the `alloc_zeroed`-with-custom-`Layout` shape** sketched in
  `design/08 §3.13`. Reckless's version (`transposition.rs:364-386`)
  sidesteps the whole "must `dealloc` with the identical `Layout`" hazard
  because `munmap` only needs `(ptr, len)`; it's less unsafe-surface than
  either of the doc's two proposed options and reads as a validated,
  already-shipped reference for exactly the platform-gated `libc`-dependency
  shape the doc's option (a) already leans toward. Touches: `design/08
  §3.13 Phase 2`, roadmap phase P9/P10 polish or later (explicitly deferred,
  no urgency). Effort: **M** (still needs our own fallible-resize
  wrapper around it — see next item — plus the `#[cfg(target_os =
  "linux")]` gating and decisions.md amendment the doc already flags as a
  prerequisite). Impact: real but deferred; TT probes are random misses,
  and huge pages measurably cut TLB-miss overhead on large tables — SF's
  own notes call this "measurable Elo" (`stockfish_reference/04-tt-search-structures.md`
  "Large pages").

- **Do not adopt Reckless's unchecked allocation path** — keep our
  fallible, halving-fallback `resize` (`src/tt.rs:522-547`) wrapped
  *around* whatever `mmap`-based allocator Phase 2 introduces, i.e. check
  `mmap`'s return against `MAP_FAILED` / `alloc_zeroed`'s against null and
  feed failure into the same halving loop rather than porting Reckless's
  unconditional-success assumption verbatim. This is a "keep what we have"
  item, not a "port this" item — flagged so whoever implements Phase 2
  doesn't accidentally copy the gap along with the mmap technique.

- **Do not adopt the raw-pointer/non-atomic `Cluster` design or the
  `&self`-based resize/clear.** Both are called out in §4 as the one place
  Reckless's design is *riskier* than ours, not cleverer — our per-field
  atomics (D14) and `&mut self`-gated resize/clear (`design/13 §3.6`) are
  the more defensible choice and this note recommends explicitly keeping
  them. No action item beyond "don't regress this under P11 lazy-SMP
  pressure to 'simplify' the TT" — worth a one-line callout in `design/08`
  or `design/11` (whichever owns the eventual lazy-SMP TT-sharing writeup)
  the next time either doc is touched, so a future contributor doesn't
  independently rediscover Reckless's shape and assume it's the more
  battle-tested option merely because Reckless is stronger overall.

- **No action needed, but worth a shared footnote:** our `hashfull(max_age)`
  is already a strict generalization of Reckless's hardcoded
  current-generation-only count (§9), and our `Probe.hit` boolean plus the
  `zero_tag_matches_empty_slot_and_writes_back_into_it` test (`src/tt.rs:718-731`)
  already close the class of "tag-matched-but-blank-slot" edge case this
  note's §7 flags as a possible (rare, unconfirmed) Reckless edge case.
  Nothing to change; useful to know our design already tested for the thing
  their leaner API doesn't structurally rule out.
