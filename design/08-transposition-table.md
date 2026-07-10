# 08 — Transposition table

Status: Draft v1

The transposition table (TT) is the engine's shared, lock-free cache of search
results, keyed by the 64-bit Zobrist key. This document specifies the full TT
design per (D19): a 10-byte entry of relaxed atomics, 3-entry 32-byte
clusters, multiply-high indexing, 5-bit generation aging, and Stockfish's
current replacement policy — plus the eval-cache role, the `Hash` size
limits (1..=33554432 MiB), the allocation strategy (portable `Vec` with a
fallible, never-aborting `resize` first, Linux huge pages later), the
prefetch API, and the validation contract that makes deliberately racy
reads safe to consume. The TT
is the only large shared-mutable structure in the engine (D23); everything
here is safe Rust except the prefetch intrinsic (D14a).

## 1. Purpose and scope

This document owns:

- The `tt` module: entry/cluster layout, probe, replacement/aging policy,
  `save` semantics, generation cycling, `hashfull`, resize/clear, allocation,
  and the prefetch wrapper.
- The consumer-side *validation contract* (what a reader must check before
  trusting TT data), stated here because it is part of the TT's API.

This document does NOT own:

- The raw move accessors `Move::raw16()` / `Move::from_raw16()` — owned by
  the core-types doc (design/02-core-types.md §3.7, `pub const fn`); the TT
  consumes them for `move16` packing, it does not define them.

- Mate/TB score ply-conversion (`value_to_tt` / `value_from_tt`) and every
  probe/write call site, TT cutoff conditions, and the `penalize` trigger —
  those live in the search (design/07-search.md). Section 3.12 restates the
  boundary contract only.
- The Zobrist keys themselves — key construction, ep-only-when-legal keying,
  and rule50 folding (D9) belong to the board/Zobrist doc. The TT treats the
  key as an opaque, well-distributed `u64`.
- The `Hash` / `Clear Hash` UCI options and `info hashfull` output (UCI doc).
- Thread pool mechanics for the future multithreaded clear (threads doc, D23).

## 2. References

- `stockfish_reference/04-tt-search-structures.md` — distilled TT notes
  (entry layout, cluster, replacement, indexing, large pages).
- `stockfish_reference/stockfish_search_notes.md` §11 (TT), §12 (Zobrist
  keys/prefetch interplay).
- Stockfish source (verified for this doc, read-only, D27 — ideas and
  constants only, no code copied):
  - `stockfish_reference/Stockfish/src/tt.h`, `tt.cpp` — entry fields, bit
    packing, save/replacement rules, probe, hashfull, generation cycling.
  - `stockfish_reference/Stockfish/src/types.h` — value/depth constants and
    predicates (`is_valid`, `is_win`, `is_loss`, `is_decisive`, `is_mate`).
  - `stockfish_reference/Stockfish/src/search.cpp` — call-site shapes:
    eval-cache writes at `DEPTH_UNSEARCHED`, `penalize(1)` trigger,
    `value_to_tt`/`value_from_tt`, speculative prefetch.
  - `stockfish_reference/Stockfish/src/memory.cpp` — 2 MB-aligned allocation
    + `madvise(MADV_HUGEPAGE)` huge-page strategy.

## 3. Design

### 3.1 Module, constants, and packed encodings

New module `src/tt.rs` (D13), re-exported from `lib.rs`. It depends on
`types` (for `Value`, `Depth`, and the D5/D6 constants) and `moves::Move`.

Value/depth constants used here, per (D5)/(D6), mirroring SF-current
`types.h` exactly (D20):

```text
VALUE_MATE                = 32000
VALUE_INFINITE            = 32001
VALUE_NONE                = 32002
MAX_PLY                   = 246
VALUE_MATE_IN_MAX_PLY     = VALUE_MATE - MAX_PLY            = 31754
VALUE_TB                  = VALUE_MATE_IN_MAX_PLY - 1       = 31753
VALUE_TB_WIN_IN_MAX_PLY   = VALUE_TB - MAX_PLY              = 31507
VALUE_TB_LOSS_IN_MAX_PLY  = -VALUE_TB_WIN_IN_MAX_PLY        = -31507

DEPTH_QS          = 0
DEPTH_UNSEARCHED  = -2      (eval-only entries, §3.11)
DEPTH_NONE        = -3      (absent / empty)
```

Predicates (free functions in `types`, D5): `is_valid(v) = v != VALUE_NONE`,
`is_win(v) = v >= VALUE_TB_WIN_IN_MAX_PLY`, `is_loss(v) = v <=
VALUE_TB_LOSS_IN_MAX_PLY`, `is_decisive(v) = is_win(v) || is_loss(v)`,
`is_mate(v) = v >= VALUE_MATE_IN_MAX_PLY`.

TT-local packing constants (SF-current, D20):

```rust
pub const CLUSTER_SIZE: usize = 3;

const GENERATION_BITS: u8 = 5;
const GENERATION_MASK: u8 = (1 << GENERATION_BITS) - 1; // 0x1F, cycle of 32
const BOUND_SHIFT: u8     = GENERATION_BITS;            // 5
const PV_SHIFT: u8        = BOUND_SHIFT + 2;            // 7
const PV_MASK: u8         = 1 << PV_SHIFT;              // 0x80
```

So `gen_bound8 = generation | (bound << 5) | (pv << 7)`: 5 generation bits,
2 bound bits, 1 PV bit.

The bound type is a real enum (idiomatic Rust) whose discriminants keep SF's
bit-flag semantics — `Exact` is literally `Upper | Lower`, and the search
tests bound *inclusion* rather than equality:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    None  = 0,
    Upper = 1, // fail-low result: true value <= stored value
    Lower = 2, // fail-high result: true value >= stored value
    Exact = 3, // == Upper | Lower
}

impl Bound {
    pub const fn includes_lower(self) -> bool { self as u8 & Bound::Lower as u8 != 0 }
    pub const fn includes_upper(self) -> bool { self as u8 & Bound::Upper as u8 != 0 }
    /// Decode 2 bits; total over all inputs (masks to the low 2 bits).
    const fn from_bits(bits: u8) -> Bound { /* match bits & 0b11 { 0..=3 } */ }
}
```

Raw move bits: the TT stores moves as raw `u16` via `Move::raw16()` /
`Move::from_raw16()` — the `pub const fn` accessor pair owned by the
core-types doc (design/02-core-types.md §3.7); this doc consumes it, it does
not introduce its own. Every 16-bit pattern decodes to a *structurally*
valid `Move` (from/to are always squares 0..63, kind and promotion are 2-bit
fields), so the conversion is total and safe; *semantic* validity is the
consumer's job (§3.5).

### 3.2 TTEntry: 10 bytes of relaxed atomics

Field order matches probe's read order and save's store order (memory is
fastest sequentially — SF orders it this way deliberately):

```rust
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicI16, Ordering::Relaxed};

#[repr(C)]
struct TtEntry {
    key16:      AtomicU16, // low 16 bits of the Zobrist key = in-cluster tag
    depth8:     AtomicU8,  // stored as (depth - DEPTH_NONE), so 0 == empty
    gen_bound8: AtomicU8,  // generation | bound << 5 | pv << 7
    move16:     AtomicU16, // best move, raw Move bits; 0 (Move::NULL) = none
    value16:    AtomicI16, // search value (already ply-converted, §3.12)
    eval16:     AtomicI16, // RAW static eval, uncorrected (§3.11, D22)
}

const _: () = assert!(size_of::<TtEntry>() == 10);
const _: () = assert!(align_of::<TtEntry>() == 2);
```

Key points:

- **Only 16 tag bits are stored.** The high key bits are consumed by the
  cluster index (§3.4); the low 16 bits disambiguate within the cluster.
  Genuine tag collisions (~1 per 2^16 tag comparisons) are accepted; the
  consumer contract (§3.5) makes them harmless-ish. Larger TT ⇒ rarer harm.
- **`depth8` doubles as the occupancy flag.** Stored form is
  `depth - DEPTH_NONE`, so the all-zero (freshly cleared) entry decodes to
  `DEPTH_NONE` and `is_occupied() == (depth8 != 0)`. Valid written depths
  satisfy `DEPTH_NONE < d` and `d - DEPTH_NONE < 256` (debug-asserted; with
  `MAX_PLY = 246` this can't trigger).
- **Every access is `Ordering::Relaxed`.** Individual field loads/stores
  cannot tear (they are atomic), so there is no UB — this is the sanctioned
  safe-Rust equivalent of SF's `RelaxedAtomic<T>` fields (D14: TT atomics are
  safe Rust; D19). There is no entry-wide atomicity: no lock, no seqlock, no
  key-XOR-data checksum (modern SF removed it). A concurrently written entry
  can be read *self-inconsistently* — see §3.5 for why that is accepted.
- Entry methods (all private to the module): `read() -> TtData`,
  `is_occupied() -> bool`, `relative_age(gen: u8) -> u8`, `save(...)`.

`relative_age` — 5-bit clock arithmetic. Generations wrap like a 32-hour
clock, and the wrapping u8 subtraction makes the bound/PV bits of
`gen_bound8` cancel out under the mask (any multiple of 32 vanishes mod 32):

```rust
fn relative_age(&self, generation: u8) -> u8 {
    generation.wrapping_sub(self.gen_bound8.load(Relaxed)) & GENERATION_MASK
}
```

### 3.3 Cluster: 3 entries + 2 pad bytes = 32 bytes

```rust
#[repr(C, align(32))]
struct Cluster {
    entries: [TtEntry; CLUSTER_SIZE], // 30 bytes
    _pad: [u8; 2],                    // 32 total
}

const _: () = assert!(size_of::<Cluster>() == 32);

impl Cluster {
    const fn zeroed() -> Self { /* all atomics new(0) */ }
}
```

32 bytes is exactly half a cache line: a probe touches one line, and the
`align(32)` guarantees a cluster never straddles a line boundary (a 32-byte
object at a 32-byte boundary fits entirely inside one 64-byte line). Rust's
`Vec<Cluster>` allocates with the element type's layout, so the alignment
holds on the portable path too.

### 3.4 Table type and indexing: multiply-high, not modulo

```rust
pub type Key = u64; // alias owned by the zobrist module; shown here for clarity

pub struct TranspositionTable {
    clusters: Vec<Cluster>, // later: huge-page-backed buffer (§3.13)
    generation8: AtomicU8,  // current age, 0..32; interior-mutable (D23)
}

#[inline]
fn mul_hi64(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

impl TranspositionTable {
    #[inline]
    fn cluster_for(&self, key: Key) -> &Cluster {
        let idx = mul_hi64(key, self.clusters.len() as u64) as usize;
        &self.clusters[idx]
    }
}
```

`mul_hi64(key, n)` maps a uniform 64-bit key onto `[0, n)` for **any** n —
no power-of-two restriction, no modulo. Proof of bounds: the product is at
most `(2^64 − 1)·n`, so its high 64 bits are strictly less than `n`; the
index is always in range, and the bounds check on the slice index is
statically satisfiable (keep the safe index until a profile says otherwise,
D14c). One `mul` (x86 `mulx`) versus a ~25-cycle division. Consequence: any
`Hash` size in MB is usable directly — `cluster_count = MB · 1024 · 1024 /
32` with no rounding to powers of two, so the user gets all the memory they
asked for.

### 3.5 Probe: tag scan, decoded copy, and the validation contract

Public read API. `TtData` is a plain decoded copy; `TtWriter` is a thin
handle to the entry the caller should write back to.

```rust
#[derive(Debug, Clone, Copy)]
pub struct TtData {
    pub mv: Move,      // Move::NULL when absent
    pub value: Value,  // node-relative (§3.12); meaningful only per contract below
    pub eval: Value,   // RAW static eval or VALUE_NONE
    pub depth: Depth,  // DEPTH_NONE when absent
    pub bound: Bound,
    pub is_pv: bool,
}

impl TtData {
    pub const EMPTY: TtData = TtData {
        mv: Move::NULL, value: VALUE_NONE, eval: VALUE_NONE,
        depth: DEPTH_NONE, bound: Bound::None, is_pv: false,
    };
}

pub struct TtWriter<'tt> {
    entry: &'tt TtEntry,
}

pub struct Probe<'tt> {
    pub hit: bool,
    pub data: TtData,
    pub writer: TtWriter<'tt>,
}

impl TranspositionTable {
    pub fn probe(&self, key: Key) -> Probe<'_>;
}
```

Algorithm (mirrors SF exactly):

1. `cluster = cluster_for(key)`; `tag = key as u16`.
2. Scan the 3 entries in order. On the **first** entry whose
   `key16.load(Relaxed) == tag`: return
   `Probe { hit: entry.is_occupied(), data: entry.read(), writer }`.
   Returning on the first tag match even when the entry is *unoccupied*
   (depth8 == 0, e.g. after `penalize`, or a zeroed entry whose tag happens
   to be 0) matters: the same position writes back into its own slot instead
   of evicting a neighbor. In that case `hit == false` and the decoded data
   must be ignored (it decodes to depth `DEPTH_NONE`, value 0 — note NOT
   `VALUE_NONE`; the search normalizes: `tt_value = if hit {
   value_from_tt(data.value, ply, rule50) } else { VALUE_NONE }`, doc 07).
3. No tag match: select the replacement victim (§3.6) and return
   `Probe { hit: false, data: TtData::EMPTY, writer: victim }`.

`TtEntry::read` decodes each field with one relaxed load:
`depth = depth8 + DEPTH_NONE`, `bound = Bound::from_bits(gen_bound8 >>
BOUND_SHIFT)`, `is_pv = gen_bound8 & PV_MASK != 0`, `mv =
Move::from_raw16(move16)`, `value`/`eval` sign-extended from i16.

**Why a possibly self-inconsistent copy is accepted.** Between the tag
comparison and the last field load, another thread may overwrite the entry
(same or different position). Each field is individually untorn, but the
six fields together may mix two writers: a move from position A with a value
from position B, or a value written for a different bound. Synchronizing
this (lock, seqlock, checksum) costs time on every probe — measurably more
Elo than the rare garbage entry costs (SF's explicit trade-off, D19). The
race window is a handful of nanoseconds; corruption is rare and *bounded*
because every consumer validates what it reads:

- **ttMove**: before any use (move ordering seed, TT-move extension, actually
  playing it in singular verification) the search runs
  `board.pseudo_legal(mv) && board.legal(mv)` — full semantic validation
  against the *current* position. A garbage or collided move is simply
  dropped. (Those two predicates are owned by the movegen doc.)
- **value / eval**: checked with `is_valid(v)` (`!= VALUE_NONE`) and used
  only under the bound logic (`bound.includes_lower()` for fail-high use,
  `includes_upper()` for fail-low use); mate-band values additionally pass
  through `value_from_tt`'s plausibility downgrades (§3.12).
- **depth**: only compared (`data.depth >= depth`), never used to index.
- Internal arithmetic on `depth8` saturates at 0 (§3.7, §3.8) so racy
  read-modify-writes can't wrap negative.

The worst surviving failure mode is a wrong-but-plausible score from a tag
collision — a rare mis-ordered node or bizarre blunder, vanishing as the
table grows. This is the same defect budget Stockfish accepts.

### 3.6 Replacement: victim minimizes `depth8 − 8·relative_age`

On a probe miss, choose the entry to sacrifice:

```text
victim = argmin over the 3 entries of
    depth8(e) as i32  -  8 * relative_age(e, generation) as i32
```

scanning in order and keeping the first strict minimum (ties keep the
earlier entry). The weight **8** (SF-current, D20) means one generation of
staleness outweighs 8 plies of stored depth; an empty entry (`depth8 == 0`,
age 0 after clear) scores 0 and loses to any current-generation occupied
entry, so empties are consumed first. Arithmetic in `i32` (range is
`0..=255` minus `0..=248`).

### 3.7 save(): the write policy, rule by rule

`TtWriter::write(key, value, is_pv, bound, depth, mv, eval, generation)`
forwards to `TtEntry::save`. All loads/stores relaxed; each field is loaded
at most once into a local (a benign simplification of SF's repeated loads —
races between the two reads of `key16` are tolerated either way).

```rust
impl TtEntry {
    fn save(&self, key: Key, v: Value, pv: bool, b: Bound,
            d: Depth, m: Move, ev: Value, generation: u8) {
        let tag = key as u16;
        let same_key = tag == self.key16.load(Relaxed);

        // Rule 1 — ttMove preservation: keep the old move only when the
        // new write is for the same position and brings no move.
        if m != Move::NULL || !same_key {
            self.move16.store(m.raw16(), Relaxed);
        }

        let new_d = d - DEPTH_NONE;                       // internal form
        let old_d = self.depth8.load(Relaxed) as i32;     // internal form

        // Rule 2 — overwrite test, cheapest checks first. Write if ANY of:
        //   (a) the new bound is EXACT (PV results always win),
        //   (b) the slot holds a different position,
        //   (c) refresh rule: new_d + 2*pv > old_d - 4
        //       (same position may re-write with up to ~4 less depth;
        //        PV nodes get a +2 credit),
        //   (d) the stored entry is from an older search (relative_age != 0).
        if b == Bound::Exact || !same_key
            || new_d + 2 * pv as i32 > old_d - 4
            || self.relative_age(generation) != 0
        {
            debug_assert!(d > DEPTH_NONE && new_d < 256);
            debug_assert!(generation <= GENERATION_MASK);
            self.key16.store(tag, Relaxed);
            self.depth8.store(new_d as u8, Relaxed);
            self.gen_bound8.store(
                generation | (b as u8) << BOUND_SHIFT | (pv as u8) << PV_SHIFT,
                Relaxed);
            self.value16.store(v as i16, Relaxed);
            self.eval16.store(ev as i16, Relaxed);
        }
        // Rule 3 — secondary aging: a REJECTED write against a deep,
        // non-EXACT entry holding a decisive (mate/TB-band) score erodes
        // it by one ply of stored depth, so stale mate claims age out
        // instead of squatting forever. Important for elementary mates.
        else if old_d + DEPTH_NONE >= 5
            && Bound::from_bits(self.gen_bound8.load(Relaxed) >> BOUND_SHIFT)
               != Bound::Exact
        {
            let v16 = self.value16.load(Relaxed) as Value;
            if v16.abs() < VALUE_INFINITE && is_decisive(v16) {
                self.depth8.store((old_d - 1).max(0) as u8, Relaxed);
            }
        }
    }
}
```

Notes, all SF-current (D20):

- The refresh margin is **4** and the PV credit **2**: condition (c) in
  internal-depth form is `new_d + 2·pv > old_d − 4`, identical in external
  form. Strict `>`: an equal-value write against a same-generation entry is
  rejected (and may then trigger Rule 3).
- The `v16.abs() < VALUE_INFINITE` guard exists because a racing writer may
  have left `VALUE_NONE` (32002) or garbage in `value16`; `is_decisive` is
  only meaningful on in-range values.
- The `max(0)` clamp guards racy decrements from wrapping the `u8`; hitting
  0 means "unoccupied", which is the graceful failure mode.
- Rule 3's depth threshold is **external depth ≥ 5** and the decrement is
  **1**.
- Field store order = field declaration order = probe's read order.
- Writes deliberately are plain relaxed stores, not compare-exchange loops:
  a lost race merely produces one slightly-wrong replacement decision, never
  UB and never a corrupted invariant that validation can't absorb.

### 3.8 penalize(): demote entries that keep failing the caller

```rust
impl TtWriter<'_> {
    pub fn penalize(&self, penalty: i32) {
        let d = self.entry.depth8.load(Relaxed) as i32;
        self.entry.depth8.store((d - penalty).max(0) as u8, Relaxed);
    }
}
```

Clamped at zero — a fully penalized entry becomes unoccupied (`depth8 == 0`)
and is the first replacement victim. The (single) call site is in search
(doc 07): a non-PV node whose TT entry had enough depth and a valid value
but whose bound direction mismatched the current window (so it could not cut
off) calls `writer.penalize(1)` when `depth > 5` — the entry is provably
useless for the windows this subtree keeps seeing. Constant SF-current
(D20).

### 3.9 Generations: new_search() and wraparound

```rust
impl TranspositionTable {
    /// Call once at the start of every root search ("go").
    pub fn new_search(&self) {
        let g = (self.generation8.load(Relaxed) + 1) & GENERATION_MASK;
        self.generation8.store(g, Relaxed);
    }
    pub fn generation(&self) -> u8 { self.generation8.load(Relaxed) }
}
```

32 generations cycle like a clock; `relative_age` (§3.2) does the wrapping
subtraction. `new_search` takes `&self` with an interior-mutable `AtomicU8`
(D23): workers reach the table through a shared `Arc<TT>` handle, and the
main worker bumps the generation at the start of every search through that
handle — a `&mut self` method would not compile there. The load+store pair
is not atomic as a unit, which is fine: `new_search` is only ever called
from one place (the main worker in `start_searching`) while helpers are
still parked. `resize`/`clear` DO take `&mut self`, obtained on the UCI
thread between searches via `Arc::get_mut` (doc 13 §3.6).

The generation is passed *into* `write(...)` by the caller (search reads
`tt.generation()` once per node) rather than re-read per write, matching SF
and keeping `TtWriter` free of a table back-reference.

### 3.10 hashfull: sampled occupancy in permille

```rust
/// Permille of entries written during the current search (age <= max_age).
/// UCI `info hashfull` uses max_age = 0.
pub fn hashfull(&self, max_age: u8) -> u32 {
    let mut count = 0;
    for cluster in &self.clusters[..1000] {
        for e in &cluster.entries {
            count += (e.is_occupied()
                   && e.relative_age(self.generation()) <= max_age) as u32;
        }
    }
    count / CLUSTER_SIZE as u32
}
```

Samples exactly the first **1000** clusters (3000 entries), counts occupied
entries with age ≤ `max_age`, divides by **3** → permille. Requires
`clusters.len() >= 1000`; `resize` enforces a 1 MB minimum (32768 clusters),
so the invariant holds. Cost: 32 KB scan, negligible at UCI info cadence.

### 3.11 Eval-cache role: DEPTH_UNSEARCHED entries

The TT doubles as a static-eval cache. When search reaches a node with no
usable TT entry, computes the static eval, and then leaves without a real
search result (the common case for the eval-cache write is: main search, TT
miss, after evaluation), it writes an *eval-only* entry:

```text
write(key, VALUE_NONE, ss.tt_pv, Bound::None, DEPTH_UNSEARCHED,
      Move::NULL, raw_eval, generation)
```

- `depth = DEPTH_UNSEARCHED = -2` → stored `depth8 = 1`: occupied, but
  minimum priority; the refresh rule overwrites it with almost anything.
- `value = VALUE_NONE`, `bound = Bound::None`: consumers see "no search
  result here", only `data.eval` is useful.
- **`eval` is the RAW, uncorrected static eval** (D22). Correction history
  is applied by the *reader* at probe time (`corrected = adjust(data.eval,
  correction)`), never baked into the cache — corrections drift over time
  and per thread, and a pre-corrected cached value would poison later reads.
  The same rule applies to `eval16` on every write, including real search
  writes: search passes `unadjusted_static_eval` through.
- Quiescence stand-pat fail-highs write a variant with `Bound::Lower` and
  `value = VALUE_NONE` at `DEPTH_UNSEARCHED` (SF-current shape; exact call
  sites are doc 07's).

This is why `DEPTH_UNSEARCHED` must compare below `DEPTH_QS = 0` and above
`DEPTH_NONE` (D6): TT-depth conditions like `data.depth >= depth` naturally
exclude eval-only entries from cutoffs, while `is_occupied` still protects
them (briefly) from replacement.

### 3.12 Mate-score conversion at the API boundary

Owned by search (design/07-search.md); restated here because it is a hard
precondition of `write` and postcondition of `probe`:

- **The TT stores node-relative decisive scores.** Callers of `write` must
  convert with `value_to_tt(v, ply)`: `is_win(v) → v + ply`,
  `is_loss(v) → v − ply`, else `v` ("mate in N from *here*", not from root).
- **Readers must convert back** with `value_from_tt(v, ply, rule50)`:
  `VALUE_NONE` passes through; wins return `v − ply` unless the score could
  be stale under the 50-move rule, in which case it is *downgraded* to just
  outside the decisive band: if `is_mate(v) && VALUE_MATE − v > 100 − rule50`
  or `VALUE_TB − v > 100 − rule50`, return `VALUE_TB_WIN_IN_MAX_PLY − 1`
  (losses mirrored with `+ ply` and `VALUE_TB_LOSS_IN_MAX_PLY + 1`).
- The TT itself never inspects values beyond the `i16` cast (debug-assert
  `-32768 <= v <= 32767`; all engine values fit — `VALUE_NONE = 32002 <
  32768`).

Related GHI mitigations live elsewhere but are part of why this design is
sound: the rule50 key folding (D9, Zobrist doc) separates same-position/
different-clock entries every 8 halfmove steps, and search refuses TT
cutoffs at `rule50 >= 96` (doc 07).

### 3.13 Allocation: portable Vec first, huge pages later

**Size limits (owned here, consumed by the UCI doc — this closes that doc's
open question on the `Hash` maximum):**

```rust
pub const MIN_HASH_MB: usize = 1;          // hashfull needs >= 1000 clusters
pub const MAX_HASH_MB: usize = 33_554_432; // SF's 64-bit cap (32 TiB)
```

The UCI `Hash` spin option is `default 16, min MIN_HASH_MB, max
MAX_HASH_MB` (SF-current values, D20). The cap is a sanity bound on the
option range, not a promise the allocation succeeds — which is why `resize`
must be fallible.

**Failure semantics.** `Vec::with_capacity` calls `handle_alloc_error` and
aborts the process on failure; an engine that dies on `setoption name Hash
value 1048576` on a 32 GB machine forfeits the game. So `resize` never uses
the infallible path. Phase 1 (lands with the module — zero unsafe, zero
dependencies, D15):

```rust
/// Resize to `mb` MiB, clamped to MIN_HASH_MB..=MAX_HASH_MB. Returns the
/// MiB actually allocated: on allocation failure it falls back through
/// halved sizes, always leaving a usable table. Never aborts on a
/// mid-session `setoption` (only startup default allocation may be fatal,
/// doc 01 §3.4).
pub fn resize(&mut self, mb: usize) -> usize {
    let mb = mb.clamp(MIN_HASH_MB, MAX_HASH_MB);
    self.clusters = Vec::new();          // drop old buffer first: peak RSS
    *self.generation8.get_mut() = 0;
    let mut try_mb = mb;
    loop {
        let count = try_mb * 1024 * 1024 / size_of::<Cluster>();
        let mut v: Vec<Cluster> = Vec::new();
        if v.try_reserve_exact(count).is_ok() {
            v.resize_with(count, Cluster::zeroed); // stays within capacity
            self.clusters = v;
            return try_mb;
        }
        assert!(try_mb > MIN_HASH_MB, "cannot allocate even a 1 MiB TT");
        try_mb /= 2;                     // integer halving reaches 1 MiB
    }
}
```

The UCI layer compares the return value against the request and on mismatch
prints `info string Failed to allocate <req> MB hash, using <got> MB`
(exact wording owned by the UCI doc), then keeps playing with the smaller
table. The `assert!` documents the one deliberate exception: if even the
1 MiB floor fails, the process is out of memory for good and dies either
way (`panic = "abort"` in release, D16). Residual risk: on
overcommit-happy Linux, `try_reserve_exact` can succeed and the kernel may
still OOM-kill on first touch inside `resize_with`; the cap plus the
fallible path handle every honest failure mode, and that residue is one
every engine accepts.

`Cluster::zeroed()` is a `const fn` of `AtomicXX::new(0)`; `resize_with`
compiles to a memset-shaped loop and cannot reallocate (capacity is already
reserved). Default size 16 MB (SF default); only that initial startup
allocation may be fatal (stderr diagnostic + nonzero exit, doc 01 §3.4) —
never a `setoption` mid-session.

Phase 2 (later, Linux-gated, measured — TT probes are random misses and 2 MB
pages cut TLB misses for real Elo): replace the `Vec` with a small
`TtBuffer` abstraction that on `cfg(target_os = "linux")` allocates via
`std::alloc::alloc_zeroed(Layout::from_size_align(bytes, 2 * 1024 * 1024))`
— 2 MB alignment is what lets the kernel back the region with huge pages —
then issues `madvise(ptr, len, MADV_HUGEPAGE)` (advice value 14) so
`madvise`-mode THP kernels promote it (`always`-mode kernels promote the
aligned region without the call); `madvise` failure is ignored (it is
advisory), and allocation failure falls back through the Phase-1 fallible
path. Be honest about the unsafe surface — this is more than one small
block. `alloc_zeroed` with a custom `Layout` must be paired with `dealloc`
using the *same* layout, so `TtBuffer` stores its layout and needs a manual
`Drop` plus `Deref` to `[Cluster]` — several `unsafe` blocks, each with a
`// SAFETY:` comment. And under D15's zero-runtime-dependency rule, calling
`madvise` from std-only Rust means a raw syscall via stable inline `asm!` —
Linux- and x86-64-specific twice over (the syscall number 28 AND the asm),
a fourth unsafe category beyond D14's three, with its own
`#![deny(unsafe_code)]` module-allow bookkeeping. Two acceptable shapes,
one of which must be chosen and recorded in decisions.md *before* Phase 2
starts (nothing earlier blocks on it — see Open questions):

- **(a) Preferred**: a `cfg(target_os = "linux")`-gated `libc` dependency
  for the single, well-typed `madvise` call, amending D15 ("target-gated
  platform bindings for page management allowed"). The aligned
  alloc/dealloc unsafe remains and is sanctioned by the proposed D14
  clause (d).
- **(b) Dependency-free**: a self-contained `tt::hugepage` module owning
  both the `asm!`-based `madvise` and the aligned alloc/dealloc pair behind
  its own `Drop`, with the D14 clause (d) amendment.

The fallback on any other platform is the Phase-1 `Vec`. SF additionally
uses explicit 1 GB hugetlb pages for huge tables; we skip that (diminishing
returns, heavy setup requirements).

### 3.14 clear() and the future multithreaded clear

```rust
pub fn clear(&mut self) {
    *self.generation8.get_mut() = 0;
    for cluster in &mut self.clusters {
        for e in &mut cluster.entries {
            *e.key16.get_mut() = 0;
            *e.depth8.get_mut() = 0;
            *e.gen_bound8.get_mut() = 0;
            *e.move16.get_mut() = 0;
            *e.value16.get_mut() = 0;
            *e.eval16.get_mut() = 0;
        }
    }
}
```

`&mut self` gives exclusive access, so `Atomic::get_mut` yields plain `&mut`
integers — safe, no atomic ops, and the optimizer collapses it to a memset.
For multi-GB tables this single-threaded clear takes noticeable wall time;
when the thread pool exists (threads doc, D23), split `self.clusters` with
`chunks_mut` across scoped threads — each thread zeroes a contiguous slice
(which also gives first-touch NUMA placement for free). Not needed for the
single-threaded milestone.

`resize` implies `clear` (fresh buffer is already zeroed). UCI `Clear Hash`
maps to `clear()`; `ucinewgame` calls it too (UCI doc).

### 3.15 Prefetch wrapper

```rust
impl TranspositionTable {
    /// Hint the cache to pull the probe target for `key`. No-op off x86_64.
    #[inline]
    pub fn prefetch(&self, key: Key) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let p: *const Cluster = self.cluster_for(key);
            // SAFETY: prefetch reads nothing and writes nothing; the
            // pointer derives from a live reference into our allocation.
            unsafe { _mm_prefetch(p.cast::<i8>(), _MM_HINT_T0) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = key;
    }
}
```

Sanctioned by D14(a). Callers (owned by docs 02/07, listed here for the API
shape): (1) inside `make_move`, as soon as the new key is computed and
before the rest of the update — hides most of the probe's cache-miss
latency; (2) earlier still in search's move loop via a cheap *predicted* key
(piece-square XORs only, ignoring castling/ep/promotion side effects — wrong
keys just prefetch a useless line, harmlessly). An aarch64 `prfm` inline-asm
variant can be added later if we ever target it; the no-op fallback is
correct everywhere.

## 4. Implementation plan

Each step is one commit; `cargo test` and `cargo clippy --all-targets` clean
at every step (D24 gates apply once search integration exists).

1. **Prerequisite check** — confirm `types.rs` has the D5/D6 constants and
   predicates (`VALUE_*`, `DEPTH_*`, `is_valid`/`is_win`/`is_loss`/
   `is_decisive`/`is_mate`); land them if the search-doc work hasn't.
   Done when: constants + predicate unit tests pass.
2. **`Move::raw16` / `Move::from_raw16` prerequisite** — owned by the
   core-types doc (design/02-core-types.md §3.7, `pub const fn`); confirm
   they landed with their roundtrip tests, land them there first if not.
   Done when: `from_raw16(m.raw16()) == m` for all move kinds; arbitrary
   bits decode without panic (doc 02's tests, step 11 there).
3. **`tt.rs` skeleton: `Bound`, `TtEntry`, `Cluster`** — packing constants,
   const size/align asserts, `read`, `is_occupied`, `relative_age`,
   `Cluster::zeroed`. Done when: unit tests cover gen_bound8 pack/unpack for
   all (gen, bound, pv) combos and wraparound ages for all 32×32 pairs.
4. **`TtEntry::save` + `TtWriter` (`write`, `penalize`)** — Rules 1–3 of
   §3.7, clamps of §3.8. Done when: the replacement-policy unit tests of §5
   (preservation, refresh boundary, EXACT, older-gen, secondary aging,
   penalize clamp) all pass.
5. **`TranspositionTable`** — `Vec` storage, `resize`, `clear`,
   `new_search`, `generation`, `mul_hi64` + `cluster_for`, `probe`,
   `hashfull`; `pub(crate) fn with_cluster_count(n)` test constructor.
   Done when: probe/miss/victim and hashfull tests pass; a 1024 MB resize
   works; `size_of::<Cluster>() * clusters.len()` equals the requested MB;
   the §5 clamp and allocation-failure tests pass (no abort on absurd
   sizes, table usable afterwards).
6. **Prefetch wrapper** (§3.15). Done when: builds on x86_64 and (via
   `cargo check --target aarch64-unknown-linux-gnu`) on a non-x86 target.
7. **Search integration** (call sites owned by doc 07; this step is joint) —
   probe at node entry, write at node exit, eval-cache writes, `new_search`
   per `go`, prefetch in `make_move`. Done when: `bench` node count drops
   materially versus TT disabled and repeated single-threaded runs are
   bit-identical (D24).
8. **UCI surfacing** — `Hash`, `Clear Hash`, `info hashfull` (UCI doc owns;
   TT API already sufficient). Done when: GUI shows growing hashfull.
9. **Multithreaded clear** — with the thread pool (threads doc). Done when:
   clear of a 4 GB table scales with threads.
10. **Huge-page path** (§3.13 Phase 2) behind `cfg(target_os = "linux")`.
    Done when: `/proc/<pid>/smaps_rollup` shows `AnonHugePages` covering the
    TT and bench shows a measured nps gain (record the number in the
    commit); falls back cleanly when allocation fails.

## 5. Testing and validation

Unit tests (in `tt.rs`, using `with_cluster_count(1)` so every key lands in
cluster 0 and tags are controlled via low-16 bits):

- **Layout**: compile-time `const` asserts for entry size 10 / align 2 /
  cluster size 32 (these are the tests — a layout regression fails to
  compile).
- **Roundtrip**: for a grid of (value ∈ {−VALUE_INFINITE+1, −1, 0, 1,
  mate-band values, VALUE_NONE}, eval, depth ∈ {DEPTH_UNSEARCHED, 0, 1, …,
  240}, bound × pv × move kinds): `write` then `probe` returns `hit` and
  identical decoded fields.
- **Occupancy**: fresh table probes report `hit == false`; a zero key
  (tag 0) against a zeroed cluster returns `hit == false` with a writer into
  that cluster (the tag-matches-empty path of §3.5 step 2).
- **Replacement victim**: fill a cluster with three entries of chosen
  (depth8, age) pairs and assert the argmin of `depth8 − 8·age` is evicted;
  include the case where a 2-generations-old depth-20 entry loses to a
  current depth-5 entry (20 − 16 < 5 − 0).
- **save rules**, one test each:
  - EXACT overwrites a deeper same-key, same-generation entry.
  - Key change overwrites regardless of depth.
  - Refresh boundary: same key/gen, non-exact — `new_d + 2·pv = old_d − 4`
    is rejected, `old_d − 4 + 1` is accepted (test both pv values).
  - Older-generation entries are overwritten by shallow new data after
    `new_search()`.
  - ttMove preservation: same-key write with `Move::NULL` keeps the old
    move while updating value/depth; different-key write with NULL clears it.
  - Secondary aging: rejected write against external-depth ≥ 5, non-EXACT,
    decisive-value entry decrements depth8 by exactly 1; repeated rejections
    march it to 0 (unoccupied); non-decisive or EXACT entries are untouched.
  - `penalize(1)` decrements; `penalize(255)` clamps to 0 and the entry
    stops counting in `hashfull` and reports `hit == false` on re-probe.
- **Aging wraparound**: all 32×32 (current, entry) generation pairs give
  `relative_age == (cur − ent) mod 32`; bound/pv bits set to all 8 combos
  must not perturb the age.
- **Indexing**: `mul_hi64(k, n) < n` for adversarial keys
  {0, 1, u64::MAX, high-bit patterns} × counts {1, 3, 1000, 32768, odd
  primes}; distribution sanity: 1e6 random keys over 1000 clusters, no
  cluster > 3× mean (loose smoke bound).
- **Collision behavior** (documents the accepted defect): craft two distinct
  keys with equal low-16 bits mapping to cluster 0; write A, probe B →
  `hit == true` with A's payload. The test asserts this happens — pinning
  the contract that consumers MUST validate (`pseudo_legal`+`legal`,
  `is_valid`) rather than pretending it can't.
- **hashfull**: empty → 0; exactly the first 1000 clusters filled with
  current-gen entries → 1000; after `new_search()`, `hashfull(0) == 0` but
  `hashfull(1) == 1000`.
- **Resize clamping and allocation failure** (§3.13): `resize(0)` yields
  `MIN_HASH_MB`; `resize(MAX_HASH_MB + 1)` clamps to at most `MAX_HASH_MB`.
  A `resize` to an absurd size (beyond physical memory + swap, e.g. several
  TiB, still under the cap) must NOT abort the process: it returns a smaller
  achieved size and the table stays usable — a write/probe roundtrip passes
  afterwards. Caveat: on hosts with `vm.overcommit_memory = 1` the reserve
  can succeed and first-touch can OOM-kill; gate/`#[ignore]` this test where
  overcommit is unconditional (documented residual risk, §3.13).
- **Concurrency smoke** (`--release`, ignored under miri): 4+ threads hammer
  `probe`/`write`/`penalize` on a 1 MB table with overlapping keys for ~1 s;
  passes if nothing panics and every decoded `TtData` is structurally sane
  (bound decodes, depth in `DEPTH_NONE..=255+DEPTH_NONE`). Asserts
  crash-freedom, not consistency — consistency is explicitly not promised.
- **Miri**: `cargo +nightly miri test tt::` on the Vec path must be clean —
  it verifies the atomics usage has no UB (the huge-page path is
  `cfg`-excluded under miri; the concurrency smoke test is `#[ignore]`d
  there for time). Record the miri invocation in CI notes.

Regression strategy (D24): the TT does not touch movegen, so perft is
unaffected by construction (any perft failure after TT work indicates key
plumbing damage — investigate the Zobrist layer, not the TT). The `bench`
node count is the oracle: every search-visible TT change (policy constants,
integration) must report its new bench number; refactors inside `tt.rs`
must leave it bit-identical. Strength claims for the huge-page path or any
policy deviation go through SPRT (D25).

## 6. Performance notes

Targets and things to measure:

- **Probe = one cache line.** The 32-byte aligned cluster guarantees it;
  verify no accidental `Cluster` layout change via the const asserts.
- **Index cost**: `mul_hi64` should compile to a single widening multiply
  (check with `cargo asm` or godbolt once); if the slice bounds check on
  `cluster_for` ever shows up in a profile, that is the designated D14(c)
  candidate — record the measurement before switching to `get_unchecked`.
- **Prefetch distance**: the make_move-time prefetch typically covers the
  probe latency; measure bench nps with the prefetch no-op'd vs enabled
  (expect low single-digit % on big TTs; keep the number in the commit).
- **Huge pages**: expect a measurable nps gain on 256 MB+ tables from TLB
  relief (SF observed real Elo). Measure bench at 16 MB and 1 GB with and
  without `MADV_HUGEPAGE`, confirm via `AnonHugePages` in smaps.
- **Write traffic**: `save` stores at most 6 fields; the early-out (Rule 2
  rejection) path stores at most 1 (move) — don't "optimize" the move store
  behind an extra load of `move16`; the unconditional store is cheaper than
  a compare on contended lines.
- **hashfull cost** is a 32 KB scan — only call it at UCI-info cadence
  (~1 Hz), never per node.
- **Pitfalls**: (1) do NOT round cluster count to a power of two — mul_hi64
  exists precisely so a 512 MB request uses all 512 MB; (2) never store
  corrected evals (§3.11); (3) never store root-relative mate scores
  (§3.12); (4) `resize` must drop the old buffer before allocating the new
  one or peak RSS doubles; (5) keep `TtWriter` non-`Clone` and short-lived —
  it pins a probe decision that goes stale after deeper re-searches at the
  same node re-probe.

## 7. Interactions

- **design/07-search.md** (primary consumer): probe/write call sites, TT
  cutoff conditions (including the rule50 ≥ 96 no-cutoff rule and depth ≥ 7
  next-ply verification), `value_to_tt`/`value_from_tt` ownership,
  `penalize(1)` trigger, eval-cache reads/writes, `ss.tt_pv` propagation,
  `new_search()` per `go`.
- **design/02-core-types.md**: supplies `Move::raw16`/`from_raw16` (§3.7
  there) for `move16` packing, plus the `Value`/`Depth` constants and
  predicates listed in §3.1.
- **Board/Zobrist doc** (the make/unmake + keys design, D8/D9): supplies the
  64-bit `key` with ep-only-when-legal and rule50-folding semantics the TT's
  correctness leans on; hosts the in-make_move prefetch call and the cheap
  predicted-key prefetch helper.
- **Movegen doc**: `Board::pseudo_legal(Move)` / `Board::legal(Move)` — the
  ttMove validation half of the §3.5 contract.
- **MovePicker doc**: consumes the validated ttMove as its first stage.
- **UCI doc**: `Hash` option (default 16 MB, range
  `MIN_HASH_MB..=MAX_HASH_MB` = 1..=33554432 per §3.13 — this doc owns the
  limits, closing the UCI doc's open question on the maximum); the UCI
  layer prints the fallback `info string` when `resize` returns less than
  requested. `Clear Hash` button, `ucinewgame → clear()`, `info hashfull`
  from `hashfull(0)`.
- **Threads doc (D23)**: `&TranspositionTable` shared via `std::thread::scope`;
  multithreaded `clear`; the `&mut` phasing argument of §3.9.
- **Bench doc (D24)**: bench node count is this component's regression
  oracle post-integration.

(Filenames other than 07 to be normalized against the final suite index —
see Open questions.)

## 8. Acceptance criteria (D26)

The TT component is done when all of the following hold:

1. Const asserts pin `size_of::<TtEntry>() == 10`,
   `size_of::<Cluster>() == 32`, `align_of::<Cluster>() == 32`; the crate
   compiles with them.
2. All §5 unit tests pass, including all six save-rule tests, the 32×32
   aging wraparound matrix, the collision-contract test, and the hashfull
   tests; `cargo clippy --all-targets` is clean.
3. `cargo +nightly miri test tt::` passes on the portable path.
4. The concurrency smoke test runs clean (no panic, structurally sane
   decodes) with ≥ 4 threads.
5. `resize` honors any MB size from 1 to at least 1024, uses all requested
   memory (cluster count = MB·2^20/32, no power-of-two rounding), clamps
   requests to `MIN_HASH_MB..=MAX_HASH_MB` (1..=33554432), and `clear`
   restores generation 0 / hashfull 0. An unsatisfiable `resize` never
   aborts the process: it falls back to a smaller size, reports the
   achieved size to the caller, and leaves a usable table (§5's
   allocation-failure test passes).
6. Integrated with search (doc 07): fixed-depth bench with the TT enabled
   searches materially fewer nodes than with it disabled, repeated
   single-threaded runs are bit-identical, and the bench number is recorded
   (D24).
7. `hashfull(0)` is exposed and observed rising toward 1000 in a long
   search from the start position.
8. The only `unsafe` in the module is the prefetch intrinsic (until the
   huge-page commit lands, which adds the documented allocation/madvise
   blocks with `// SAFETY:` comments and a measured justification, D14).

## 9. Open questions

1. **D14/D15 amendment for huge pages**: the Phase-2 allocator needs
   `unsafe` beyond D14's three categories — raw `alloc_zeroed`/`dealloc`
   with a stored 2 MB-aligned layout behind a manual `Drop`, plus
   `madvise`, which under D15's zero-dependency rule means stable inline
   `asm!` (Linux/x86-64-specific in both the syscall number and the asm).
   §3.13 specifies the two acceptable shapes: (a) a `cfg`-gated `libc`
   dependency for `madvise` (preferred: one well-typed call, weaker D15,
   needs a D15 amendment) or (b) a self-contained `tt::hugepage` module
   with the asm syscall (stronger D15, more unsafe surface). Either shape
   also needs D14 extended with "(d) page-management syscalls/allocation
   for the TT, Linux-gated, advisory-failure-tolerant". The choice must be
   recorded in decisions.md before Phase 2 starts; nothing before then
   depends on it.
2. **Sibling-doc filenames**: cross-references other than
   design/07-search.md (Zobrist/board, movegen, movepick, UCI, threads,
   bench) are cited by topic; normalize to exact filenames when the suite
   index is assembled.
3. **`hashfull(max_age > 0)`**: SF exposes the parameter (some callers use a
   nonzero age for saturation heuristics). We expose the parameter now but
   only wire `max_age = 0`; revisit if our time manager (timeman doc) wants
   a saturation signal.
