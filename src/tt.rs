//! Transposition table — the engine's shared, lock-free cache of search
//! results, keyed by the 64-bit Zobrist key (design/08-transposition-table.md,
//! D19, D23).
//!
//! Layout and policy are SF-current (D20): a 10-byte [`TtEntry`] of relaxed
//! atomics (`key16`, `depth8`, `gen_bound8`, `move16`, `value16`, `eval16`),
//! 3-entry 32-byte clusters that never straddle a cache line, multiply-high
//! indexing (any table size, no power-of-two rounding), 5-bit generation
//! aging, and Stockfish's replacement rules in [`TtEntry::save`]. Everything
//! is deliberately racy: individual field accesses are atomic (no UB), but an
//! entry read concurrently with a write may mix two writers. That is the
//! doc's §3.5 validation contract — consumers MUST validate what they read
//! (`pseudo_legal` + `legal` on the move, `is_valid` on values, bound
//! inclusion before use) instead of paying for synchronization on every
//! probe.
//!
//! Landed here (doc 08 §4 steps 3–6): entry/cluster layout with const
//! size asserts, [`Bound`], probe/save/penalize, replacement + aging,
//! [`TranspositionTable`] with portable `Vec` storage, fallible
//! [`TranspositionTable::resize`], [`TranspositionTable::clear`],
//! [`TranspositionTable::new_search`] (interior-mutable generation, D23),
//! [`TranspositionTable::hashfull`], and the
//! [`TranspositionTable::prefetch`] wrapper (§3.15). Deliberately NOT here
//! yet, per the doc's staging: search integration (step 7, doc 07), the
//! multithreaded clear (step 9), and the Linux huge-page allocator (§3.13
//! phase 2, step 10).
//!
//! Relaxed atomics are the sanctioned safe equivalent of SF's
//! `RelaxedAtomic<T>` fields (D14); the ONE `unsafe` block in this module
//! is the prefetch intrinsic below.
//!
//! Validation (doc 08 §5): `cargo +nightly miri test --lib tt::` must stay
//! clean on this portable path (the wall-clock concurrency smoke and the
//! MiB-scale/1e6-iteration tests are `#[ignore]`d under miri for
//! interpreter time — their atomics surface runs at small scale in
//! `small_table_lifecycle_exercises_all_paths`; run the full set natively),
//! and the non-x86_64 no-op prefetch builds via
//! `cargo check --target aarch64-unknown-linux-gnu`.

// D14(a): module-scoped relaxation of the crate-wide `#![deny(unsafe_code)]`
// for exactly one sanctioned use — the `_mm_prefetch` cache hint in
// [`TranspositionTable::prefetch`] (doc 08 §3.15, acceptance criterion 8:
// "the only unsafe in the module is the prefetch intrinsic"). Each block
// carries its `// SAFETY:` comment; the measured justification lives on the
// method's doc comment (and `prefetch_microbenchmark` reproduces it).
#![allow(unsafe_code)]

use std::sync::atomic::{AtomicI16, AtomicU8, AtomicU16, Ordering::Relaxed};

use crate::moves::Move;
use crate::types::{DEPTH_NONE, Depth, Key, VALUE_INFINITE, VALUE_NONE, Value, is_decisive};

// ---------------------------------------------------------------------------
// Packing constants (doc 08 §3.1, SF-current, D20)
// ---------------------------------------------------------------------------

/// Entries per cluster; one probe touches exactly one 32-byte cluster.
pub const CLUSTER_SIZE: usize = 3;

const GENERATION_BITS: u8 = 5;
/// Low 5 bits of `gen_bound8`: the entry's generation, a 32-cycle clock.
const GENERATION_MASK: u8 = (1 << GENERATION_BITS) - 1; // 0x1F
const BOUND_SHIFT: u8 = GENERATION_BITS; // 5
const PV_SHIFT: u8 = BOUND_SHIFT + 2; // 7
const PV_MASK: u8 = 1 << PV_SHIFT; // 0x80

crate::tunables! {
    /// **The transposition table's replacement policy, made tunable
    /// (2026-07-27).**
    ///
    /// The TT decides what the entire search is allowed to reuse, and until
    /// now every number in that decision was a plain `const` — no SPSA
    /// campaign could reach any of it. Two of the five were also *implicit*:
    /// a coefficient of 1 that nobody wrote down, invisible to design/01
    /// §3.8's "bare numeric literal" rule because they are not literals.
    ///
    /// All default to exact identity with the previous consts.

    /// Weight on stored depth when choosing a replacement victim:
    /// `TT_REPLACE_DEPTH_WEIGHT * depth - TT_REPLACE_AGE_WEIGHT * age`.
    /// Was an implicit 1 — the depth term had no coefficient at all, so the
    /// depth-vs-age exchange rate could only ever be moved from one side.
    pub TT_REPLACE_DEPTH_WEIGHT: i32 = 1;
    /// Plies of stored depth one generation of age is worth when choosing a
    /// replacement victim. Was `REPLACEMENT_AGE_WEIGHT`.
    pub TT_REPLACE_AGE_WEIGHT: i32 = 8;
    /// How much shallower a re-write of the SAME position may be and still
    /// overwrite. Was `REFRESH_DEPTH_MARGIN`.
    pub TT_REFRESH_DEPTH_MARGIN: i32 = 4;
    /// Depth credit a PV result carries into the refresh test. Was
    /// `REFRESH_PV_CREDIT`.
    pub TT_REFRESH_PV_CREDIT: i32 = 2;
    /// Minimum stored depth for secondary aging to erode a stale decisive
    /// entry. Was `SECONDARY_AGING_MIN_DEPTH`.
    pub TT_SECONDARY_AGING_MIN_DEPTH: i32 = 5;
    /// Plies of stored depth a rejected write erodes from a stale decisive
    /// entry. Another implicit 1 (`old_d - 1`): it sets how fast a wrong mate
    /// claim ages out, which is exactly what elementary-mate regressions turn
    /// on, and it was never adjustable.
    pub TT_SECONDARY_AGING_STEP: i32 = 1;
}

/// `hashfull` samples exactly this many leading clusters (doc 08 §3.10);
/// `resize`'s [`MIN_HASH_MB`] floor guarantees they exist.
const HASHFULL_SAMPLE_CLUSTERS: usize = 1000;

/// Smallest `Hash` size in MiB (doc 08 §3.13): keeps `hashfull`'s
/// 1000-cluster sample valid (1 MiB = 32768 clusters).
pub const MIN_HASH_MB: usize = 1;
/// Largest `Hash` size in MiB — SF's 64-bit cap (32 TiB). A sanity bound on
/// the UCI option range, not a promise the allocation succeeds ([`
/// TranspositionTable::resize`] is fallible).
pub const MAX_HASH_MB: usize = 33_554_432;

/// 5-bit clock arithmetic on the packed `gen_bound8` byte (doc 08 §3.2):
/// the wrapping subtraction makes the bound/PV bits cancel out under the
/// mask (any multiple of 32 vanishes mod 32).
const fn relative_age_bits(gen_bound8: u8, generation: u8) -> u8 {
    generation.wrapping_sub(gen_bound8) & GENERATION_MASK
}

// ---------------------------------------------------------------------------
// Bound
// ---------------------------------------------------------------------------

/// What a stored TT value proves about the true value (doc 08 §3.1).
///
/// The discriminants keep SF's bit-flag semantics — [`Bound::Exact`] is
/// literally `Upper | Lower` — and search tests bound *inclusion*
/// ([`Bound::includes_lower`]/[`Bound::includes_upper`]), never equality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    /// No search result (eval-only entries, empty slots).
    None = 0,
    /// Fail-low result: the true value is <= the stored value.
    Upper = 1,
    /// Fail-high result: the true value is >= the stored value.
    Lower = 2,
    /// Exact (PV) result: `Upper | Lower`.
    Exact = 3,
}

impl Bound {
    /// True for `Lower` and `Exact` — the stored value is usable as a
    /// lower bound (fail-high direction).
    pub const fn includes_lower(self) -> bool {
        self as u8 & Bound::Lower as u8 != 0
    }

    /// True for `Upper` and `Exact` — the stored value is usable as an
    /// upper bound (fail-low direction).
    pub const fn includes_upper(self) -> bool {
        self as u8 & Bound::Upper as u8 != 0
    }

    /// Decode 2 bits; total over all inputs (masks to the low 2 bits), so a
    /// racy `gen_bound8` byte always decodes to *some* bound.
    const fn from_bits(bits: u8) -> Bound {
        match bits & 0b11 {
            0 => Bound::None,
            1 => Bound::Upper,
            2 => Bound::Lower,
            _ => Bound::Exact,
        }
    }
}

// ---------------------------------------------------------------------------
// TtEntry: 10 bytes of relaxed atomics (doc 08 §3.2)
// ---------------------------------------------------------------------------

/// One TT slot. Field order matches probe's read order and save's store
/// order (memory is fastest sequentially; SF orders it this way
/// deliberately). Every access is `Ordering::Relaxed`: individual loads and
/// stores cannot tear, so there is no UB — but there is no entry-wide
/// atomicity either (no lock, no seqlock, no checksum). See the module docs
/// for the consumer validation contract that makes this acceptable.
#[repr(C)]
struct TtEntry {
    /// Low 16 bits of the Zobrist key: the in-cluster tag. The high bits
    /// are consumed by the cluster index.
    key16: AtomicU16,
    /// Depth stored as `depth - DEPTH_NONE`, so the all-zero entry decodes
    /// to `DEPTH_NONE` and `depth8 != 0` doubles as the occupancy flag.
    depth8: AtomicU8,
    /// `generation | bound << 5 | pv << 7`.
    gen_bound8: AtomicU8,
    /// Best move, raw [`Move`] bits; 0 (`Move::NULL`) = none.
    move16: AtomicU16,
    /// Search value, already ply-converted by the caller (doc 08 §3.12).
    value16: AtomicI16,
    /// RAW static eval, uncorrected (doc 08 §3.11, D22).
    eval16: AtomicI16,
}

const _: () = assert!(size_of::<TtEntry>() == 10);
const _: () = assert!(align_of::<TtEntry>() == 2);

impl TtEntry {
    const fn zeroed() -> Self {
        TtEntry {
            key16: AtomicU16::new(0),
            depth8: AtomicU8::new(0),
            gen_bound8: AtomicU8::new(0),
            move16: AtomicU16::new(0),
            value16: AtomicI16::new(0),
            eval16: AtomicI16::new(0),
        }
    }

    /// A freshly cleared entry has `depth8 == 0` (`DEPTH_NONE`); every real
    /// write stores `depth - DEPTH_NONE > 0`.
    fn is_occupied(&self) -> bool {
        self.depth8.load(Relaxed) != 0
    }

    /// How many generations ago this entry was written, on the 32-cycle
    /// clock (doc 08 §3.2).
    fn relative_age(&self, generation: u8) -> u8 {
        relative_age_bits(self.gen_bound8.load(Relaxed), generation)
    }

    /// Decode the entry into a plain copy, one relaxed load per field in
    /// declaration order. The copy may be self-inconsistent under
    /// concurrent writes — the consumer contract absorbs that.
    fn read(&self) -> TtData {
        let depth8 = self.depth8.load(Relaxed);
        let gen_bound8 = self.gen_bound8.load(Relaxed);
        let move16 = self.move16.load(Relaxed);
        let value16 = self.value16.load(Relaxed);
        let eval16 = self.eval16.load(Relaxed);
        TtData {
            mv: Move::from_raw16(move16),
            value: Value::from(value16),
            eval: Value::from(eval16),
            depth: Depth::from(depth8) + DEPTH_NONE,
            bound: Bound::from_bits(gen_bound8 >> BOUND_SHIFT),
            is_pv: gen_bound8 & PV_MASK != 0,
        }
    }

    /// The write policy, rule by rule (doc 08 §3.7). All accesses relaxed;
    /// each field is loaded at most once into a local (a benign
    /// simplification of SF's repeated loads). Plain stores, not
    /// compare-exchange loops: a lost race merely produces one
    /// slightly-wrong replacement decision, never UB.
    // The 8-parameter shape is the doc 08 §3.7 contract (mirrors SF's save).
    #[allow(clippy::too_many_arguments)]
    fn save(
        &self,
        key: Key,
        v: Value,
        pv: bool,
        b: Bound,
        d: Depth,
        m: Move,
        ev: Value,
        generation: u8,
    ) {
        let tag = key as u16;
        #[cfg(feature = "ttcensus42")]
        crate::ttcensus42::bump(&crate::ttcensus42::SAVES);
        let same_key = tag == self.key16.load(Relaxed);

        // Rule 1 — ttMove preservation: keep the old move only when the
        // new write is for the same position and brings no move. Runs even
        // when rule 2 rejects the rest of the write.
        if m != Move::NULL || !same_key {
            self.move16.store(m.raw16(), Relaxed);
        }

        let new_d = d - DEPTH_NONE; // internal form
        let old_d = i32::from(self.depth8.load(Relaxed)); // internal form
        let gen_bound = self.gen_bound8.load(Relaxed);

        // Rule 2 — overwrite test, cheapest checks first. Write if ANY of:
        //   (a) the new bound is EXACT (PV results always win),
        //   (b) the slot holds a different position,
        //   (c) refresh rule: same position may re-write with up to
        //       REFRESH_DEPTH_MARGIN less depth; PV nodes get a credit,
        //   (d) the stored entry is from an older search.
        if b == Bound::Exact
            || !same_key
            || new_d + TT_REFRESH_PV_CREDIT() * i32::from(pv) > old_d - TT_REFRESH_DEPTH_MARGIN()
            || relative_age_bits(gen_bound, generation) != 0
        {
            debug_assert!(d > DEPTH_NONE && new_d < 256);
            debug_assert!(generation <= GENERATION_MASK);
            debug_assert!(i16::try_from(v).is_ok() && i16::try_from(ev).is_ok());
            #[cfg(feature = "ttcensus42")]
            crate::ttcensus42::bump(&crate::ttcensus42::SAVE_ACCEPT);
            self.key16.store(tag, Relaxed);
            self.depth8.store(new_d as u8, Relaxed);
            self.gen_bound8.store(
                generation | (b as u8) << BOUND_SHIFT | u8::from(pv) << PV_SHIFT,
                Relaxed,
            );
            self.value16.store(v as i16, Relaxed);
            self.eval16.store(ev as i16, Relaxed);
        }
        // Rule 3 — secondary aging: a REJECTED write against a deep,
        // non-EXACT entry holding a decisive (mate/TB-band) score erodes it
        // by one ply of stored depth, so stale mate claims age out instead
        // of squatting forever. Important for elementary mates.
        else if old_d + DEPTH_NONE >= TT_SECONDARY_AGING_MIN_DEPTH()
            && Bound::from_bits(gen_bound >> BOUND_SHIFT) != Bound::Exact
        {
            let v16 = Value::from(self.value16.load(Relaxed));
            // A racing writer may have left VALUE_NONE or garbage in
            // value16; is_decisive is only meaningful on in-range values.
            if v16.abs() < VALUE_INFINITE && is_decisive(v16) {
                // max(0) guards racy decrements from wrapping the u8;
                // hitting 0 means "unoccupied" — the graceful failure mode.
                self.depth8
                    .store((old_d - TT_SECONDARY_AGING_STEP()).max(0) as u8, Relaxed);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cluster: 3 entries + 2 pad bytes = 32 bytes (doc 08 §3.3)
// ---------------------------------------------------------------------------

/// Half a cache line: a probe touches one line, and `align(32)` guarantees
/// a cluster never straddles a line boundary.
#[repr(C, align(32))]
struct Cluster {
    entries: [TtEntry; CLUSTER_SIZE], // 30 bytes
    _pad: [u8; 2],                    // 32 total
}

const _: () = assert!(size_of::<Cluster>() == 32);
const _: () = assert!(align_of::<Cluster>() == 32);

impl Cluster {
    const fn zeroed() -> Self {
        Cluster {
            entries: [const { TtEntry::zeroed() }; CLUSTER_SIZE],
            _pad: [0; 2],
        }
    }
}

// ---------------------------------------------------------------------------
// Probe API: decoded copy + write handle (doc 08 §3.5)
// ---------------------------------------------------------------------------

/// A plain decoded copy of one entry. `value` is node-relative (doc 08
/// §3.12) and meaningful only under the validation contract: check
/// `is_valid`, use only under the matching bound direction, and convert
/// through `value_from_tt` (doc 07 owns the call sites).
#[derive(Debug, Clone, Copy)]
pub struct TtData {
    /// Stored best move; [`Move::NULL`] when absent. Semantic validity is
    /// the consumer's job (`pseudo_legal` + `legal` before any use).
    pub mv: Move,
    /// Search value; only meaningful when the probe `hit`.
    pub value: Value,
    /// RAW static eval or [`VALUE_NONE`].
    pub eval: Value,
    /// External depth; [`DEPTH_NONE`] when absent.
    pub depth: Depth,
    pub bound: Bound,
    pub is_pv: bool,
}

impl TtData {
    /// What a probe miss returns. Note a tag-matched but unoccupied entry
    /// decodes to depth `DEPTH_NONE` and value 0 — NOT `VALUE_NONE`; the
    /// search normalizes via the `hit` flag (doc 08 §3.5).
    pub const EMPTY: TtData = TtData {
        mv: Move::NULL,
        value: VALUE_NONE,
        eval: VALUE_NONE,
        depth: DEPTH_NONE,
        bound: Bound::None,
        is_pv: false,
    };
}

/// Write handle to the entry a probe decided on (the tag match, or the
/// replacement victim). Deliberately non-`Clone` and short-lived: it pins a
/// probe decision that goes stale after deeper re-searches at the same node
/// re-probe (doc 08 §6).
pub struct TtWriter<'tt> {
    entry: &'tt TtEntry,
}

impl TtWriter<'_> {
    /// Store a search result (or eval-only entry) under the doc 08 §3.7
    /// replacement rules. `value` must already be ply-converted with
    /// `value_to_tt`; `eval` must be the RAW uncorrected static eval;
    /// `generation` is the caller's cached `tt.generation()`.
    // The 8-parameter shape is the doc 08 §3.7 contract (mirrors SF's save).
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &self,
        key: Key,
        value: Value,
        is_pv: bool,
        bound: Bound,
        depth: Depth,
        mv: Move,
        eval: Value,
        generation: u8,
    ) {
        self.entry
            .save(key, value, is_pv, bound, depth, mv, eval, generation);
    }

    /// Demote an entry that keeps failing the caller (doc 08 §3.8): erode
    /// its stored depth, clamped at zero — a fully penalized entry becomes
    /// unoccupied and is the first replacement victim. The single call site
    /// (doc 07) uses `penalize(1)`.
    pub fn penalize(&self, penalty: i32) {
        debug_assert!(penalty >= 0);
        #[cfg(feature = "ttcensus42")]
        crate::ttcensus42::bump(&crate::ttcensus42::PENALIZE);
        let d = i32::from(self.entry.depth8.load(Relaxed));
        self.entry.depth8.store((d - penalty).max(0) as u8, Relaxed);
    }
}

/// Result of [`TranspositionTable::probe`]: the decoded data (garbage
/// unless `hit`) plus the entry to write back to.
pub struct Probe<'tt> {
    pub hit: bool,
    pub data: TtData,
    pub writer: TtWriter<'tt>,
}

// ---------------------------------------------------------------------------
// TranspositionTable (doc 08 §3.4, §3.9–§3.10, §3.13–§3.14)
// ---------------------------------------------------------------------------

/// Maps a uniform 64-bit key onto `[0, n)` for ANY `n` — no power-of-two
/// restriction, no modulo: the product is at most `(2^64 − 1)·n`, so its
/// high 64 bits are strictly less than `n`. One widening `mul` versus a
/// ~25-cycle division (doc 08 §3.4).
#[inline]
fn mul_hi64(a: u64, b: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) >> 64) as u64
}

/// The transposition table. Shared across workers as `Arc<TT>` (D23):
/// `probe`/`write`/`new_search` take `&self` (interior mutability via the
/// relaxed atomics); `resize`/`clear` take `&mut self`, obtained on the UCI
/// thread between searches via `Arc::get_mut` (doc 13 §3.6).
pub struct TranspositionTable {
    clusters: Vec<Cluster>, // later: huge-page-backed buffer (doc 08 §3.13 phase 2)
    /// Current age, `0..32`; interior-mutable so [`Self::new_search`] works
    /// through the shared handle (D23).
    generation8: AtomicU8,
}

impl TranspositionTable {
    /// A table of `mb` MiB (clamped and fallible per [`Self::resize`]).
    pub fn new(mb: usize) -> Self {
        let mut tt = TranspositionTable {
            clusters: Vec::new(),
            generation8: AtomicU8::new(0),
        };
        tt.resize(mb);
        tt
    }

    /// Test constructor: `n` clusters so tests control the cluster/tag
    /// split precisely (doc 08 §5 uses `with_cluster_count(1)`).
    #[cfg(test)]
    pub(crate) fn with_cluster_count(n: usize) -> Self {
        let mut clusters = Vec::new();
        clusters.resize_with(n, Cluster::zeroed);
        TranspositionTable {
            clusters,
            generation8: AtomicU8::new(0),
        }
    }

    /// Test observer: allocated table size in bytes (uci wiring tests
    /// assert `setoption Hash` really resized).
    #[cfg(test)]
    pub(crate) fn size_bytes(&self) -> usize {
        self.clusters.len() * size_of::<Cluster>()
    }

    #[inline]
    fn cluster_for(&self, key: Key) -> &Cluster {
        // Safe indexing: mul_hi64(key, len) < len by construction. Keep the
        // bounds check until a profile says otherwise (D14c).
        let idx = mul_hi64(key, self.clusters.len() as u64) as usize;
        &self.clusters[idx]
    }

    /// Look up `key` (doc 08 §3.5): scan the cluster's 3 entries in order
    /// and return the FIRST tag match — even an unoccupied one, so the same
    /// position writes back into its own slot instead of evicting a
    /// neighbor (`hit == false` there; ignore the data). With no tag match,
    /// the writer points at the replacement victim: the entry minimizing
    /// `depth8 − 8·relative_age`, first strict minimum wins (§3.6) — one
    /// generation of staleness outweighs 8 plies of stored depth, and
    /// empties (score 0) are consumed before current-generation entries.
    pub fn probe(&self, key: Key) -> Probe<'_> {
        let cluster = self.cluster_for(key);
        let tag = key as u16;

        #[cfg(feature = "ttcensus42")]
        {
            use crate::ttcensus42 as c;
            c::bump(&c::PROBES);
            c::note_cluster(std::ptr::from_ref(cluster) as usize);
        }

        for (_slot, entry) in cluster.entries.iter().enumerate() {
            if entry.key16.load(Relaxed) == tag {
                #[cfg(feature = "ttcensus42")]
                {
                    use crate::ttcensus42 as c;
                    c::bump(&c::TAG_MATCH);
                    c::bump([&c::SLOT0, &c::SLOT1, &c::SLOT2][_slot]);
                    if entry.is_occupied() {
                        c::bump(&c::HITS);
                    }
                }
                return Probe {
                    hit: entry.is_occupied(),
                    data: entry.read(),
                    writer: TtWriter { entry },
                };
            }
        }

        let generation = self.generation();
        let replace_score = |entry: &TtEntry| {
            TT_REPLACE_DEPTH_WEIGHT() * i32::from(entry.depth8.load(Relaxed))
                - TT_REPLACE_AGE_WEIGHT() * i32::from(entry.relative_age(generation))
        };
        let mut victim = &cluster.entries[0];
        let mut victim_score = replace_score(victim);
        for entry in &cluster.entries[1..] {
            let score = replace_score(entry);
            if score < victim_score {
                victim = entry;
                victim_score = score;
            }
        }
        Probe {
            hit: false,
            data: TtData::EMPTY,
            writer: TtWriter { entry: victim },
        }
    }

    /// Advance the 32-cycle age. Call once at the start of every root
    /// search (`go`). `&self` + relaxed atomic per D23: the main worker
    /// bumps it through the shared `Arc<TT>` while helpers are still
    /// parked, so the non-atomic load+store pair is single-writer.
    pub fn new_search(&self) {
        let g = (self.generation8.load(Relaxed) + 1) & GENERATION_MASK;
        self.generation8.store(g, Relaxed);
    }

    /// The current age, `0..32`. Search reads it once per node and passes
    /// it into [`TtWriter::write`] (doc 08 §3.9).
    pub fn generation(&self) -> u8 {
        self.generation8.load(Relaxed)
    }

    /// Resize to `mb` MiB, clamped to `MIN_HASH_MB..=MAX_HASH_MB`, and
    /// clear. Returns the MiB actually allocated: on allocation failure it
    /// falls back through halved sizes, always leaving a usable table —
    /// never aborting on a mid-session `setoption` (doc 08 §3.13; only the
    /// startup default allocation may be fatal, doc 01 §3.4). The UCI layer
    /// compares the return value against the request and reports shortfall.
    pub fn resize(&mut self, mb: usize) -> usize {
        let mb = mb.clamp(MIN_HASH_MB, MAX_HASH_MB);
        // Drop the old buffer first or peak RSS doubles (doc 08 §6).
        self.clusters = Vec::new();
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
            // If even the 1 MiB floor fails, the process is out of memory
            // for good and dies either way (doc 08 §3.13).
            assert!(try_mb > MIN_HASH_MB, "cannot allocate even a 1 MiB TT");
            try_mb /= 2; // integer halving reaches exactly 1 MiB
        }
    }

    /// Zero every entry and reset the generation (UCI `Clear Hash` /
    /// `ucinewgame`). `&mut self` gives exclusive access, so `get_mut`
    /// yields plain integers — no atomic ops, and the optimizer collapses
    /// the loop to a memset (doc 08 §3.14). Multithreaded clear for huge
    /// tables comes with the thread pool (plan step 9).
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

    /// Permille of entries written during the current search (age <=
    /// `max_age`), sampled over the first 1000 clusters — a 32 KB scan,
    /// only call at UCI-info cadence. UCI `info hashfull` uses
    /// `max_age == 0` (doc 08 §3.10).
    pub fn hashfull(&self, max_age: u8) -> u32 {
        debug_assert!(self.clusters.len() >= HASHFULL_SAMPLE_CLUSTERS);
        let generation = self.generation();
        let mut count: u32 = 0;
        for cluster in &self.clusters[..HASHFULL_SAMPLE_CLUSTERS] {
            for e in &cluster.entries {
                count += u32::from(e.is_occupied() && e.relative_age(generation) <= max_age);
            }
        }
        count / CLUSTER_SIZE as u32
    }

    /// Hint the cache to pull the probe target for `key` (doc 08 §3.15).
    /// No-op off x86_64 (an aarch64 `prfm` variant can come later).
    ///
    /// Call sites are owned by docs 04/07: inside `make_move` as soon as
    /// the child key is known, and speculatively in the search's move loop
    /// via the cheap predicted key — a wrong key just prefetches a useless
    /// line, harmlessly.
    ///
    /// D14(a) measurement (2026-07-10, WSL2 x86-64, release, 256 MB table,
    /// 1M uniform-random probes, `prefetch_microbenchmark`): cold probes
    /// stream at ~23 ns/probe (throughput — out-of-order overlap already
    /// hides part of each DRAM miss); with `prefetch` issued 8 keys ahead
    /// the same loop runs at ~11 ns/probe, a 2x gain, matching doc 14
    /// §3.4's "mostly hidden by prefetch" target. The search-integrated
    /// bench-nps delta is re-measured when the call sites land (doc 08 §4
    /// step 7).
    #[inline]
    pub fn prefetch(&self, key: Key) {
        #[cfg(feature = "ttcensus42")]
        crate::ttcensus42::bump(&crate::ttcensus42::PREFETCH);
        #[cfg(all(target_arch = "x86_64", not(feature = "ttnopf42")))]
        {
            // Perf study 42 arms. `ttnopf42` removes the hint entirely (the
            // subtractive arm that prices it); `ttpft242` demotes it from L1
            // to L2/L3. Both are node-identical: a cache hint changes no
            // search result, so the `Bench:` fingerprint is unmoved.
            #[cfg(not(feature = "ttpft242"))]
            use std::arch::x86_64::_MM_HINT_T0 as HINT;
            #[cfg(feature = "ttpft242")]
            use std::arch::x86_64::_MM_HINT_T2 as HINT;
            use std::arch::x86_64::_mm_prefetch;
            let p: *const Cluster = self.cluster_for(key);
            // SAFETY: `_mm_prefetch` is a pure cache hint: it performs no
            // memory access observable by the abstract machine, writes
            // nothing, and cannot fault even on an invalid address (the
            // hardware drops bad hints). The pointer additionally derives
            // from a live `&Cluster` into our own allocation, so it is
            // valid for the cache line being hinted.
            unsafe { _mm_prefetch(p.cast::<i8>(), HINT) };
        }
        #[cfg(any(not(target_arch = "x86_64"), feature = "ttnopf42"))]
        let _ = key;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attacks::Prng;
    use crate::types::{DEPTH_QS, DEPTH_UNSEARCHED, PieceType, Square, VALUE_MATE};

    /// Shorthand: probe `key` and write through the returned writer with
    /// the table's current generation.
    fn store(
        tt: &TranspositionTable,
        key: Key,
        value: Value,
        pv: bool,
        bound: Bound,
        depth: Depth,
        mv: Move,
    ) {
        tt.probe(key)
            .writer
            .write(key, value, pv, bound, depth, mv, 0, tt.generation());
    }

    // --- §5 roundtrip -----------------------------------------------------

    #[test]
    fn write_probe_roundtrip_grid() {
        // Negative values, band edges, VALUE_NONE; all bounds; both pv
        // flags; every move kind; eval-only and deep depths.
        let values = [
            -VALUE_INFINITE + 1,
            -VALUE_MATE,
            -1,
            0,
            1,
            VALUE_MATE - 10,
            VALUE_MATE,
            VALUE_NONE,
        ];
        let evals = [-1234, 0, 777, VALUE_NONE];
        let depths = [DEPTH_UNSEARCHED, DEPTH_QS, 1, 5, 100, 240];
        let bounds = [Bound::None, Bound::Upper, Bound::Lower, Bound::Exact];
        let moves = [
            Move::NULL,
            Move::normal(Square::E2, Square::E4),
            Move::promotion(Square::E7, Square::E8, PieceType::Queen),
            Move::en_passant(Square::D5, Square::E6),
            Move::castling(Square::E1, Square::G1),
        ];

        let tt = TranspositionTable::with_cluster_count(1);
        let mut key: Key = 0;
        for &v in &values {
            for &ev in &evals {
                for &d in &depths {
                    for &b in &bounds {
                        for pv in [false, true] {
                            for &m in &moves {
                                key += 1; // distinct nonzero tags (< 65536 combos)
                                tt.probe(key).writer.write(
                                    key,
                                    v,
                                    pv,
                                    b,
                                    d,
                                    m,
                                    ev,
                                    tt.generation(),
                                );
                                let probe = tt.probe(key);
                                assert!(probe.hit);
                                assert_eq!(probe.data.mv, m);
                                assert_eq!(probe.data.value, v);
                                assert_eq!(probe.data.eval, ev);
                                assert_eq!(probe.data.depth, d);
                                assert_eq!(probe.data.bound, b);
                                assert_eq!(probe.data.is_pv, pv);
                            }
                        }
                    }
                }
            }
        }
    }

    // --- §5 occupancy and tag behavior -------------------------------------

    #[test]
    fn fresh_table_misses() {
        let tt = TranspositionTable::with_cluster_count(1);
        let probe = tt.probe(0xABCD);
        assert!(!probe.hit);
        assert_eq!(probe.data.mv, Move::NULL);
        assert_eq!(probe.data.value, VALUE_NONE);
        assert_eq!(probe.data.eval, VALUE_NONE);
        assert_eq!(probe.data.depth, DEPTH_NONE);
        assert_eq!(probe.data.bound, Bound::None);
        assert!(!probe.data.is_pv);
    }

    #[test]
    fn zero_tag_matches_empty_slot_and_writes_back_into_it() {
        // A key whose low 16 bits are 0 tag-matches the zeroed entry 0:
        // hit == false (unoccupied), and the writer targets that slot —
        // the same position writes back instead of evicting a neighbor.
        // Note the tag-match decode is depth DEPTH_NONE / value 0 (NOT
        // VALUE_NONE); the hit flag is what protects consumers.
        let tt = TranspositionTable::with_cluster_count(1);
        let key: Key = 0x7_0000;
        let probe = tt.probe(key);
        assert!(!probe.hit);
        assert_eq!(probe.data.depth, DEPTH_NONE);
        assert_eq!(probe.data.value, 0);
        assert!(std::ptr::eq(probe.writer.entry, &tt.clusters[0].entries[0]));
        probe.writer.write(
            key,
            42,
            false,
            Bound::Exact,
            7,
            Move::normal(Square::E2, Square::E4),
            13,
            tt.generation(),
        );
        let again = tt.probe(key);
        assert!(again.hit);
        assert_eq!(again.data.value, 42);
        assert_eq!(again.data.depth, 7);
    }

    #[test]
    fn tag_collision_returns_foreign_payload() {
        // Documents the ACCEPTED defect (doc 08 §3.5): two distinct keys
        // with equal low-16 bits in the same cluster alias each other, so
        // probing B after writing A reports a hit with A's payload. This is
        // the contract that forces consumers to validate everything they
        // read (pseudo_legal + legal on the move, is_valid on values).
        let tt = TranspositionTable::with_cluster_count(1);
        let a: Key = 0x0001_0000_0000_1234;
        let b: Key = 0x0002_0000_0000_1234;
        assert_ne!(a, b);
        store(&tt, a, 321, false, Bound::Lower, 9, Move::NULL);
        let probe = tt.probe(b);
        assert!(probe.hit);
        assert_eq!(probe.data.value, 321);
        assert_eq!(probe.data.depth, 9);
    }

    // --- §5 replacement victim ---------------------------------------------

    #[test]
    fn empty_cluster_victim_is_first_entry() {
        // All three candidates score 0; ties keep the earlier entry.
        let tt = TranspositionTable::with_cluster_count(1);
        let probe = tt.probe(0x1111);
        assert!(!probe.hit);
        assert!(std::ptr::eq(probe.writer.entry, &tt.clusters[0].entries[0]));
    }

    #[test]
    fn victim_minimizes_depth_minus_age_weight() {
        // Doc 08 §5: a 2-generations-old depth-20 entry loses to current
        // shallow entries: internal 23 − 8·2 = 7 < 8 (depth 5, age 0)
        // < 11 (depth 8, age 0).
        let tt = TranspositionTable::with_cluster_count(1);
        tt.new_search(); // generation 1
        let a: Key = 0xA1;
        store(&tt, a, 100, false, Bound::Lower, 20, Move::NULL); // slot 0
        tt.new_search();
        tt.new_search(); // generation 3: A is now 2 generations old
        let b: Key = 0xB2;
        let c: Key = 0xC3;
        store(&tt, b, 100, false, Bound::Lower, 5, Move::NULL); // slot 1
        store(&tt, c, 100, false, Bound::Lower, 8, Move::NULL); // slot 2

        let d: Key = 0xD4;
        let probe = tt.probe(d);
        assert!(!probe.hit);
        probe
            .writer
            .write(d, 7, false, Bound::Exact, 3, Move::NULL, 0, tt.generation());
        assert!(!tt.probe(a).hit, "the aged deep entry was the victim");
        assert!(tt.probe(b).hit);
        assert!(tt.probe(c).hit);
        assert!(tt.probe(d).hit);
    }

    // --- §5 save rules, one test each ---------------------------------------

    #[test]
    fn exact_overwrites_deeper_same_key_entry() {
        let tt = TranspositionTable::with_cluster_count(1);
        let k: Key = 0x2222;
        store(&tt, k, 100, false, Bound::Lower, 20, Move::NULL);
        // Shallower, same key, same generation — refresh rule alone would
        // reject (8 <= 23 − 4), but EXACT always wins.
        store(&tt, k, 300, false, Bound::Exact, 5, Move::NULL);
        let data = tt.probe(k).data;
        assert_eq!(data.depth, 5);
        assert_eq!(data.value, 300);
        assert_eq!(data.bound, Bound::Exact);
    }

    #[test]
    fn key_change_overwrites_regardless_of_depth() {
        // Fill all three slots with deep current-generation entries, then
        // write a new position at minimal depth: the victim is overwritten
        // even though every stored entry is far deeper.
        let tt = TranspositionTable::with_cluster_count(1);
        for (i, tag) in [0xA1u64, 0xB2, 0xC3].into_iter().enumerate() {
            store(
                &tt,
                tag,
                100 + i as Value,
                false,
                Bound::Lower,
                20,
                Move::NULL,
            );
        }
        let k: Key = 0xD4;
        store(&tt, k, 55, false, Bound::Upper, 1, Move::NULL);
        let probe = tt.probe(k);
        assert!(probe.hit);
        assert_eq!(probe.data.depth, 1);
        assert_eq!(probe.data.value, 55);
    }

    #[test]
    fn refresh_boundary_non_pv() {
        // Same key/generation, non-exact: new_d + 2·pv > old_d − 4 strict.
        // Stored external depth 12 → old_d 15. new_d == 11 (d == 8) is
        // rejected; new_d == 12 (d == 9) is accepted. Value 100 is
        // non-decisive so rule 3 leaves rejected depth untouched.
        let tt = TranspositionTable::with_cluster_count(1);
        let k: Key = 0x3333;
        store(&tt, k, 100, false, Bound::Lower, 12, Move::NULL);
        store(&tt, k, 555, false, Bound::Lower, 8, Move::NULL);
        assert_eq!(tt.probe(k).data.depth, 12, "boundary write rejected");
        assert_eq!(tt.probe(k).data.value, 100);
        store(&tt, k, 555, false, Bound::Lower, 9, Move::NULL);
        assert_eq!(tt.probe(k).data.depth, 9, "one past the boundary accepted");
        assert_eq!(tt.probe(k).data.value, 555);
    }

    #[test]
    fn refresh_boundary_pv_credit() {
        // PV writes get +2: against stored external depth 12 (old_d 15),
        // a pv write at d == 6 (new_d 9 + 2 == 11) is rejected and d == 7
        // (new_d 10 + 2 == 12 > 11) is accepted.
        let tt = TranspositionTable::with_cluster_count(1);
        let k: Key = 0x4444;
        store(&tt, k, 100, false, Bound::Lower, 12, Move::NULL);
        store(&tt, k, 555, true, Bound::Lower, 6, Move::NULL);
        assert_eq!(tt.probe(k).data.depth, 12, "pv boundary write rejected");
        store(&tt, k, 555, true, Bound::Lower, 7, Move::NULL);
        assert_eq!(tt.probe(k).data.depth, 7, "pv one past boundary accepted");
        assert!(tt.probe(k).data.is_pv);
    }

    #[test]
    fn rejected_write_still_updates_the_move() {
        // Rule 1 runs before rule 2: a rejected same-key write carrying a
        // real move still refreshes move16 (SF parity).
        let tt = TranspositionTable::with_cluster_count(1);
        let k: Key = 0x5555;
        let old = Move::normal(Square::E2, Square::E4);
        let new = Move::normal(Square::G1, Square::F3);
        store(&tt, k, 100, false, Bound::Lower, 12, old);
        store(&tt, k, 555, false, Bound::Lower, 3, new); // rejected by rule 2
        let data = tt.probe(k).data;
        assert_eq!(data.depth, 12, "write body was rejected");
        assert_eq!(data.value, 100);
        assert_eq!(data.mv, new, "but the move was refreshed");
    }

    #[test]
    fn older_generation_overwritten_by_shallow_data() {
        let tt = TranspositionTable::with_cluster_count(1);
        let k: Key = 0x6666;
        store(&tt, k, 100, false, Bound::Lower, 30, Move::NULL);
        tt.new_search();
        // Same key, non-exact, way shallower (5 <= 33 − 4) — accepted
        // purely because the entry is from an older search.
        store(&tt, k, 200, false, Bound::Upper, 2, Move::NULL);
        let data = tt.probe(k).data;
        assert_eq!(data.depth, 2);
        assert_eq!(data.value, 200);
        assert_eq!(data.bound, Bound::Upper);
    }

    #[test]
    fn ttmove_preserved_on_same_key_null_write() {
        let tt = TranspositionTable::with_cluster_count(1);
        let k: Key = 0x7777;
        let m = Move::normal(Square::E2, Square::E4);
        store(&tt, k, 100, false, Bound::Lower, 10, m);
        store(&tt, k, 200, false, Bound::Lower, 12, Move::NULL);
        let data = tt.probe(k).data;
        assert_eq!(data.mv, m, "old move kept when the new write brings none");
        assert_eq!(data.value, 200);
        assert_eq!(data.depth, 12);
    }

    #[test]
    fn ttmove_cleared_on_key_change_null_write() {
        // Make the move-carrying entry the replacement victim (lowest
        // depth), then evict it with a different key and no move: the
        // stale move must not leak into the new position's entry.
        let tt = TranspositionTable::with_cluster_count(1);
        let a: Key = 0xA1;
        store(
            &tt,
            a,
            100,
            false,
            Bound::Lower,
            4,
            Move::normal(Square::E2, Square::E4),
        );
        store(&tt, 0xB2, 100, false, Bound::Lower, 20, Move::NULL);
        store(&tt, 0xC3, 100, false, Bound::Lower, 20, Move::NULL);
        let d: Key = 0xD4;
        store(&tt, d, 200, false, Bound::Upper, 6, Move::NULL);
        assert!(!tt.probe(a).hit, "the shallow entry was the victim");
        let data = tt.probe(d).data;
        assert!(tt.probe(d).hit);
        assert_eq!(data.mv, Move::NULL, "foreign move cleared");
    }

    #[test]
    fn secondary_aging_erodes_decisive_non_exact_entries() {
        // A rejected write against a deep, non-EXACT, decisive entry
        // erodes exactly 1 ply of stored depth per rejection; the payload
        // stays. Erosion stops once external depth drops below 5 (rule 3's
        // threshold), leaving the stale mate claim easy to replace.
        let tt = TranspositionTable::with_cluster_count(1);
        let k: Key = 0x8888;
        store(&tt, k, VALUE_MATE - 10, false, Bound::Lower, 10, Move::NULL);
        // Eval-only-shaped rejected write: new_d == 1, never accepted while
        // old_d − 4 >= 1.
        store(
            &tt,
            k,
            50,
            false,
            Bound::Lower,
            DEPTH_UNSEARCHED,
            Move::NULL,
        );
        let data = tt.probe(k).data;
        assert_eq!(data.depth, 9, "eroded by exactly 1");
        assert_eq!(data.value, VALUE_MATE - 10, "payload untouched");
        assert_eq!(data.bound, Bound::Lower);
        // March it down: erosion continues only while external depth >= 5.
        for _ in 0..20 {
            store(
                &tt,
                k,
                50,
                false,
                Bound::Lower,
                DEPTH_UNSEARCHED,
                Move::NULL,
            );
        }
        assert_eq!(tt.probe(k).data.depth, 4, "erosion floor: ext depth 5 − 1");
        assert!(tt.probe(k).hit, "still occupied at the floor");
    }

    #[test]
    fn secondary_aging_skips_non_decisive_and_exact_entries() {
        let tt = TranspositionTable::with_cluster_count(1);
        // Non-decisive value: rejected writes leave depth alone.
        let k1: Key = 0x9999;
        store(&tt, k1, 100, false, Bound::Lower, 10, Move::NULL);
        store(&tt, k1, 50, false, Bound::Lower, 3, Move::NULL);
        assert_eq!(tt.probe(k1).data.depth, 10);
        // EXACT stored bound: decisive value, but rejected writes leave
        // depth alone.
        let k2: Key = 0xAAAA;
        store(
            &tt,
            k2,
            VALUE_MATE - 10,
            false,
            Bound::Exact,
            10,
            Move::NULL,
        );
        store(&tt, k2, 50, false, Bound::Lower, 3, Move::NULL);
        assert_eq!(tt.probe(k2).data.depth, 10);
    }

    #[test]
    fn penalize_decrements_and_clamps_to_unoccupied() {
        let tt = TranspositionTable::with_cluster_count(1);
        let k: Key = 0xBBBB;
        store(&tt, k, 100, false, Bound::Lower, 10, Move::NULL);
        tt.probe(k).writer.penalize(1);
        let probe = tt.probe(k);
        assert!(probe.hit);
        assert_eq!(probe.data.depth, 9);
        tt.probe(k).writer.penalize(255);
        let probe = tt.probe(k);
        assert!(!probe.hit, "fully penalized entry reads as unoccupied");
        // ... but the tag still matches, so the same position writes back
        // into its own slot.
        assert!(std::ptr::eq(probe.writer.entry, &tt.clusters[0].entries[0]));
    }

    // --- §5 aging wraparound -------------------------------------------------

    #[test]
    fn relative_age_wraps_for_all_generation_pairs() {
        // All 32×32 (current, entry) pairs behave like a 32-hour clock, and
        // the bound/pv bits (all 8 combos) never perturb the age.
        let cluster = Cluster::zeroed();
        let entry = &cluster.entries[0];
        for cur in 0u8..32 {
            for ent in 0u8..32 {
                for bits in 0u8..8 {
                    let bound = bits & 0b11;
                    let pv = bits >> 2;
                    entry
                        .gen_bound8
                        .store(ent | bound << BOUND_SHIFT | pv << PV_SHIFT, Relaxed);
                    assert_eq!(entry.relative_age(cur), (32 + cur - ent) % 32);
                }
            }
        }
    }

    #[test]
    fn new_search_cycles_through_32_generations() {
        let tt = TranspositionTable::with_cluster_count(1);
        assert_eq!(tt.generation(), 0);
        tt.new_search();
        assert_eq!(tt.generation(), 1);
        for _ in 1..32 {
            tt.new_search();
        }
        assert_eq!(tt.generation(), 0, "wraps after 32 searches");
    }

    // --- §5 indexing -----------------------------------------------------------

    #[test]
    fn mul_hi64_in_range_for_adversarial_inputs() {
        let keys = [
            0u64,
            1,
            2,
            u64::MAX,
            u64::MAX - 1,
            1 << 63,
            (1 << 63) + 1,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
        ];
        let counts = [1u64, 3, 1000, 32768, 65537, 1_000_003];
        for &k in &keys {
            for &n in &counts {
                assert!(mul_hi64(k, n) < n, "mul_hi64({k:#x}, {n}) out of range");
            }
        }
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "1e6 PRNG iterations of pure integer math — minutes under the \
                  interpreter, and nothing for miri to check (no atomics, no unsafe)"
    )]
    fn mul_hi64_distribution_smoke() {
        // 1e6 random keys over 1000 clusters: loose bound, no cluster gets
        // more than 3× the mean (doc 08 §5).
        let mut rng = Prng::new(0x9E37_79B9_7F4A_7C15);
        const N: usize = 1000;
        const SAMPLES: u32 = 1_000_000;
        let mut counts = [0u32; N];
        for _ in 0..SAMPLES {
            counts[mul_hi64(rng.rand64(), N as u64) as usize] += 1;
        }
        let mean = SAMPLES / N as u32;
        for (i, &c) in counts.iter().enumerate() {
            assert!(c < 3 * mean, "cluster {i} got {c} of {SAMPLES} keys");
        }
    }

    // --- §5 hashfull, resize, clear ---------------------------------------------

    /// A key that lands in cluster `i` of an `n`-cluster table, with its
    /// low 16 bits (the tag) forced to `tag`. Verified against `mul_hi64`
    /// by the caller's writes hitting the intended clusters.
    fn key_for_cluster(i: u64, n: u64, tag: u16) -> Key {
        let base = ((u128::from(i) << 64) / u128::from(n)) as u64 + (1 << 32);
        let key = (base & !0xFFFF) | u64::from(tag);
        assert_eq!(mul_hi64(key, n), i);
        key
    }

    /// Every atomics/`get_mut` path at a miri-tractable scale (1000
    /// clusters — `hashfull`'s minimum sample). The MiB-scale siblings
    /// below are `cfg_attr(miri, ignore)`d: constructing 32768+ clusters
    /// takes minutes under the interpreter's borrow tracking, and the
    /// aliasing/atomics surface it would check is identical here.
    #[test]
    fn small_table_lifecycle_exercises_all_paths() {
        let mut tt = TranspositionTable::with_cluster_count(1000);
        tt.new_search();
        assert_eq!(tt.generation(), 1);
        let key: Key = 0xABCD_EF01_2345_6789;
        tt.prefetch(key);
        let probe = tt.probe(key);
        assert!(!probe.hit);
        probe.writer.write(
            key,
            42,
            true,
            Bound::Exact,
            9,
            Move::normal(Square::E2, Square::E4),
            7,
            tt.generation(),
        );
        {
            let probe = tt.probe(key);
            assert!(probe.hit);
            assert_eq!(probe.data.value, 42);
            assert_eq!(probe.data.eval, 7);
            assert_eq!(tt.hashfull(0), 0, "1 of 3000 sampled rounds to 0");
            probe.writer.penalize(1);
        }
        assert_eq!(tt.probe(key).data.depth, 8);
        tt.clear();
        assert_eq!(tt.generation(), 0);
        assert!(!tt.probe(key).hit);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "MiB-scale table; covered by small_table_lifecycle_exercises_all_paths"
    )]
    fn hashfull_counts_current_generation_permille() {
        let mut tt = TranspositionTable::new(MIN_HASH_MB);
        let n = tt.clusters.len() as u64;
        assert_eq!(tt.hashfull(0), 0);
        // Fill exactly the first 1000 clusters, 3 entries each.
        for i in 0..1000 {
            for tag in 1..=3u16 {
                let k = key_for_cluster(i, n, tag);
                store(&tt, k, 100, false, Bound::Lower, 5, Move::NULL);
            }
        }
        assert_eq!(tt.hashfull(0), 1000);
        tt.new_search();
        assert_eq!(tt.hashfull(0), 0, "previous-generation entries don't count");
        assert_eq!(tt.hashfull(1), 1000, "unless the caller widens max_age");
        // Penalized-to-zero entries stop counting entirely.
        tt.clear();
        for tag in 1..=3u16 {
            let k = key_for_cluster(0, n, tag);
            store(&tt, k, 100, false, Bound::Lower, 5, Move::NULL);
        }
        assert_eq!(tt.hashfull(0), 1);
        tt.probe(key_for_cluster(0, n, 1)).writer.penalize(255);
        assert_eq!(tt.hashfull(0), 0, "2 of 3000 entries rounds to 0 permille");
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "MiB-scale table; covered by small_table_lifecycle_exercises_all_paths"
    )]
    fn resize_clamps_and_uses_all_requested_memory() {
        // resize(0) clamps to the 1 MiB floor; cluster count is exactly
        // MB·2^20/32 with no power-of-two rounding (doc 08 §3.4).
        let mut tt = TranspositionTable::new(0);
        assert_eq!(tt.clusters.len() * size_of::<Cluster>(), 1024 * 1024);
        let got = tt.resize(3);
        assert_eq!(got, 3);
        assert_eq!(tt.clusters.len() * size_of::<Cluster>(), 3 * 1024 * 1024);
        // Resize implies clear: fresh generation, empty table, usable.
        tt.new_search();
        let k: Key = 0xCCCC;
        store(&tt, k, 100, false, Bound::Lower, 5, Move::NULL);
        let got = tt.resize(1);
        assert_eq!(got, 1);
        assert_eq!(tt.generation(), 0);
        assert!(!tt.probe(k).hit);
    }

    #[test]
    #[ignore = "allocates 1 GiB; run manually in release: \
                cargo test --release -- --ignored resize_1024_mb"]
    fn resize_1024_mb_uses_all_requested_memory() {
        // Doc 08 §8 criterion 5: resize honors sizes up to at least
        // 1024 MB, cluster count exactly MB*2^20/32 (no power-of-two
        // rounding), and the far end of the table is usable.
        let mut tt = TranspositionTable::new(MIN_HASH_MB);
        let got = tt.resize(1024);
        assert_eq!(got, 1024);
        assert_eq!(tt.clusters.len(), 1024 * 1024 * 1024 / size_of::<Cluster>());
        let n = tt.clusters.len() as u64;
        let k = key_for_cluster(n - 1, n, 0x7777);
        store(&tt, k, 99, false, Bound::Lower, 9, Move::NULL);
        assert!(tt.probe(k).hit);
        assert_eq!(tt.hashfull(0), 0, "1 entry in 3000 sampled rounds to 0");
    }

    #[test]
    #[ignore = "commits multiple GiB on overcommit-happy hosts (doc 08 §3.13 \
                residual risk); run manually in release under a virtual-memory \
                cap so the halving fallback bottoms out early and cheaply: \
                cargo test --release --no-run, then \
                bash -c 'ulimit -v 2097152; <test-binary> --ignored resize_absurd'"]
    fn resize_absurd_request_falls_back_without_abort() {
        // Doc 08 §5 allocation-failure contract / §8 criterion 5: an
        // unsatisfiable mid-session resize NEVER aborts the process — it
        // falls back through halved sizes, reports the achieved size, and
        // leaves a usable table.
        let mut tt = TranspositionTable::new(MIN_HASH_MB);
        let request = 8 * 1024 * 1024; // 8 TiB in MB, well under MAX_HASH_MB
        let got = tt.resize(request);
        assert!(
            got < request,
            "8 TiB reserve succeeded?! (vm.overcommit_memory=1 host)"
        );
        assert!(got >= MIN_HASH_MB);
        assert_eq!(tt.clusters.len(), got * 1024 * 1024 / size_of::<Cluster>());
        // Usable at whatever size was achieved: write/probe roundtrip.
        tt.new_search();
        let n = tt.clusters.len() as u64;
        let k = key_for_cluster(0, n, 0x4242);
        store(&tt, k, 123, false, Bound::Lower, 7, Move::NULL);
        assert!(tt.probe(k).hit);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "MiB-scale table; covered by small_table_lifecycle_exercises_all_paths"
    )]
    fn clear_zeroes_entries_and_generation() {
        let mut tt = TranspositionTable::new(MIN_HASH_MB);
        tt.new_search();
        let n = tt.clusters.len() as u64;
        let k = key_for_cluster(0, n, 0x42);
        store(
            &tt,
            k,
            100,
            true,
            Bound::Exact,
            12,
            Move::normal(Square::E2, Square::E4),
        );
        assert!(tt.probe(k).hit);
        tt.clear();
        assert_eq!(tt.generation(), 0);
        assert!(!tt.probe(k).hit);
        assert_eq!(tt.hashfull(GENERATION_MASK), 0);
    }

    // --- §3.15 prefetch, §5 concurrency smoke -------------------------------

    #[test]
    fn prefetch_is_callable_for_any_key() {
        // The hint must be harmless for every key (including keys computed
        // speculatively from wrong predictions) and on tiny tables. Also
        // the one path miri walks through the module's single unsafe block.
        let tt = TranspositionTable::with_cluster_count(1);
        for key in [0u64, 1, u64::MAX, 0xDEAD_BEEF] {
            tt.prefetch(key);
        }
        let tt = TranspositionTable::with_cluster_count(1000);
        tt.prefetch(0x1234_5678_9ABC_DEF0);
        assert!(
            !tt.probe(0x1234_5678_9ABC_DEF0).hit,
            "prefetch never writes"
        );
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "wall-clock thread hammer; doc 08 §5 excludes it under miri"
    )]
    fn concurrency_smoke_probe_write_penalize() {
        // Doc 08 §5 / §8 criterion 4: 4 threads hammer probe/write/penalize
        // on a 1 MB table with overlapping keys for ~1 s. Passes if nothing
        // panics and every decoded TtData is structurally sane. Asserts
        // crash-freedom, NOT consistency — mixed-writer reads are the
        // documented §3.5 contract, absorbed by consumer validation.
        use std::time::{Duration, Instant};
        let tt = TranspositionTable::new(MIN_HASH_MB);
        tt.new_search();
        // Every thread draws from the same 512-key set, so the same entries
        // are concurrently probed, overwritten, and penalized.
        let mut rng = Prng::new(0x5EED_CAFE_F00D_D00D);
        let keys: Vec<Key> = (0..512).map(|_| rng.rand64()).collect();
        let deadline = Instant::now() + Duration::from_millis(900);
        std::thread::scope(|scope| {
            for t in 1..=4u64 {
                let (tt, keys) = (&tt, &keys);
                scope.spawn(move || {
                    let mut rng = Prng::new(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(t));
                    let values = [0, 1, -1, 500, VALUE_MATE - 3, 3 - VALUE_MATE, VALUE_NONE];
                    let bounds = [Bound::None, Bound::Upper, Bound::Lower, Bound::Exact];
                    while Instant::now() < deadline {
                        for _ in 0..4096 {
                            let r = rng.rand64();
                            let key = keys[(r & 511) as usize];
                            tt.prefetch(key);
                            let probe = tt.probe(key);
                            // Structural sanity: depth decodes into
                            // DEPTH_NONE..=DEPTH_NONE+255 and the bound
                            // byte into a real Bound no matter how many
                            // writers the fields mix.
                            assert!(probe.data.depth >= DEPTH_NONE);
                            assert!(probe.data.depth <= DEPTH_NONE + 255);
                            let _: Bound = probe.data.bound;
                            if r >> 60 == 0 {
                                probe.writer.penalize(1);
                            } else {
                                probe.writer.write(
                                    key,
                                    values[(r >> 16) as usize % values.len()],
                                    r & (1 << 12) != 0,
                                    bounds[(r >> 24) as usize % bounds.len()],
                                    ((r >> 32) % 240) as Depth + 1, // 1..=240
                                    Move::from_raw16((r >> 40) as u16),
                                    values[(r >> 8) as usize % values.len()],
                                    tt.generation(),
                                );
                            }
                        }
                    }
                });
            }
        });
        // Still coherent for single-threaded use afterwards (EXACT write
        // is unconditionally accepted, rule 2a).
        let key = keys[0];
        tt.probe(key).writer.write(
            key,
            77,
            false,
            Bound::Exact,
            30,
            Move::NULL,
            5,
            tt.generation(),
        );
        let probe = tt.probe(key);
        assert!(probe.hit);
        assert_eq!(probe.data.value, 77);
    }

    #[test]
    #[ignore = "timing microbenchmark (D14a measurement); run manually: \
                cargo test --release -- --ignored prefetch_microbenchmark --nocapture"]
    fn prefetch_microbenchmark() {
        // The doc 14 §3.4 cold-probe shape: uniform-random probes over a
        // table far beyond LLC, without vs with `prefetch` issued DIST
        // keys ahead. Records the D14(a) measured justification quoted on
        // the `prefetch` doc comment; the search-integrated bench-nps
        // delta is re-measured at doc 08 §4 step 7.
        use std::hint::black_box;
        use std::time::Instant;
        let tt = TranspositionTable::new(256);
        const N: usize = 1 << 20;
        const DIST: usize = 8;
        let mut rng = Prng::new(0xC0FF_EE00_D15E_A5E5);
        let keys: Vec<Key> = (0..N + DIST).map(|_| rng.rand64()).collect();
        let run = |prefetch: bool| -> f64 {
            let mut sink = 0i64;
            let start = Instant::now();
            for i in 0..N {
                if prefetch {
                    tt.prefetch(keys[i + DIST]);
                }
                let probe = tt.probe(black_box(keys[i]));
                sink = sink.wrapping_add(i64::from(probe.data.depth));
            }
            black_box(sink);
            start.elapsed().as_secs_f64() * 1e9 / N as f64
        };
        // Interleave and keep the best of three, defusing turbo/paging
        // noise in either direction.
        let (mut cold, mut ahead) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..3 {
            cold = cold.min(run(false));
            ahead = ahead.min(run(true));
        }
        println!("cold probe:                 {cold:.1} ns/probe");
        println!("prefetch {DIST} keys ahead:   {ahead:.1} ns/probe");
        assert!(ahead < cold, "prefetch made cold probes slower");
    }
}
