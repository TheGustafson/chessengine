# 11 — Evaluation: PSQT to HCE to NNUE

Status: Draft v1

This document designs the evaluation subsystem behind the single
`evaluate(&Board, ...) -> Value` seam (D22), covering all three phases of the
roadmap (D21): Phase A material + tapered PSQT (enough to validate the search
machinery), Phase B a hand-crafted evaluation (HCE) with our own texel tuner,
and Phase C NNUE with incremental accumulators, quantized SIMD inference, and
a self-trained network. It also pins down exactly how search consumes the
static eval — the dead-draw material guard, optimism, complexity/material
blending, rule-50 damping, clamping, and the boundary with correction
history — because those formulas live at the eval/search seam and must not
drift as phases swap.

---

## 1. Purpose and scope

**This doc owns:**

- The `evaluate` function signature and semantics (side-to-move perspective,
  never called in check, raw vs. corrected eval boundary).
- The `eval` module tree (`src/eval/…`, per D13's "eval (later a dir)").
- Phase A: `Score {mg, eg}`, game phase, PeSTO PSQT tables, from-scratch
  computation, symmetry testing.
- Phase B: HCE term catalogue (mobility, passed pawns, king safety basics,
  bishop pair, rook on open file), the pawn-structure hash table keyed by
  `pawn_key` (D9), and the texel-tuning plan including a `tuner` bin target.
- Phase C: NNUE architecture, quantization, incremental accumulator design
  (dirty-piece driven), SIMD strategy, net file format, and the training
  pipeline (external `bullet` trainer + our `datagen` bin, per D15).
- The eval-side half of the search-consumption formulas (optimism blend,
  material scaling, rule-50 damping, TB-range clamping).

**This doc does NOT own** (it only defines the contract with):

- `make_move`/`unmake_move`, `StateInfo`, `non_pawn_material`, and the
  `MakeMoveHooks` seam on `make_move_with` — owned by the make/unmake doc
  (`design/04-make-unmake.md` §3.10; doc numbers per the suite index). This
  doc owns only the `DirtyPiece` payload attached through that seam (3.4.4);
  nothing NNUE-related lives inside `make_move` or `StateInfo`.
- The five Zobrist keys including `pawn_key` (D9) — owned by the Zobrist doc.
- Correction history tables, their update rule, and where the corrected eval
  is used — owned by the search doc. This doc only fixes the boundary:
  **correction is applied in search, never inside `evaluate`; the TT stores
  the raw eval** (D22).
- Optimism *computation at aspiration setup* — owned by the search doc; this
  doc owns how `evaluate` consumes the passed-in optimism.
- The TT `eval16` field — owned by the TT doc (D19).
- Per-thread `Worker` ownership of `EvalContext` — owned by the threading doc
  (D23).
- The `bench` harness — owned by the bench/UCI doc (D24); this doc specifies
  what eval contributes to it.

## 2. References

- `stockfish_reference/stockfish_search_notes.md` §10 (evaluate, correction
  history, optimism — all constants), §3.1 (optimism at aspiration setup).
- `stockfish_reference/Stockfish/src/evaluate.cpp` — the ~60-line outer
  `evaluate` shape: NNUE output → complexity/optimism blend → material
  scaling → rule-50 damping → clamp. Read for shape; not copied (D27).
- `stockfish_reference/Stockfish/src/types.h` — Value bands
  (`VALUE_TB_WIN_IN_MAX_PLY` derivation), SEE piece values (D11).
- `stockfish_reference/Stockfish/src/nnue/` — architecture shape only
  (feature transformer L1=1024, sparse-input layers, SqrClippedReLU,
  dual-perspective accumulators) as the far end of our upgrade path.
- `stockfish_reference/02-position-state-make-unmake.md` — DirtyPiece
  tracking lives inside `do_move`; put/remove centralization is why our
  `put_piece`/`remove_piece` are the only mutation points.
- PSQT provenance: **PeSTO's Evaluation Function**, Ronald Friederich
  (author of PeSTO / rofChade), published on the Chess Programming Wiki
  (chessprogramming.org/PeSTO%27s_Evaluation_Function). Widely used as a
  public-domain-style starting point; we transcribe values, not code.
- Texel tuning method: Peter Österlund's original forum description
  (CPW: chessprogramming.org/Texel%27s_Tuning_Method).
- NNUE training: the `bullet` trainer (github.com/jw1912/bullet) — external
  tooling, not a dependency (D15).

## 3. Design

### 3.1 The seam (D22)

One function, one signature, stable across all three phases:

```rust
// src/eval/mod.rs
pub fn evaluate(board: &Board, ctx: &mut EvalContext, optimism: Value) -> Value;
```

Contract:

- **Perspective**: returned `Value` is from the side to move (negamax
  convention).
- **Precondition**: never called in check. `debug_assert!(board.checkers().is_empty())`
  at the top; search substitutes `VALUE_NONE`-handling / skips static eval at
  in-check nodes (search doc).
- **Range**: always strictly inside the non-TB band (see 3.6); never a mate
  or TB value.
- **Purity boundary**: `evaluate` reads `board` and mutates only `ctx`
  (accumulator stack, pawn hash). It never touches correction history — the
  raw return value is what the TT's `eval16` stores; search applies
  correction on top (D22).
- `EvalContext` is per-thread, owned by `Worker` (D23):

```rust
pub struct EvalContext {
    // Phase B:
    pawn_table: PawnTable,          // pawn-structure hash, see 3.3.2
    // Phase C:
    nnue: NnueContext,              // accumulator stack + loaded network ref
}
```

Phase A ignores `ctx` entirely (pass it anyway so the signature never
changes). `optimism` is computed by search at aspiration-window setup (see
3.6) and is `0` in tools/tests that want a "neutral" eval (e.g. `eval`
command, tuner).

This three-argument form is the authoritative reading of D22's
`evaluate(&Board, ...)`, mirrored in doc 01 §3.3.11: per-thread mutable eval
state (pawn hash, accumulator stack) cannot hide behind `&Board` without
interior mutability or `thread_local!`, both banned on the search path (doc
13 §3.12). Borrow-wise it splits disjoint `Worker` fields —
`evaluate(&self.board, &mut self.eval_ctx, opt)`.

### 3.2 Phase A — material + tapered PSQT

#### 3.2.1 Score type

Stockfish classic packed (mg, eg) into one i32; we don't — the packing saves
nothing measurable in a leaf loop this small and invites overflow subtleties.
Idiomatic two-field struct with const constructors:

```rust
// src/eval/score.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Score { pub mg: i32, pub eg: i32 }

pub const fn s(mg: i32, eg: i32) -> Score { Score { mg, eg } }

impl Add for Score { /* field-wise */ }
impl Sub for Score { /* field-wise */ }
impl Neg for Score { /* field-wise */ }
impl Mul<i32> for Score { /* scale both */ }
```

#### 3.2.2 Game phase and tapering

PeSTO's phase scheme: per-piece increments `P=0, N=1, B=1, R=2, Q=4, K=0`,
summed over both colors; start position total = 24. Early promotions can
push the sum above 24, so clamp:

```rust
pub const GAME_PHASE_INC: [i32; PieceType::COUNT] = [0, 1, 1, 2, 4, 0];

pub fn game_phase(board: &Board) -> i32 {
    let mut p = 0;
    for color in Color::ALL {
        for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
            p += GAME_PHASE_INC[pt as usize] * board.pieces(color, pt).count() as i32;
        }
    }
    p.min(24)
}

pub fn taper(score: Score, phase: i32) -> Value {
    (score.mg * phase + score.eg * (24 - phase)) / 24
}
```

#### 3.2.3 PSQT tables (PeSTO, with provenance)

Source: *PeSTO's Evaluation Function* by Ronald Friederich, published on the
Chess Programming Wiki. These are texel-tuned tables from his engine and are
the standard public starting point for tapered-PSQT engines. We transcribe
the **values** into `src/eval/psqt.rs`; we do not copy code.

Piece values (PeSTO, exact):

| | P | N | B | R | Q | K |
|---|---|---|---|---|---|---|
| mg | 82 | 337 | 365 | 477 | 1025 | 0 |
| eg | 94 | 281 | 297 | 512 | 936 | 0 |

**Transcription trap**: the CPW arrays are written board-top-down, i.e.
index 0 = a8. Our mapping is little-endian rank-file, a1 = 0 (D3). So when
transcribing: `our_table[sq] = cpw_table[sq ^ 56]` (vertical flip). Get this
wrong and the symmetry test (3.2.6) catches it immediately — that is one of
its jobs.

Example — PeSTO pawn tables, already flipped nowhere (shown in CPW order,
first row = rank 8), to be flipped at transcription time and verified against
the CPW page:

```text
mg_pawn (a8..h8 first row):          eg_pawn:
  0   0   0   0   0   0   0   0       0   0   0   0   0   0   0   0
 98 134  61  95  68 126  34 -11     178 173 158 134 147 132 165 187
 -6   7  26  31  65  56  25 -20      94 100  85  67  56  53  82  84
-14  13   6  21  23  12  17 -23      32  24  13   5  -2   4  17  17
-27  -2  -5  12  17   6  10 -25      13   9  -3  -7  -7  -8   3  -1
-26  -4  -4 -10   3   3  33 -12       4   7  -6   1   0  -5  -1  -8
-35  -1 -20 -23 -15  24  38 -22      13   8   8  10  13   0   2  -7
  0   0   0   0   0   0   0   0       0   0   0   0   0   0   0   0
```

The remaining ten tables (N/B/R/Q/K × mg/eg) are transcribed the same way
directly from the CPW page at implementation time. Material is **baked into
the tables at startup** so the eval loop is a single lookup-and-add:

```rust
// Built once (const fn or LazyLock alongside the attack tables, D12):
// PSQT[pt][sq] = s(MG_VALUE[pt], EG_VALUE[pt]) + s(MG_TABLE[pt][sq], EG_TABLE[pt][sq])
static PSQT: [[Score; Square::COUNT]; PieceType::COUNT];

/// White uses sq as-is; Black mirrors ranks.
fn relative_sq(color: Color, sq: Square) -> usize {
    match color { Color::White => sq.index() as usize,
                  Color::Black => (sq.index() ^ 56) as usize }
}
```

#### 3.2.4 From-scratch, not incremental (decision)

Phase A computes material + PSQT **from scratch** on every call:

```rust
fn psqt_eval(board: &Board) -> Value {
    let mut total = Score::default();          // from White's POV
    let mut phase = 0;
    for color in Color::ALL {
        let sign = if color == Color::White { 1 } else { -1 };
        for pt in PieceType::ALL {
            for sq in board.pieces(color, pt) {
                total = total + PSQT[pt as usize][relative_sq(color, sq)] * sign;
                phase += GAME_PHASE_INC[pt as usize];
            }
        }
    }
    let v = taper(total, phase.min(24));
    if board.side_to_move() == Color::White { v } else { -v }
}
```

Rationale: (a) ~32 table adds is well under 100 ns — noise next to make_move
and movegen; (b) Phase B recomputes non-incremental terms (mobility, king
safety) from scratch anyway, so incremental PSQT buys nothing durable;
(c) Phase C has its own incremental machinery. Escape hatch per D14
philosophy: if a profile ever shows this loop above ~2% of search time, add
an incremental `psq: Score` to `CopiedState` (doc 04) — but only after the
measurement, and the from-scratch function stays as the test oracle.

No tempo bonus in Phase A: PeSTO ships without one, and every hand-added
constant before the tuner exists is a guess. Tempo enters (if at all) as a
tunable Phase B term.

#### 3.2.5 Assembling the Phase A `evaluate`

Even in Phase A, `evaluate` wears the full outer shape from 3.6 (dead-draw
material guard, optimism blend with `complexity = 0`, material scaling,
rule-50 damping, clamp), so that swapping in Phase B/C never changes the
seam or the search code.

#### 3.2.6 Symmetry test

Test helper (test-only, `#[cfg(test)]` or behind a `testing` feature on
`Board`):

```rust
/// Color-flip: mirror all pieces across the horizontal axis, swap colors,
/// swap side to move, mirror castling rights (K<->k, Q<->q), mirror ep square,
/// keep clocks.
pub fn flipped(&self) -> Board;
```

Invariant: `evaluate(pos) == evaluate(flipped(pos))` — equal, not negated,
because both are side-to-move relative and flipping swaps the side to move
along with everything else. Run over: the start position, the bench suite,
and a corpus of ≥ 10 000 positions harvested from perft walks of varied
openings/endgames. This test survives all three phases (NNUE perspective
networks must satisfy it too — see 3.4.2) and is the single best tripwire
for table-flip, perspective, and feature-index bugs.

Sanity anchor: `evaluate(startpos, ctx, 0) == 0` (color-symmetric position,
no tempo term).

### 3.3 Phase B — hand-crafted evaluation + texel tuning

Phase B adds terms to the white-POV `Score` accumulator before tapering. All
constants live in `src/eval/params.rs` (D20 — named, greppable, tunable).
Seed values below are **initial guesses to be texel-tuned**, not SF
constants, except where marked.

#### 3.3.1 Term catalogue

**Mobility.** For each non-pawn, non-king piece: count attacks into the
*mobility area*, look up a per-piece-type bonus array indexed by that count.

```rust
// mobility_area(us) = all squares EXCEPT: squares attacked by enemy pawns,
// our king, our queen(?), and our pawns that are blocked or on ranks 2-3
// (relative). Start without the queen exclusion; tune later.
pub static MOBILITY: [&[Score]; 4] = [ /* N: 9 entries, B: 14, R: 15, Q: 28 */ ];
```

Seed each array as a saturating ramp, e.g. knight
`[s(-60,-75), s(-25,-30), s(-5,-10), s(5,5), s(15,15), s(25,25), s(30,30), s(35,35), s(40,40)]`;
the tuner owns the final shape. Slider attacks come from the magic tables
(D12) with occupancy from `board.all_occupancy()`; x-ray conventions (rook
through own rook/queen, bishop through own queen) are a later tweak.

**Passed pawns.** A pawn with no enemy pawns ahead on its own or adjacent
files. Detected via precomputed `passed_pawn_span[color][sq]` masks (built
`const fn`, same family as the leaper tables, D12). Bonus by relative rank, plus
endgame king-proximity adjustment:

```rust
pub static PASSED_RANK: [Score; 8] =
    [s(0,0), s(5,10), s(5,15), s(10,30), s(25,60), s(60,120), s(120,200), s(0,0)];
// eg adjustment: + 5*dist(their_king, push_sq) - 2*dist(our_king, push_sq),
// applied only for rank >= 4; doubled if the push square is empty & unattacked.
```

**King safety (basics).** Two ingredients only, for now:

1. *Pawn shield*: for the three files around our king, bonus per own pawn on
   relative ranks 2-3 in front of the king, penalty per missing shield file:
   seed `SHIELD_PAWN = s(15, -5)` per pawn, `OPEN_FILE_NEAR_KING = s(-25, 5)`.
2. *Attack units* (classic CPW scheme): `units = Σ over enemy pieces
   attacking the king ring (king attacks + 1 forward rank) of
   UNIT[pt]` with seed `UNIT = [0, 2, 2, 3, 5, 0]` for P..K, then
   `danger = SAFETY_TABLE[units.min(63)]`, a monotone convex table seeded
   quadratically (`SAFETY_TABLE[u] ≈ s(-(u*u)/2, 0)`), subtracted from the
   defender. Indexing a table keeps each position linear in exactly one
   table entry, so the texel tuner can tune the table directly.

**Bishop pair.** `BISHOP_PAIR = s(25, 50)` if a side has ≥ 2 bishops (no
same-color-square check needed at this level).

**Rook on (semi-)open file.** File with no own pawns = semi-open, with no
pawns at all = open: `ROOK_SEMI_OPEN = s(20, 10)`, `ROOK_OPEN = s(45, 20)`.

**Drawish-endgame scale factor** (a hook, not a `Score` term). Applied to
the tapered white-POV value before the 3.6 outer shape:
`v = v * scale(board, stronger_side) / 64`, `scale` in `0..=64`, default 64.
Seed rules: opposite-colored bishops with no other non-pawn material →
`scale = 22 + 4 * passed_count(stronger)`; wrong-rook-pawn (stronger side's
only pawn(s) are rook pawns, its lone bishop is the wrong color for the
promotion corner, defending king controls the corner) → `scale = 0`;
stronger side pawnless and at most one minor ahead in material →
`scale = 8`. *Dead* draws never reach this hook — the 3.6 material-draw
guard returns `VALUE_DRAW` first; this hook covers the merely-drawish tail.
Constants are tuner-owned like everything else here.

**Pawn structure** (computed inside the pawn hash, 3.3.2): isolated
(`ISOLATED = s(-12, -16)`), doubled (`DOUBLED = s(-8, -25)`), backward and
connected as later additions. Passed-pawn *detection* is also cached here
(the rank/king bonuses are position-dependent and stay outside the cache).

#### 3.3.2 Pawn-structure hash

Pawn-only terms depend solely on the pawn configuration, which repeats
massively across the tree. Key by `pawn_key` (D9 — wired from the start,
zeroed on promotion ranks):

```rust
// src/eval/pawns.rs
#[derive(Clone, Copy)]
pub struct PawnEntry {
    key: u64,                     // full pawn_key; 0 = empty sentinel is fine
                                  // (pawn_key of "no pawns" is the noPawns
                                  // baseline, never 0 in practice — verify in tests)
    score: Score,                 // white-POV pawn-structure-only score
    passed: [Bitboard; 2],        // passed pawns per color
    pawn_attacks: [Bitboard; 2],  // squares attacked by each side's pawns
    semi_open_files: [u8; 2],     // bitmask: files without that side's pawns
}

pub struct PawnTable(Box<[PawnEntry]>);  // power-of-two size, seed 16384
                                         // entries (~1 MiB), per thread

impl PawnTable {
    /// Probe by board.pawn_key(); on miss, compute and overwrite (always-replace).
    pub fn probe(&mut self, board: &Board) -> &PawnEntry;
}
```

`pawn_attacks` and `semi_open_files` feed mobility area, king safety, and
rook-file terms, so the probe happens once at the top of the HCE evaluate and
the entry threads through the term functions. Always-replace is correct
here: hit rates are typically > 95% and entries are cheap to recompute.

#### 3.3.3 Texel tuning plan

**Model.** Convert eval (white POV, centipawns, `optimism = 0`) to an
expected score with a logistic:

```text
sigma(q) = 1 / (1 + 10^(-K*q/400))
E(theta) = (1/N) * SUM_i ( R_i - sigma(eval_i(theta)) )^2,   R_i in {0, 0.5, 1}
```

Fit `K` first: with the untuned eval frozen, 1-D minimize `E` over
`K in [0.5, 2.0]` (ternary search); freeze `K` for all subsequent tuning.

**Why gradients work.** The tapered HCE is *linear in its parameters*: each
parameter `theta_j` (an mg or eg constant, including every array/table
entry) contributes `c_ij * theta_j` to position i's eval, where `c_ij` is
(feature count white minus black) × (phase/24 for mg params, (24-phase)/24
for eg params). So we extract each position **once** into a sparse
coefficient vector and iterate cheaply:

```rust
// src/bin/tuner.rs  (bin target; std-only, D15)
struct Sample { coeffs: Vec<(u32 /*param idx*/, f32)>, result: f32 }

// dE/dtheta_j = -(2/N) * SUM_i (R_i - s_i) * s_i * (1 - s_i) * (K*ln(10)/400) * c_ij
```

Optimizer: Adam (lr ≈ 0.05 on centipawn-scale params, minibatch 16k,
shuffle per epoch), 5% holdout for early stopping. Fallback/cross-check:
classic ±1 coordinate descent (Texel's original local search) on a subset —
slower but assumption-free; use it to validate the gradient path once.

**Training data**, two sources, mixed:

1. *Public datasets* to bootstrap the tuner itself — e.g. the Zurichess
   `quiet-labeled` EPD set (fen + game result, pre-filtered to quiet
   positions), a standard public tuning corpus. Verify each dataset's
   license before committing any of it to the repo; prefer downloading in a
   `tools/` script over vendoring.
2. *Fixed-node self-play* (own generator, shared with Phase C datagen):
   games at ~5000 nodes/move from randomized 8-ply openings; sample ≤ 16
   positions per game; label each with the final game result.

Position filters: not in check; quiet (qsearch value == static eval, or
resolve to the qsearch PV leaf and store that); skip the first 8 plies;
skip |static eval| > 1000; deduplicate by Zobrist key. Target corpus:
1M+ positions to start, growing to ~10M.

**Acceptance for a tuning run**: holdout error strictly decreases AND the
retuned engine passes SPRT vs. the previous params (D25) — error going down
without Elo going up means the data or filters are wrong, not the engine.

### 3.4 Phase C — NNUE

#### 3.4.1 First architecture

`768 -> (2x256) -> 1`, perspective networks, SCReLU activation:

- **Input features** (768 = 2 colors × 6 piece types × 64 squares): for
  perspective `p`, piece `(c, pt)` on square `sq` activates
  `idx = ((c != p) as usize * 6 + pt as usize) * 64 + rel_sq(p, sq)` where
  `rel_sq(White, sq) = sq`, `rel_sq(Black, sq) = sq ^ 56`. No king buckets
  in v1 — this makes accumulators refresh-free except at root (a large
  simplification; buckets come with the upgrade path).
- **Feature transformer**: one shared weight matrix `W0: [768][256] i16` +
  bias `b0: [256] i16`; each perspective maintains its own accumulator
  `acc[p] = b0 + Σ_active W0[feature]`.
- **Output layer**: concatenate `[activation(acc[stm]), activation(acc[nstm])]`
  (512 values, side-to-move half first — this ordering is what makes the net
  side-to-move relative) and dot with `W1: [512] i16`, add `b1`, scale.
- **Activation**: SCReLU `f(x) = clamp(x, 0, QA)^2` (stronger than CReLU at
  equal size, standard in bullet); CReLU `clamp(x, 0, QA)` kept as a format
  option (`activation` field in the header, 3.4.5).

**Upgrade path** (each step SPRT-gated, D25): L1 256 → 512 → 1024; output
buckets by piece count (8 buckets, `idx = (popcount(occ) - 2) / 4`)
selecting among 8 output columns; horizontally-mirrored king-bucketed inputs
(HalfKAv2_hm-style, which is when king moves force accumulator refreshes and
an accumulator-cache table becomes worthwhile); a separate PSQT output head
per bucket (restores the SF `psqt/positional` split and with it the
complexity term, 3.6). Current SF for calibration: threat-augmented feature
set, L1 = 1024, deep 15/32-neuron tail with sparse propagation, dual
big/small nets — that is the destination, not the first step.

#### 3.4.2 Symmetry

The perspective construction guarantees `evaluate(pos) == evaluate(flipped(pos))`
structurally (the two perspectives swap roles). The 3.2.6 symmetry test stays
on in Phase C and now guards the feature indexer and accumulator updates.

#### 3.4.3 Quantization scheme

bullet-standard integer scheme; all constants recorded in the net header:

- `QA = 255` — feature-transformer scale: `w0_q = round(w0 * QA)`,
  `b0_q = round(b0 * QA)`, stored i16.
- `QB = 64` — output-layer scale: `w1_q = round(w1 * QB)`, stored i16;
  output bias `b1_q = round(b1 * QA * QB)`, stored i32.
- `SCALE = 400` — centipawn output scale.
- Accumulation is i32 throughout.

Forward pass, SCReLU (per-term scale QA²·QB):

```text
sum   = SUM_i clamp(acc_i, 0, QA)^2 * w1_i        // i32 (see overflow note)
v_cp  = (sum / QA + b1_q) * SCALE / (QA * QB)
```

For CReLU drop the `^2` and the `/ QA`. **Training-side constraint**: L1
weights are clipped to ±(32767 / QA / QB) ≈ ±1.98 in float during training
so that the i16 `mullo(a, w)` in the SIMD SCReLU trick (3.4.6) cannot
overflow (`255 * 127 = 32385 < 32767`). Theoretical worst-case i32 overflow
of the 512-term sum exists but is unreachable with real nets (activations
and weights never co-saturate); the scalar reference path accumulates in i64
so any real overflow would show up as a scalar/SIMD mismatch in tests.

#### 3.4.4 Incremental accumulators and dirty pieces

**Recording (through doc 04's hook seam).** Doc 04 deliberately keeps NNUE
knowledge out of `make_move` proper; what it provides is the monomorphized
`MakeMoveHooks` trait on `make_move_with` (doc 04 §3.10, where `NoHooks`
compiles to nothing). This doc attaches to that seam by adding one
default-no-op method to the trait and owning its payload:

```rust
// Trait owned by doc 04 (§3.10 there); this method and its payload owned here.
pub trait MakeMoveHooks {
    // ... on_key / on_state (doc 04) ...
    /// Hook C: the move's piece deltas, complete before make_move_with returns.
    #[inline] fn on_dirty(&mut self, _dp: &DirtyPiece) {}
}

// src/eval/nnue/: consumed by the accumulator walk below.
#[derive(Clone, Copy, Default)]
pub struct DirtyPiece {
    adds: [(Piece, Square); 2],   // <= 2 used (castling: king + rook)
    subs: [(Piece, Square); 2],   // <= 2 used (capture-promotion: pawn + victim)
    n_add: u8,
    n_sub: u8,
}
```

Normal move: 1 sub + 1 add. Capture (incl. ep): 2 subs + 1 add. Promotion:
sub pawn (+ sub victim), add promoted piece. Castling: 2 subs + 2 adds.
Null move: empty record (0 adds, 0 subs). Because the hook is a default
no-op, plain `make_move` (which delegates via `NoHooks`), perft, and Phases
A/B are bit-identical to today — the bench neutrality doc 04 promises for
the seam, for free instead of by an "unconditional stores are cheap" bet.
In Phase C the *search's* `do_move` wrapper calls
`make_move_with(m, gives_check, &mut NnueHooks { .. })` and the hook writes
the record directly into the accumulator-stack entry it belongs to (below).

**Lazy update (SF's approach).** Most made moves are pruned before ever
being evaluated, so accumulators update lazily at evaluate time, not in
`make_move`:

```rust
// src/eval/nnue/accumulator.rs
pub struct AccumulatorEntry {
    acc: Aligned64<[[i16; L1]; 2]>,   // indexed by Color
    dirty: DirtyPiece,                // deltas of the move that LED TO this entry
    computed: [bool; 2],
}

pub struct NnueContext {
    stack: Box<[AccumulatorEntry]>,   // MAX_PLY + margin entries, indexed by
                                      // SEARCH ply (root = 0) — never by
                                      // board.states.len(), which includes
                                      // UCI-replayed game history and can
                                      // exceed MAX_PLY in a long game
    net: &'static Network,            // loaded once (LazyLock / UCI EvalFile)
}
```

**Invalidation rule (the correctness linchpin).** An entry may be trusted
only while the path that produced it is still the current search path.
Every `do_move` from search ply `k` — via the `on_dirty` hook; null moves
directly in the `do_null_move` wrapper — stores
`stack[k + 1].dirty = dp; stack[k + 1].computed = [false, false];` before
the child node runs. Without this, unmaking back to ply `k` and making a
*different* move would silently reuse the stale accumulator a sibling
subtree left at `k + 1`. (Stockfish gets the same guarantee structurally by
storing the accumulator inside `StateInfo`, which is fresh per `do_move`;
with a ply-indexed stack the explicit invalidation *is* the freshness
guarantee.) `stack[0]` is refreshed from scratch (both perspectives,
`computed = [true, true]`) at the start of every `go`, so the root entry is
always valid regardless of how much game history UCI replayed.

At `evaluate` (search ply `p`), for each perspective: if
`stack[p].computed[side]`, done. Otherwise walk back to the nearest computed
ancestor entry `k` (guaranteed to exist — `stack[0]` is always computed),
then for each `i in k+1..=p` apply `stack[i].dirty`'s subs/adds forward as
column add/subs on the 256-wide vector. Dirty records are addressed entirely
within the stack — no reach into `Board`'s StateInfo history, no
root-offset arithmetic against `board.states.len()`. With unbucketed 768
inputs a refresh is never *forced* by a king move — that starts only with
king buckets (upgrade path), which is also when a refresh-cache (per-bucket
cached accumulator + diff) earns its keep.

Fused add+sub kernels (`acc += W0[a] - W0[s]`) handle the dominant
1-add/1-sub case in one pass over the vector.

#### 3.4.5 Net file format (own, versioned)

We never ship Stockfish nets (D27); we define our own container:

```text
offset  size  field
0       4     magic = b"CENN"
4       4     version: u32 LE (= 1)
8       4     input_dims: u32 (= 768)
12      4     l1: u32 (= 256)
16      1     activation: u8 (0 = CReLU, 1 = SCReLU)
17      3     padding (zero)
20      2     qa: u16 (= 255)
22      2     qb: u16 (= 64)
24      2     scale: u16 (= 400)
26      6     padding (zero)
32      8     payload_len: u64 LE
40      8     payload_checksum: u64 LE (FNV-1a 64 over payload bytes)
48      -     payload, all little-endian, in order:
              ft_weights: i16[input_dims * l1]   (feature-major: W0[feature][neuron])
              ft_bias:    i16[l1]
              out_weights:i16[2 * l1]            (stm half first)
              out_bias:   i32
```

Loader (`src/eval/nnue/loader.rs`): validate magic, version, dims,
activation id, checksum; reject anything unexpected with a real error type
(never panic on user-supplied files). The default net is embedded via
`include_bytes!` (zero runtime deps, D15) with a UCI `EvalFile` option to
load an external file; if the crate is built without a net (pre-Phase-C or
CI), a cargo feature keeps the PSQT/HCE path as the compiled fallback so the
engine always builds and runs.

Feature-major weight layout means an accumulator update touches one
contiguous 512-byte row per feature — cache-optimal for the add/sub kernels.

#### 3.4.6 SIMD strategy

Stable `std::arch` AVX2 intrinsics with a scalar fallback; **no `std::simd`**
(nightly-only — explicitly out, per the scope decision here). Two backends
only, selected once:

```rust
// src/eval/nnue/simd.rs
#[repr(align(64))] pub struct Aligned64<T>(pub T);

pub struct Kernels {
    // acc = base + W0[add] - W0[sub]   (and 2-add/2-sub variants)
    pub add_sub: unsafe fn(dst: &mut [i16; L1], src: &[i16; L1], add: &[i16; L1], sub: &[i16; L1]),
    pub refresh: unsafe fn(dst: &mut [i16; L1], bias: &[i16; L1], feats: &[u16], w: &FtWeights),
    pub forward: unsafe fn(us: &[i16; L1], them: &[i16; L1], w1: &[i16; 2 * L1]) -> i32,
}

pub static KERNELS: LazyLock<Kernels> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") { return avx2::KERNELS; }
    scalar::KERNELS
});
```

- AVX2 kernels are `#[target_feature(enable = "avx2")] unsafe fn`; the single
  dispatch call site carries the one `// SAFETY: detected at startup` comment
  (D14b). With `-C target-cpu=x86-64-v3` (D16) the compile-time
  `cfg(target_feature = "avx2")` path can skip runtime detection.
- SCReLU forward per 16-lane chunk: clamp via `_mm256_min_epi16` /
  `_mm256_max_epi16` (a in [0, QA]), `_mm256_mullo_epi16(a, w)` (fits i16 by
  the 3.4.3 weight clip), then `_mm256_madd_epi16(prod, a)` → pairwise i32
  `a² * w`; accumulate i32 lanes, horizontal-add once at the end.
- Scalar backend is plain safe Rust (the autovectorizer does respectably on
  the add/sub kernels), accumulates the forward pass in i64, and is the
  reference the AVX2 path must match **bit-exactly** in tests.
- AVX-512 / VNNI: later, same dispatch seam, only after AVX2 is measured.

#### 3.4.7 Training pipeline (external bullet, D15)

1. **`datagen` bin target** (`src/bin/datagen.rs`): fixed-node self-play
   (~5000 nodes/move) from randomized 8-ply openings; emits positions with
   (packed board, stm-relative search score, game result) in bulletformat
   (or a text form converted by bullet's utilities). Filters: no in-check
   positions, skip first 8 plies, skip |score| ≥ 2500, cap positions/game.
   Multithreaded via `std::thread` (one self-play worker per core).

   *Game loop and termination.* Fixed-node games never end on their own in
   shuffle positions, so every game terminates by an explicit rule and the
   result label `R ∈ {0, 0.5, 1}` (white-POV) is always well-defined:
   - no legal moves: checkmate → win for the mating side; stalemate → draw;
   - threefold repetition and the 50-move rule, counted over the **whole
     game history** (game-level equivalents of doc 04's `is_draw`, not its
     search-ply-relative encoding) → draw;
   - insufficient material (the 3.6 `is_material_draw` rule) → draw;
   - hard cap: game length ≥ 400 plies → draw;
   - score adjudication (on by default, CLI flags to disable/retune):
     win if |score| ≥ 2500 for 4 consecutive moves (from the winning side's
     perspective); draw if |score| ≤ 8 for 10 consecutive moves after
     ply 80.
   All sampled positions of a game carry that game's `R`.

   *Reproducibility.* D15 forbids a `rand` dependency, so opening
   randomization reuses the zobrist module's `xorshift64star` PRNG (doc 04
   §3.4): one PRNG per worker thread, seeded
   `seed = cli_seed.wrapping_add(thread_index as u64 * 0x9E37_79B9_7F4A_7C15)`
   with `--seed` a required CLI argument. Each worker owns a private TT and
   plays its games sequentially, so a worker's output stream is a pure
   function of (engine version, seed, thread index, node limit, filter and
   adjudication settings). All of those — plus the bench signature of the
   generating binary (D24) — are recorded in the corpus metadata header so
   any corpus is regenerable.
2. **Train with bullet** (external, GPU): arch `768 -> 2x256 -> 1` SCReLU,
   our quantization constants; standard recipe: interleave/shuffle data,
   WDL-blended target (`lambda ≈ 0.7` score / 0.3 result to start).
3. **`netpack` helper** (small bin or script): wrap bullet's raw quantized
   output in the CENN container, compute checksum.
4. **Iterate**: gen-0 data from the Phase B (HCE) engine → net-1 → engine
   with net-1 generates gen-1 data → retrain. First competitive nets
   typically want ≥ 100M positions; every net swap is SPRT-gated (D25).
5. Nets are build artifacts, not source: keep them out of git history
   (release attachments or LFS), embed at build time.

### 3.5 (reserved)

Numbering intentionally skips ahead so search-consumption formulas sit in
one place:

### 3.6 How search consumes eval (the outer shape)

All constants below are SF-current starting points, recorded verbatim
(D20); shapes matter more than exact values. They live in
`src/eval/params.rs` / `src/search/params.rs`, named.

**Value bands** (derived from D5's constants, matching SF `types.h`):

```rust
pub const VALUE_MATE: Value = 32000;
pub const MAX_PLY: i32 = 246;
pub const VALUE_MATE_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY;        // 31754
pub const VALUE_TB: Value = VALUE_MATE_IN_MAX_PLY - 1;                // 31753
pub const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_TB - MAX_PLY;        // 31507
pub const VALUE_TB_LOSS_IN_MAX_PLY: Value = -VALUE_TB_WIN_IN_MAX_PLY; // -31507
```

**Optimism** (computed by search at aspiration-window setup, §3.1 of the
search notes; consumed here): with `avg` = the root move's running average
score, `optimism[us] = 137 * avg / (|avg| + 81)`, `optimism[them] = -optimism[us]`.
The side winning at the root evaluates slightly optimistically.

**The outer evaluate** (Phase C form; Phases A/B use the same shape with
`complexity = 0` since they have no psqt/positional split):

```rust
pub fn evaluate(board: &Board, ctx: &mut EvalContext, optimism: Value) -> Value {
    debug_assert!(board.checkers().is_empty());

    // Dead-draw guard (all phases inherit it): insufficient mating
    // material is an exact draw — bypass the whole pipeline. Dither-free;
    // search's rule50/repetition draw handling is untouched.
    if is_material_draw(board) {
        return VALUE_DRAW;
    }

    // Inner eval, stm-relative. Phase A/B: (v, 0). Phase C with a PSQT-head
    // net: (psqt, positional); first net (no head): (nnue_out, 0).
    let (psqt, positional) = inner_eval(board, ctx);
    let mut v: i64 = (psqt + positional) as i64;
    let mut opt: i64 = optimism as i64;

    // Complexity blend (active once the net has a psqt/positional split):
    let complexity = (psqt - positional).abs() as i64;
    opt += opt * complexity / 476;
    v -= v * complexity / 18236;

    // Material scaling: heavier boards trust eval more, lighter boards
    // trust optimism relatively more.
    let material: i64 = 534 * pawn_count_both(board) as i64
                      + board.non_pawn_material_total() as i64;
    v = (v * (77871 + material) + opt * (7191 + material)) / 77871;

    // Rule-50 damping: shuffle toward zero as the counter grows.
    v -= v * board.halfmove_clock() as i64 / 199;

    // Never collide with TB/mate bands.
    (v as Value).clamp(VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
}
```

**`is_material_draw`** (the dead-draw guard): true iff no pawns are on the
board, one side is a bare king, and the other side's non-pawn material is at
most a single minor or exactly two knights — i.e. KvK, KNvK, KBvK, KNNvK.
None of these has a forced mate (KNNvK is included deliberately; only
helpmates exist). Cheap to compute from doc 04's cached state, with
`npm = non_pawn_material` and the `PIECE_VALUE` constants:
`pawns == 0 && npm(weak) == 0 && (npm(strong) <= PIECE_VALUE[Bishop] || npm(strong) == 2 * PIECE_VALUE[Knight])`.
The guard exists because doc 04's `is_draw` covers only rule50 + repetition
(correct, SF parity) — nothing else in the suite knows about dead material.
Without it, Phase A PSQT scores KBvK around +365: the engine would trade a
winning position into a dead draw believing it is +3, shuffle pointlessly in
drawn endings, report misleading scores, and mislabel Phase B/C training
positions. Symmetric by construction (`flipped()` swaps
the bare king with the minor side and the guard still fires), so the 3.2.6
symmetry test stays green. Broader *drawish* (not dead) endings are handled
by the Phase B scale factor (3.3.1), never by this guard.

Notes:

- **i64 intermediates are mandatory**: `v * (77871 + material)` reaches
  ~3.3e9 and overflows i32. SF casts to 64-bit here for the same reason.
- `534` is the pawn weight in the material term; `77871`, `7191`, `476`,
  `18236`, `199` are SF-current (evaluate.cpp) — starting points (D20).
- The **shape to keep** across phases (per the search notes §10.1): *eval
  blended with a side-to-move optimism term that scales with material*, then
  rule-50 damped and clamped.

**Correction history — applied in search, not here** (D22). For reference,
the boundary contract: search computes
`cv = 13345*pawn_corr + 9280*minor_corr + 11840*(non_pawn_w + non_pawn_b) + 8363*(cont_corr(ss-2) + cont_corr(ss-4))`
and uses `corrected = clamp(raw + cv/131072, TB band)` for all pruning
decisions, while the **raw** `evaluate` return is what goes into the TT's
`eval16`. Nothing in `src/eval/` may read or write correction tables.

### 3.7 Module map

```text
src/eval/
  mod.rs        evaluate() seam, EvalContext, outer shape (3.6)
  score.rs      Score, s(), taper, game_phase
  psqt.rs       PeSTO tables (material baked in), psqt_eval
  params.rs     every named constant (D20)
  hce.rs        Phase B terms (mobility, king safety, threats later)
  pawns.rs      PawnEntry, PawnTable
  trace.rs      term-by-term breakdown for the `eval` CLI/UCI command
  nnue/
    mod.rs      NnueContext, inner_eval wiring
    network.rs  Network (weights), forward pass
    accumulator.rs  AccumulatorEntry, lazy update walk, refresh
    loader.rs   CENN parse/validate, include_bytes! default
    simd.rs     Kernels, scalar + avx2 backends
src/bin/
  tuner.rs      Phase B texel tuner
  datagen.rs    self-play data generator (Phase B texel + Phase C training)
```

## 4. Implementation plan

Each step is one commit-sized unit with a done-condition. Steps 1-4 are
Phase A (needed before search work can be validated); 5-8 Phase B; 9-14
Phase C.

1. **`Score` + phase machinery** (`eval/score.rs`): `Score`, ops, `s()`,
   `game_phase`, `taper`. *Done*: unit tests for ops, tapering edge cases
   (phase 0, 24, clamped > 24).
2. **PeSTO PSQT** (`eval/psqt.rs`): transcribe 12 tables + values, bake
   material, `relative_sq`, `psqt_eval`. *Done*: startpos evals to 0;
   spot-check three asymmetric positions against hand-computed values;
   `sq ^ 56` flip verified by a test that a white pawn on e4 scores what a
   black pawn on e5 scores.
3. **Seam + outer shape** (`eval/mod.rs`, `eval/params.rs`): `evaluate` with
   dead-draw guard + optimism/material/rule-50/clamp wiring (complexity = 0),
   `EvalContext` stub. *Done*: formula unit tests (rule-50 damping at clock
   0/100/199 boundaries; clamp engages on synthetic ±40000 inner values; i64
   path exercised with max material; dead-draw guard returns 0 for
   KvK/KNvK/KBvK/KNNvK and does not fire for KRvK/KPvK/KBvKN).
4. **`Board::flipped()` + symmetry suite**: test-only helper + corpus test.
   *Done*: symmetry holds over startpos + ≥ 10k harvested positions; this
   test is wired into `cargo test` permanently.
5. **Pawn hash** (`eval/pawns.rs`): `PawnEntry`, `PawnTable`, isolated/
   doubled/passed detection + pawn score. *Done*: probe-hit returns
   identical entry to fresh compute (randomized test); symmetry still green.
6. **HCE terms** (`eval/hce.rs`): mobility, bishop pair, rook files, passed
   pawn bonuses, king shield + attack units. Add `eval` trace command.
   *Done*: symmetry green; trace prints per-term white-POV breakdown;
   bench node count recorded (functional change, D24).
7. **Tuner** (`bin/tuner.rs`): EPD loader, coefficient extraction, K fit,
   Adam + coordinate-descent cross-check. *Done*: on the public bootstrap
   dataset, holdout error decreases monotonically to a plateau; retuned
   params committed to `eval/params.rs`.
8. **Phase B gate**: SPRT retuned-HCE vs. Phase A (D25). *Done*: clear pass
   (expected to be large, ≥ +150 Elo at STC).
9. **DirtyPiece hook** (with doc 04): add the default-no-op `on_dirty`
   method to `MakeMoveHooks` and an eval-side hooks impl that records
   through `make_move_with` (3.4.4). *Done*: unit test reconstructs board
   deltas from DirtyPiece across a perft-3 walk of tricky positions
   (castling, ep, promotions) driven through `make_move_with`; perft suite
   unchanged (D24); bench unchanged (the plain `make_move`/`NoHooks` path is
   untouched — non-functional by construction).
10. **Net format + scalar inference** (`nnue/loader.rs`, `network.rs`,
    scalar `simd.rs`): load CENN, from-scratch accumulators, scalar forward.
    *Done*: a randomly-generated test net round-trips through the format;
    checksum/dims/version rejects tested; eval of fixed positions matches a
    reference implementation of the same net in the tuner-side test harness.
11. **Lazy incremental accumulators** (`nnue/accumulator.rs`): search-ply
    stack, invalidation in `do_move`/`do_null_move`, root refresh per `go`
    (3.4.4). *Done*: incremental == from-scratch refresh at every node of a
    100k-node random game walk (debug-assert also left in place under
    `debug_assertions`); sibling test — make `m1` at ply `k`, evaluate,
    unmake, make `m2` at the same ply, evaluate — matches from-scratch (no
    stale sibling accumulator reuse); a position reached after > MAX_PLY
    replayed game moves evaluates correctly (stack indexing is search-ply,
    not game-history length).
12. **AVX2 backend**: kernels + dispatch. *Done*: AVX2 == scalar bit-exact
    on 10k random accumulator states and positions; ns/eval measured and
    recorded (see §6).
13. **Datagen + first net** (`bin/datagen.rs`, bullet config, `netpack`):
    generate ≥ 100M positions with the Phase B engine, train, embed.
    *Done*: engine loads and runs with the net; symmetry green.
14. **Phase C gate**: SPRT net vs. Phase B HCE. *Done*: clear pass
    (expected large). Then iterate the upgrade path (L1 512, buckets, ...),
    one SPRT-gated step at a time.

## 5. Testing and validation

- **Symmetry** (the workhorse): `evaluate(pos) == evaluate(pos.flipped())`
  over startpos + bench suite + ≥ 10k harvested positions. Runs in every
  phase; catches table flips, perspective bugs, feature-index bugs, castling/
  ep mirroring bugs in `flipped()` itself.
- **Anchors**: startpos == 0 (Phases A/B); dead-draw FENs (KvK, KNvK, KBvK,
  KNNvK, both colors as the strong side) evaluate exactly 0 in every phase
  while near-misses (KRvK, KPvK, KBvKN) do not hit the guard; these FENs are
  also in the symmetry suite; eval strictly inside
  `(VALUE_TB_LOSS_IN_MAX_PLY, VALUE_TB_WIN_IN_MAX_PLY)` fuzzed over random
  material dumps.
- **Determinism**: same position + same context state ⇒ same value, across
  scalar and AVX2 backends (bit-exact), across refresh vs. incremental.
- **Pawn hash coherence**: randomized make/unmake walks; probe result equals
  fresh computation every time (the pawn_key (D9) is doing the caching).
- **Formula unit tests**: rule-50 damping endpoints, clamp engagement, i64
  overflow guard (synthetic max-material max-eval input), optimism sign
  convention (positive optimism raises stm eval).
- **Net format**: fuzz the loader with truncated/corrupted headers; never
  panics, always errors.
- **Bench as regression oracle** (D24): every functional eval change reports
  the new bench node count in its commit message; refactors must leave it
  identical. Add an eval-heavy slice to the bench suite (quiet middlegames,
  fortress-ish endgames, high-rule50 positions).
- **Strength gates** (D25): phase swaps and every net/params change pass
  SPRT vs. the predecessor; no eyeballing.
- **Perft untouched**: the DirtyPiece hook (step 9) rides doc 04's
  `MakeMoveHooks` seam; the plain `make_move`/`NoHooks` path stays
  bit-identical, and step 9 lands only with the perft suite green (D24).

## 6. Performance notes

Targets (single thread, x86-64-v3, release profile per D16):

- Phase A `evaluate`: < 100 ns.
- Phase B HCE: 1-2 µs typical; pawn-hash hit rate > 90% in search.
- Phase C NNUE (768→2x256, AVX2): refresh < 300 ns; incremental
  1-add/1-sub update ~30-60 ns; forward pass < 150 ns; whole-search NPS
  roughly halving vs. Phase A is normal and pays for itself in Elo.

Pitfalls to design against (and what to measure):

- **i32 overflow** in the material blend — i64 intermediates (3.6); test it.
- **Alignment**: accumulators and weight rows in `Aligned64`; unaligned
  loads on AVX2 are legal but measurably slower on some cores.
- **Bounds checks** in kernels: iterate with `chunks_exact`/`zip` on fixed
  arrays so the optimizer elides them; only reach for `get_unchecked` after
  a profile shows a residue (D14c).
- **Lazy-update pathology**: a long walk-back replays many plies; measure
  average replay depth — if it exceeds ~2-3, evaluate-at-make for PV nodes
  or a smarter "compute at midpoint" policy.
- **Dispatch overhead**: kernels are resolved once into a static; no
  per-call feature detection.
- Measure with: a `bench eval` micro-mode (1M evals over the suite →
  ns/eval), plus criterion dev-benches for kernels (D15 allows dev-deps).
  Track ns/eval and bench NPS in commit messages for eval-touching commits.

## 7. Interactions

(Filenames per the suite index; numbers may shift as docs land.)

- **Depends on** `design/04-make-unmake.md`: StateInfo (D8), `pawn_key` and
  key family (D9), `non_pawn_material`, `checkers()`, and the `MakeMoveHooks`
  seam (§3.10 there) that carries the `DirtyPiece` recording this doc
  specifies in 3.4.4.
- **Depends on** the attack-tables doc (D12): slider/leaper attacks for
  mobility, king ring, passed-pawn spans; `LazyLock` init pattern reused for
  PSQT bake and NNUE net.
- **Feeds** the search doc: `evaluate` is called at every non-check node for
  the static eval; search owns optimism computation (137/81 at aspiration
  setup), correction history (applied on top of our raw value), and all
  pruning margins that read the eval. In Phase C search's
  `do_move`/`do_null_move` wrappers fill and invalidate the NNUE
  accumulator-stack entries (3.4.4).
- **Feeds** the TT doc (D19): `eval16` stores our raw return value.
- **Coordinates with** the threading doc (D23): `EvalContext` (pawn table +
  accumulator stack) is per-`Worker`, no sharing, no locks.
- **Coordinates with** the UCI/bench doc (D24): `EvalFile` option, `eval`
  trace command, bench suite composition, node-count reporting protocol.
- **Tooling**: `tuner`, `datagen`, `netpack` bin targets; external
  cutechess/fastchess for SPRT (D25) and bullet for training (D15).

## 8. Acceptance criteria (D26)

Phase A done when:

1. `evaluate(&Board, &mut EvalContext, Value) -> Value` exists with the full
   outer shape (optimism/material/rule-50/clamp) and search compiles against
   it.
2. Symmetry test green over startpos + ≥ 10k positions; startpos evals 0;
   KvK/KNvK/KBvK/KNNvK evaluate exactly 0 (dead-draw guard, 3.6).
3. All 12 PeSTO tables transcribed, material baked, flip verified.
4. Every constant named in `params.rs` (D20); no inline magic numbers.
5. `debug_assert!(no checkers)` in place; bench runs reproducibly (D24).

Phase B done when:

6. Pawn hash probe == fresh compute under randomized walks; hit rate
   measured > 90% in bench.
7. All 3.3.1 terms implemented with an `eval` trace command showing the
   white-POV breakdown.
8. Tuner (bin target, std-only) fits K, reduces holdout error on the
   bootstrap dataset, and the retuned engine passes SPRT vs. Phase A (D25).

Phase C done when:

9. CENN loader validates magic/version/dims/checksum and never panics on
   bad input; default net embedded; `EvalFile` override works.
10. Incremental accumulators == from-scratch refresh at every node of a
    100k-node randomized walk, including the sibling-invalidation and
    long-game-history cases (3.4.4 / step 11).
11. AVX2 kernels bit-exact vs. scalar; runtime detection falls back cleanly
    on non-AVX2 hardware.
12. A self-trained net (bullet, own datagen data) passes SPRT vs. Phase B.
13. No Stockfish network files anywhere in the repo or the release artifacts
    (D27); net provenance (data generations, training config) documented.
14. Symmetry test still green through all of the above.

## 9. Open questions

1. **Complexity term pre-PSQT-head**: the first net has no psqt/positional
   split, so `complexity = 0` until the upgrade path adds a PSQT output
   head. Alternative: approximate complexity as `|mg - eg|` of the Phase A
   tapered pair in Phases A/B. Decided *against* for now (one more untuned
   knob); revisit when the PSQT head lands.
2. **Public tuning dataset licensing**: verify the exact license of the
   Zurichess quiet-labeled set (and any lichess-derived corpus) before any
   copy enters the repo; if unclear, use download-scripts only and rely on
   own datagen. Needs a one-time check when step 7 starts.
3. **Net distribution**: embedded default via `include_bytes!` is decided,
   but where net binaries live (GitHub release assets vs. git LFS) is a
   repo-ops choice deferred until the first real net exists.
4. **Phase A incremental PSQT**: from-scratch is the decision; the escape
   hatch (incremental `psq` in `CopiedState`) triggers only on a profile
   showing > ~2% of search time in `psqt_eval` — owned jointly with doc 04
   if it ever fires.
5. **Tempo term**: omitted in Phase A by design; whether Phase B gains a
   tuned tempo constant is left to the tuner experiments (add the parameter,
   let SPRT decide).
