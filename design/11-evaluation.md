# 11 — Evaluation: PSQT to HCE to NNUE

Status: Reviewed v1

> **Amendment (2026-07-11, D21/D28):** Phase B (HCE + texel tuning) is
> **descoped to an optional contingency** — it existed mainly to generate
> NNUE data under the own-data rule, but D28 makes Stockfish the bootstrap
> data generator, so HCE is off the critical path (P9 → P10 → P11 → P13).
> The Phase B sections below are retained as the contingency spec, built only
> if NNUE stalls. Consequently the **Phase C (NNUE) SPRT baseline is Phase A
> (PSQT)** unless the HCE contingency was built; read "vs Phase B" below as
> "vs the prevailing baseline (Phase A by default)". NNUE data provenance
> follows D28/D29 (bootstrap-distill then retrain clean); the datagen
> platform is design/15 + design/16.

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

> **Step-4 wiring note (P13 step 4, 2026-07-12; NNUE goes live).** As built,
> refined from the sketch above (all consistent with its intent):
> - **The seam gains a fourth argument, `ply: usize`** — the SEARCH ply of
>   `board` (root = 0). PSQT ignores it; NNUE needs it to index the
>   ply-indexed accumulator stack (§3.4.4). Real signature:
>   `pub fn evaluate(board: &Board, ctx: &mut EvalContext, optimism: Value, ply: usize) -> Value`.
>   Standalone callers (`eval` command, tools, tuner) pass `0` after a
>   `refresh_root`; search threads its node ply through. This is the minimal
>   honest reading of the seam once the accumulator is ply-addressed — the
>   PSQT/HCE consumption is otherwise untouched.
> - **`EvalContext` holds `nnue: Option<NnueContext<'static>>`** (the Phase B
>   `pawn_table` is still future). `None` ⇒ NNUE off (PSQT), the zero-cost
>   default that keeps the depth-6 bench bit-identical (548978). A net is
>   installed from the process-wide active net (`EvalFile`) at the search
>   boundary via `EvalContext::set_net`; the accumulator is refreshed at the
>   root of every search (`refresh_root`) and pushed per move
>   (`push_dirty`, the §3.4.4 invalidation rule) from the search's `do_move`
>   wrapper (NOT the `on_dirty` hook — the direct wrapper avoids the
>   borrow-split and the NoHooks-codegen perturbation the §3.4.4 note flagged;
>   both are permitted, this is the chosen one). Null moves push
>   `DirtyPiece::EMPTY`.
> - **`EvalFile` is opt-in (design/11 §3.4.5), NOT an embedded default net
>   this step.** A UCI `EvalFile` string option (empty = PSQT) and an optional
>   6th `bench` positional arg both load a CENN net into a
>   process-wide `RwLock<Option<&'static Network>>` (leaked once,
>   `unsafe`-free); loader errors surface as a clean UCI `info string`, never
>   a panic. Embedding a default net (so NNUE is the out-of-the-box eval) is a
>   later step. This keeps the PSQT/NNUE A/B a clean toggle and the DEFAULT
>   bench unchanged.
> - **Debug oracle:** `evaluate` calls
>   `NnueContext::assert_matches_refresh(board, ply)` at every NNUE eval node
>   in debug builds — the search analogue of the step-1 incremental==refresh
>   fuzz, trapping any stale-accumulator bug the instant search evaluates.

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

> **Step-3 precondition (review, 2026-07-12).** The bullet training recipe
> **must** apply this output/L1 weight clip (`|w1_float| ≤ 32767/(QA·QB) ≈
> 1.98` ⇒ quantized `|out_weight| ≤ 127`) — not merely the ≤ i16 range bullet
> guarantees at save (§3.4.5 packer note). The step-2 AVX2 `mullo_epi16` SCReLU
> trick (§3.4.6) is bit-exact vs the scalar reference **only** under this clip;
> a net that violates it would checksum-clean yet score differently on AVX2 vs
> scalar. The loader now enforces it as a fail-fast precondition:
> `load_bytes` rejects any `|out_weight| > 127` with
> `NnueError::OutWeightOutOfClip` rather than loading a net the two backends
> would evaluate differently. This is a net-0 (step-3) training requirement,
> backstopped at load.

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

> **Step-1 implementation note (P13 step 1, 2026-07-12; `src/eval/nnue/`).**
> As built, refined from the sketch above (all consistent with its intent):
> - **`DirtyPiece` lives in `board.rs`** (the low module), not
>   `src/eval/nnue/`, and `eval::nnue` re-exports it. Its data is a pure
>   board-mutation fact and its constructor
>   `DirtyPiece::from_move(board_before, m)` reads `Board`'s private castling
>   geometry, so keeping it in `board` avoids a `board → eval` layering
>   inversion (the foundational module never reaches up into `eval`). The
>   `on_dirty(&mut self, &DirtyPiece)` hook method is added to
>   `MakeMoveHooks` as a default no-op but is **NOT yet called from
>   `make_move_with`** — computing the payload has a panic-carrying `expect`,
>   so a speculative unconditional call could perturb `NoHooks` codegen; the
>   call site lands with the search wiring (step 4). Bench stays bit-identical
>   (548978).
> - **The `Kernels` seam uses safe `fn` pointers** this step (scalar-only
>   under `#![deny(unsafe_code)]`); they become `#[target_feature]`
>   `unsafe fn` pointers when the AVX2 backend lands (step 2, §3.4.6).
> - **`NnueContext<'a>` borrows the net** (`net: &'a Network`) rather than
>   `&'static Network`; it is not yet stored in the `Worker`'s `EvalContext`
>   (that is step 4), so a borrow keeps tests free of a global.
> - **The loader pins the header `qa`/`qb`/`scale` to the compile-time
>   `QA`/`QB`/`SCALE`** for container version 1 (mismatch ⇒ `NnueError`), and
>   the scalar forward uses the constants directly.

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
                                      // board.history.len(), which includes
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
root-offset arithmetic against `board.history.len()`. With unbucketed 768
inputs a refresh is never *forced* by a king move — that starts only with
king buckets (upgrade path), which is also when a refresh-cache (per-bucket
cached accumulator + diff) earns its keep.

Fused add+sub kernels (`acc += W0[a] - W0[s]`) handle the dominant
1-add/1-sub case in one pass over the vector; the 1-add/2-sub (capture / ep /
capture-promotion) and 2-add/2-sub (castling) variants have their own fused
kernels in the same `Kernels` seam, so the step-2 AVX2 backend covers every
update kind rather than falling back to a scalar element-wise path for a third
of moves.

> **Training-vs-inference canonicalization (review, 2026-07-12).**
> bulletformat canonicalizes each training board to the side to move
> (color-swap + vertical mirror when black is to move), which is exactly our
> inference-time `relative_to(perspective)` feature construction (§3.4.1) — so
> a net trained on that data reads our accumulators correctly. The §3.4.2
> color-flip symmetry test is the standing guard that the two constructions
> stay in agreement.

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

> **Step-3 packer note (review, 2026-07-12).** bullet emits the output bias
> `l1b` as a **single i16** at the `QA·QB` scale, whereas CENN stores
> `out_bias` as i32. The step-3 CENN packer must therefore **sign-extend**
> that i16 into the i32 field (not zero-extend, and not mis-slice a wider
> field). bullet caps `out_bias` to the i16 range at save time, so a
> sign-extended i32 CENN always carries it faithfully.

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

> **Step-2 AVX2 author notes (review, 2026-07-12).**
> - Use **unaligned loads** (`_mm256_loadu_si256`). `Aligned64` covers the
>   *stored* accumulator, but NOT the stack temporaries or the 2-aligned FT
>   weight rows — an aligned load off those would fault. The step-1 `Kernels`
>   seam now routes every update kind (1-add/1-sub, 1-add/2-sub capture/ep,
>   2-add/2-sub castling), so the AVX2 backend must supply all of them.
> - The i16-intermediate `_mm256_mullo_epi16(a, w)` trick is bit-exact vs the
>   i64 scalar reference **only while `|clamp(a)·w1|` fits i16** (real `QB=64`
>   nets satisfy this by the 3.4.3 weight clip; bound any synthetic fuzz-net
>   weights the same way, or the SIMD and scalar paths will disagree).
> - The forward pass sums first and divides by `QA` **once** at the end.
>   Integer division does not distribute over the sum, so this sum-then-single
>   `/QA` ordering is the exact spot the **step-3 vs-bullet cross-check** must
>   confirm against a real trained net.
>
> **Step-2 AS BUILT (P13 step 2, 2026-07-12; `src/eval/nnue/simd.rs`).**
> - **`Kernels` keeps SAFE `fn` pointers** — a deliberate deviation from the
>   `unsafe fn`-pointer sketch above. The AVX2 kernels are the promised
>   `#[target_feature(enable = "avx2")] unsafe fn`, but they are reached only
>   through thin **safe wrappers** installed into `KERNELS` after detection, so
>   every `unsafe` is *contained inside the `avx2` module* (localized
>   `#[allow(unsafe_code)]`, `// SAFETY:` per block — the `tt` prefetch
>   discipline, D14a/D14b). Callers (`accumulator`, `evaluate_forward`) stay
>   `unsafe`-free and the crate-wide `#![deny(unsafe_code)]` holds everywhere
>   else. The one-line cost — a non-inlined wrapper call on top of the already
>   non-inlinable `KERNELS.forward` fn-pointer indirection — is nil this step
>   (NNUE isn't wired) and a step-4 concern if ever measured. The dispatch site
>   carries the single `// SAFETY: avx2 verified at startup via
>   is_x86_feature_detected!` comment (D14b).
> - **Forward reduction is i32 lanes + i64 horizontal add.** `madd_epi16`
>   accumulates `a²·w` (SCReLU) / `a·w` (CReLU) into 8 i32 lanes; under the
>   §3.4.3 clip `|w1| ≤ 127` no lane overflows (≤ 64·255²·127 ≈ 5.3e8), and the
>   final horizontal reduction **widens each lane to i64 before summing** so the
>   worst-case *total* (≈ 4.2e9 > i32::MAX) is still exact vs the scalar i64
>   reference. This resolves reviewer B's F3: i32-lane accumulation is safe, the
>   i32 *total* is not, so only the reduction is i64. Bit-exactness (raw i64 sum
>   **and** final `Value`, all five kernels, full-range random accumulators,
>   clipped weights) is pinned by `simd::tests::avx2_bit_exact::*`;
>   `deterministic_test_net` bounds its `out_weights` to the same clip (a
>   `debug_assert`) so the accumulator fuzz + `verify_forward.sh` oracle exercise
>   the AVX2 path bit-exactly on release.

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
   **Weight-clip requirement (net-0 precondition):** the recipe must clip the
   L1/output weights to `|w1_float| ≤ 32767/(QA·QB) ≈ 1.98` (⇒ quantized
   `|out_weight| ≤ 127`) — the AVX2 `mullo_epi16` SCReLU kernel (§3.4.6) is
   bit-exact vs the scalar reference only under it, and the loader's
   `NnueError::OutWeightOutOfClip` check (§3.4.3) is the fail-fast backstop
   that rejects any net violating it.
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
2. **Dataset licensing & NNUE-data provenance**: for the *texel* tuner
   (step 7), verify the exact license of the Zurichess quiet-labeled set
   (and any lichess-derived corpus) before any copy enters the repo; if
   unclear, use download-scripts only and rely on own datagen. For the
   *NNUE* corpus, provenance is now governed by **D28 + `design/15-nnue-datagen.md`**:
   the Phase C net is bootstrapped by distilling a locally-compiled
   Stockfish (SF-labeled data = discardable dev scaffold, never shipped —
   D27/D28), and the **shipped** net is retrained from clean self-play data
   only. Doc 15 owns the training-data format contract and curation spec
   that both the bootstrap tool and the `bin/datagen.rs` generator of §3.4.7
   conform to. Needs the one-time texel-licensing check when step 7 starts;
   the NNUE side is closed by doc 15.
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

## 10. Reckless study inputs (D31, 2026-07-11)

P13 NNUE candidates from the Reckless study (`reckless_reference/11,12,13`,
summary `00-design-implications.md` §P13); ideas only, safe-Rust (D27). Mostly
confirmations of our planned machinery:
- **Confirmations** — ply-indexed flat accumulator stack + `split_at_mut(i+1)`
  adjacent-slot update (§3.4.4); Finny refresh cache; king input buckets;
  hybrid int8-FT/L1 → f32-L2/L3 quantization (Reckless `FT_QUANT=255`/`L1=64`
  == our `QA=255`/`QB=64`, §3.4.6); the SIMD bit-exactness checklist (`mul_add`,
  bias-as-initial-accumulator, fixed recursive-halving reduction order).
- **The big lever (post-v1 stretch, L)** — **threat input features** (relational
  attacker/victim inputs summed into a second accumulator; §3.4.1). Forces a
  `design/15` §3.2.3 datagen-record amendment — but NOT the raw store (D29
  regenerates any feature projection).

## 11. P13 kickoff plan — FINALIZED & GO (D35, user-approved 2026-07-12)

The concrete, ratified plan to START P13 (NNUE). Approach = D35 (inference-first,
bullet-oracled, own-net-first), sharpened by the Integral study (§design/17 §10:
get the full loop working cheap-and-early at a weak net before polishing).

**Preconditions (all TRUE as of 2026-07-12, HEAD `04ffd19`):** P9+P10+P11 complete
(engine ~2720 Elo single-core PSQT; depth-6 bench 548978). Datagen corpus building
at `data/nnue/raw/*.txt` (~23M positions and growing, SF-teacher-labeled text
lines `<FEN> | <cp> | <result>`). The `bullet` training half is VALIDATED end-to-end
(commit 241b937, `tools/nnue/TRAINING.md`, converter `tools/nnue/bf_convert.rs`,
bullet cebc78a, RTX 4060 Ti). The bench-neutral `MakeMoveHooks` seam already exists
(`src/board.rs:246`, `make_move_with`) — reserved for exactly this. Current eval is
the incremental material+PSQT `EvalContext`; NNUE slots alongside/replaces it.

**Locked choices (user, 2026-07-12):**
- **First net = SMALL.** `Chess768 → 2×256 → 1` (single perspective pair, 1 output
  bucket). Explicitly NOT the final net — smallest thing that proves the loop.
  Grow (king buckets, larger L1, output buckets) only AFTER step 4 works.
- **Data = whatever the corpus has NOW.** Train net-0 on the current SF-teacher
  corpus (no waiting for a target count — we regenerate/retrain later regardless).
- **Execution = consolidated impl agent → D33 review loop → my bench-neutrality +
  bullet-oracle gates → local SPRT vs PSQT.**
- **Reckless-net A/B = DEFERRED.** Only AFTER our own first NNUE works AND shows a
  big Elo jump vs PSQT; THEN implement the firewalled Reckless-net loader as a
  stress-test / baseline (D35 step 4-optional; AGPL net gitignored, never shipped).

**Step sequence:**
1. **Inference skeleton** — `src/nnue/` module: net loader (our own format, versioned),
   the perspective accumulator + incremental add/remove hung on the `MakeMoveHooks`
   seam, and a SCALAR quantized forward pass (QA=255/QB=64, SCReLU, scale). GATE:
   eval flag OFF ⇒ depth-6 bench bit-identical 548978 (D24); **bit-oracle the forward
   pass against `bullet`'s own reference inference** on shared positions (D35).
2. **SIMD forward pass** — AVX2 (x86-64-v3 target) + the cross-ISA bit-exactness
   checklist (design/17 §10 / mul_add, fixed reduction order). GATE: bit-identical
   to scalar.
3. **Train net-0** — the small arch above on the current corpus via `bullet`
   (tools/nnue/TRAINING.md). GATE: a net file that loads + evaluates sane cp.
4. **Wire NNUE as eval** — behind net-present; **local SPRT vs the PSQT baseline
   (04ffd19)**. GATE: **NNUE beats PSQT** ← the P13 "it works" milestone. Expect a
   large jump (NNUE is worth ~+400-700 Elo over PSQT).
5. **Iterate** — bigger arch + more/fresh data + the D36 SPSA scaffold; each change
   SPRT-gated. Then: self-play retrain for clean provenance (D28b) → the D34
   research heads (design/17). The Reckless A/B slots in here once step 4 shows big Elo.

**Owning docs:** this §11 + §3.4 (arch/quant/inference/net-format), design/15 (datagen/
bullet contract), design/16 (observation store), design/17 (research heads, later),
design/14 §11 (SPSA scaffold + cloud testing, D36). Datagen keeps running throughout.

### 11.1 Training observability dashboard (user, 2026-07-12)

A localhost browser dashboard to watch NNUE training (MLflow-style), folded into
P13. `bullet` is a Rust trainer with no native dashboard — it prints per-superbatch
loss to stdout and saves checkpoints — so any viewer needs a small (~30-line)
Python shim in `tools/nnue/` that taps that output. Tooling layer only (may use
any deps, design/16 §3.5.1; does NOT touch the zero-dep engine crate, D15).

**Python env (user, 2026-07-12): all tooling Python runs under `uv`.** The repo
is now a `uv` project (`pyproject.toml`, `uv.lock`, `.python-version` = 3.13);
tensorboard + tensorflow are already pinned there. Invoke every tooling script
with `uv run …` (e.g. `uv run python tools/nnue/gen_sf_data.py`, `uv run
tensorboard --logdir tools/nnue/runs`) so deps resolve from the locked env. This
is a tooling-only change — the engine crate stays zero-dependency (D15). New
Python deps get added with `uv add <pkg>` (updates `uv.lock`), never ad-hoc pip.

**Planned default (revisable): TensorBoard for live curves + a net-iteration
registry.** Two things, because for a chess NNUE the metric that matters is
net-vs-net Elo, NOT loss (two nets with equal loss can be tens of Elo apart):
1. **Live curves** — a shim tails bullet's stdout (train loss / val loss / LR /
   superbatch) → tfevents; `tensorboard --logdir tools/nnue/runs` on localhost:6006.
   Real-time "watch it train."
2. **Net-iteration registry** — each trained net logged with its inputs (arch, LR,
   data snapshot, superbatches) AND its outcome (final loss + **SPRT Elo vs the
   prior net**, from our local sprt.sh). A small CSV + a one-page view (or TB
   scalars). This is the real "am I getting stronger" signal across the iterative
   D28/D34 net series.

**Alternative if cross-run comparison tables are preferred over live curves:**
MLflow (`mlflow ui`, localhost:5000) — its run table natively does #2; feed the
same shim. Decide for-real at P13 step 3 (first `bullet` run) when bullet's exact
log format is in hand; the choice is swappable (both consume the same shim).

## 12. SOTA NNUE architecture (D40, user-approved 2026-07-13)

A major escalation ratified 2026-07-13: skip §3.4.1's intermediate SPRT rungs (a
deliberate deviation from D25) and build a SOTA-class net directly — **Stockfish's
exact feature set** feeding a **Reckless-shaped lean hybrid head**, trained as a
**triple-net grid** (no / small / full policy). Grounded in two source extractions
of the current SF net and a source study of Reckless (all `stockfish_reference/`
+ `reckless_reference/`, 2026-07-13). net-0..net-10 remain the validated 768→256
scaffold and a ready net-quality A/B (D39). This section is the contract; where it
differs from §3.4.1–§3.4.7 (the first net), §12 governs the SOTA net and §3.4 the
scaffold.

**Why this shape** (the decision trail): the user's four calls were (1) full 83k
features from day one, (2) bullet-extended trainer, (3) lean-to-iterate, (4) full
4096 policy head — then refined to (5) the triple-net grid and (6) **SF features +
Reckless head**. Reckless is the decisive input: a top-2 engine reaches ~3833 with
a *leaner* net than SF (768 acc, single-activation, int8-FT/f32-tail, no PSQT), and
**we already reimplemented its exact forward bit-for-bit** (`experiment/reckless-net`,
0 cp / 42 positions). So the head is proven + already coded (our own D31-clean
code → moves to main; only Reckless's *weights* stay firewalled), while the
feature front-end is the new, SF-faithful build.

### 12.1 Feature set — SF HalfKAv2_hm + FullThreats (83,248)

Two parallel feature transformers, each indexed from 0, both summing into the same
accumulator (they are NOT one 83,248-long array at runtime):

- **HalfKAv2_hm — 22,528 = 64·704/2** (`half_ka_v2_hm.h`). `PS_NB = 704` = 11
  piece slots × 64 (10 non-king + one *shared* king slot `PS_KING=640`; kings are
  features, both colors share the slot). **32 king buckets** (a 64-entry table of
  `bucket·704`, horizontally symmetric → 4 files × 8 ranks after mirror).
  `make_index(persp) = (sq ^ OrientTBL[ksq] ^ 56·persp) + PieceSquareIndex[persp][pc]
  + KingBuckets[ksq ^ 56·persp]` — oriented-square (0–63) + piece-slot + king-bucket.
  `OrientTBL` file-mirrors so the king is always on files e–h. i16 FT weights.
  Max 32 active/perspective.
- **FullThreats — 60,720** (`full_threats.h`). One feature per directed **attack
  relation**: attacker piece on `from` reaching a victim on `to`, encoded as
  (attacker piece incl. color, victim relation, from→to geometry). `map[6][6]`
  selects which attacker→victim type pairs are enumerated: kings never attack
  (whole row/col excluded), pawns only threaten P/N/R, B/R skip Q & K victims,
  N/Q include Q. Per-attacker geometry sizes (empty-board edge counts):
  P 132, N 336, B 560, R 896, Q 1456, K 420. Index = attacker-block +
  (victim-color, relation) sub-block + from-square base + ordinal-of-`to` (popcount
  of attack bits below `to`). Own `OrientTBL` (opposite a1/h1 polarity to HalfKA —
  a fidelity trap) and color-swap `^8·persp`. **i8 FT weights.** Max 128
  active/perspective; **≤80 changed per move** (x-ray ripple: sliders through
  touched squares + short-range attackers).
- **Total input = 83,248**; HashValues `0x7f234cb8` (HalfKA) / `0x8f234cb8`
  (threats) gate net-file compatibility.

### 12.2 Accumulator, refresh, and cache

- **Width 768 i16 per perspective** (Reckless-proven starting point; 512↔1024 is
  an SPRT knob). Two feature transformers add into it: HalfKA (i16 weights, seeded
  from FT bias) and threats (i8 weights, seeded from zero).
- **Refresh triggers.** HalfKA: any **own-king move** (it defines both bucket and
  mirror — SF refreshes on every own-king move, not just boundary crossings).
  Threats: **mirror flip only** (a strict subset of HalfKA refreshes), so threats
  need king-bucket-free indexing and refresh strictly less often (Reckless's neat
  asymmetry — adopt it).
- **Accumulator cache ("Finny tables").** HalfKA only, keyed by
  `(perspective, king-bucket)` (and mirror) — each slot caches a full accumulator +
  the occupancy it was built from; a refresh diffs current vs cached occupancy
  rather than rebuilding from bias. Threats have **no cache** (plain rebuild on
  mirror flip, like Reckless).
- **Incremental (non-king) updates** flow through the existing `MakeMoveHooks`
  seam (§3.4.4). HalfKA emits ≤2 sub / ≤2 add per move from `DirtyPiece`. Threats
  need a **second observer** producing up to 80 add/remove threat events per move
  (the x-ray dirty-diff) — the one genuinely heavy new piece of machinery.

### 12.3 Head — Reckless-shaped lean hybrid (int8-FT / f32 tail)

Feature-set-independent (operates on the accumulator), and already coded in
`experiment/reckless-net`. Pipeline per node:

1. **Pairwise-multiply FT activation** (adopt over §3.4.1's SCReLU): sum
   HalfKA+threats accumulators; clamp each half to `[0, 255]`; `out[i] =
   (left[i] · right[i]) >> FT_SHIFT` as u8 ∈ [0,127]. Halves 768 → 384/perspective;
   both perspectives (STM half first) → **768 u8 transformed features**. Bilinear
   (two independently-learned projections), same one-mul-per-output cost.
2. **NNZ sparse gate**: ~85–95% of the 768 u8 are zero after the clamp; skip
   all-zero 4-byte groups (movemask+LUT / VBMI2 compress).
3. **L1 768→16**: int8 sparse `dpbusd` over nonzero groups → i32, then **dequant to
   f32 once** (`× DEQUANT + bias`, clamp `[0,1]`). This is the ONLY integer layer
   past the FT.
4. **L2 16→32**: f32 affine, `mul_add`, clamp `[0,1]` (CReLU, single activation —
   no squared/dual, no concat).
5. **L3 32→1**: f32 affine, fixed recursive-halving reduction.
6. **8 output buckets** by `OUTPUT_BUCKETS_LAYOUT[popcount(occ)]` (non-uniform;
   copy the *shape* — fine at mid-material, coarse at extremes — then retune). Each
   bucket owns its own L1/L2/L3; the FT + pairwise trunk is shared.
7. **No PSQT pathway, no output-skip.** Single scalar output `× NETWORK_SCALE →
   stm-relative cp`.

### 12.4 Outer shape

The scalar cp feeds the existing §3.6 outer shape (optimism/material/rule-50). This
head has no psqt/positional split, so — exactly like net-0 — **complexity = 0** and
the §3.6 complexity blend vanishes identically; the material/optimism/rule-50
consumption is unchanged (D22). No new outer machinery.

### 12.5 The triple-net policy grid (D40)

Every training config produces **three independently-trained nets on one trunk
architecture**, differing only in the policy head, all on the **Lc0 soft-policy
target**:

| Net | Policy head | Purpose |
|---|---|---|
| **eval-only** | none | control — proves a jointly-trained policy head is not degrading eval |
| **eval+small** | rank-8 bilinear from×to (`from_emb·to_emb`, off the 768 trunk) | the in-search-affordable policy; ~83% of full-head ordering benefit (offline) |
| **eval+full** | 4096 from×to logits (off the 768 trunk) | strongest predictor; the small-vs-full gap measures the cheap head's cost |

The policy head reads the **768 transformed features** (post-pairwise trunk output,
bucket-independent), so its gradient shapes the FT+pairwise trunk — which is why
eval-only (no policy gradient) is a *true* control. Policy heads are **novel** (no
SF/Reckless precedent; D34 research). Comparison axes: eval strength (all three, by
SPRT) + policy quality (the two, by held-out top-1/target-CE).

### 12.6 Quantization + net format (CENN-v2)

Reckless's hybrid quant, which our port already matches (`FT_QUANT/QA = 255`,
`L1_QUANT/QB = 64` — already our constants):

- **FT**: HalfKA weights i16, threats weights i8, bias i16; activation clamp
  `[0,255]`, pairwise `>> FT_SHIFT (=9)` → u8.
- **L1**: weights i8, bias f32; `dpbusd` → i32 → f32 via `DEQUANT = 2^FT_SHIFT /
  (QA²·QB)`.
- **L2/L3**: all f32. *This is the key lean: only FT+L1 are integer, so the
  quantization-training surface is tiny vs SF's all-integer dual-activation tail.*
- `NETWORK_SCALE` maps f32 → cp (Reckless 380; we set to our eval scale, pinned in
  the header).

**Net file = CENN-v2** — extend the existing CENN loader (magic/version/dims/
checksum + `WrongL1`-style guards, §3.4.5) to carry: two FT weight tensors (i16 +
i8) with their dims, per-bucket i8 L1 + f32 L1-bias/L2/L3, the quant constants,
`NETWORK_SCALE`, and a `policy_kind ∈ {none, bilinear, full}` tag + its head
weights. Versioned so a v1 (768→256) loader rejects a v2 net and vice-versa.

**CENN-v2 byte layout (concrete).** Little-endian; the loader validates magic +
version + every dim + a trailing checksum, then borrows the tensors zero-copy
(page-aligned). All dims are header fields (not compile-time constants) so one
loader reads any width/bucket variant; a build still refuses a net whose dims
don't match its compiled head (the §3.4.5 `WrongL1`-style guard, generalized).

| Field | Type | Count | Notes |
|---|---|---|---|
| `magic` | u32 | 1 | `"CEN2"` |
| `version` | u16 | 1 | 2 |
| `flags` | u16 | 1 | bit0 = threats present; bits1-2 = `policy_kind` (0 none / 1 bilinear / 2 full) |
| `halfka_dims` | u32 | 1 | 22 528 |
| `threat_dims` | u32 | 1 | 60 720 (0 if absent — config C1) |
| `l1,l2,l3` | u32 | 3 | 768, 16, 32 (accumulator + tail widths) |
| `out_buckets` | u32 | 1 | 8 |
| `king_buckets` | u32 | 1 | 32 (HalfKAv2_hm) |
| `ft_quant,l1_quant,ft_shift` | i32 | 3 | 255, 64, 9 |
| `network_scale` | i32 | 1 | f32→cp (baked at training) |
| `policy_rank` | u32 | 1 | bilinear K (=8) or 0 |
| — FT weights — | | | |
| `halfka_w` | i16 | `halfka_dims·l1` | HalfKA transformer |
| `threat_w` | i8 | `threat_dims·l1` | threats transformer (present iff flag) |
| `ft_bias` | i16 | `l1` | shared FT bias (seeds `pst`; `threat` seeds 0) |
| — head (per bucket) — | | | |
| `l1_w` | i8 | `out_buckets·l2·l1` | |
| `l1_b` | f32 | `out_buckets·l2` | |
| `l2_w` | f32 | `out_buckets·l2·l3` | |
| `l2_b` | f32 | `out_buckets·l3` | |
| `l3_w` | f32 | `out_buckets·l3` | |
| `l3_b` | f32 | `out_buckets` | |
| — policy head (iff `policy_kind≠none`) — | | | |
| `pol_trunk_w/b` | f32 | policy trunk off the 768 features | shape per kind |
| `pol_from_emb`,`pol_to_emb` | f32 | `64·K` each | bilinear (kind=1) |
| `pol_head_w/b` | f32 | `…·4096` | full (kind=2) |
| `checksum` | u32 | 1 | over all preceding bytes |

The FT is **two separately-quantized tensors** (`halfka_w` i16, `threat_w` i8) —
matching both the trainer's two-summed-affine-tensors export (§12.7) and the
inference head's two accumulators `pst`/`threat` (§12.3). The head block (`l1_w`
… `l3_b`) is exactly `SotaHead`'s fields (`src/eval/nnue/sota_head.rs`), so the
loader fills that struct directly. `network_scale` is per-net (not the const 380
Reckless baked). **Open item to reconcile with the trainer:** `bullet` writes its
own `quantised.bin` layout; the packer (`tools/nnue/`) transcodes that → this
CENN-v2 order at save time (task #28 reports bullet's exact tensor order; the
table above is the loader-side contract that transcode must hit).

**Optional PSQ1 trailer (T6c FT PSQT skip, 2026-07-23; authoritative contract:
`jax/archon_jax/checkpoint.py`).** After the checksummed payload a CENN-v2 MAY
carry `b"PSQ1" | u32 num_buckets (= out_buckets) | u32 rows |
i32[rows·num_buckets]` row-major, `rows = halfka_dims + threat_dims` (HalfKA
rows first; a no-threats net carries only the HalfKA rows). Rows are indexed by
the engine's canonical feature indices (HalfKA `f` → row `f`, threat `t` → row
`halfka_dims + t`). Stored values fold in `0.5·400·256`, so the head adds the
unclamped linear channel `(psqt_stm[b] − psqt_ntm[b]) >> 8` cp (arithmetic
shift) using its existing output bucket `b`. The accumulators (HalfKA stack +
Finny cache, FullThreats stack) maintain the per-perspective `[i32; 8]` bucket
sums incrementally over the same feature events. The section rides outside the
header checksum, mirroring its position after the padded bullet quant stream
(`netpack.py` passes it through verbatim); nets without it — and engines
predating it — are byte-identical/bit-identical to pre-T6c behavior.

### 12.7 Trainer — bullet, extended

bullet natively covers HalfKA + output buckets + pairwise + the mixed int8-FT/f32
tail. The custom work:

1. **FullThreats input feature** — a bullet sparse-input type emitting the 60,720
   threat indices per position (bullet supports custom Rust input features).
2. **Configurable policy head** — none / bilinear / 4096, off the trunk, with a
   **combined loss** = eval (WDL+MSE blend, §3.2/§15) + `λ_policy ·` policy
   cross-entropy to the Lc0 distribution. Three build configs from one trunk.
3. **Data loader** for the extended record (§12.8).

Output = CENN-v2 nets our loader owns. (nnue-pytorch was declined: no policy path,
SF-format nets we'd re-quantize.)

> **Feasibility spike verdict (2026-07-13, bullet `cebc78a`).** GO, no fork needed —
> everything routes through public APIs. **The eval trunk is entirely native**:
> `ChessBucketsMirrored` (king-bucketed+mirrored input), `pairwise_mul()`, per-tensor
> quant (`SavedFormat`/`QuantTarget` → FT `i16(255)`, L1 `i8(64)`, L2/L3 `Float` — the
> exact D40 lean), and `OutputBuckets`/`MaterialCount<8>`. A net in this shape trained
> end-to-end on GPU + emitted a valid mixed-quant checkpoint (overflow hard-checked at
> save). **HalfKA i16 / threats i8 as two separately-quantized tensors** = two affine
> weight tensors summed in the graph (`node + node`) — which is exactly what our
> inference wants, since the head keeps `pst`/`threat` as *separate* accumulators (§12.3).
> **The two custom pieces:** (1) SF-exact HalfKAv2_hm + FullThreats via a **custom
> `SparseInputType`** (one combined impl over the concatenated `0..83,248` index space);
> bullet wiring is ~½–1 day, the index math is the engine-domain work of §12.9 (shared
> with inference). (2) The **policy head + combined loss** drops off the eval-only
> convenience `ValueTrainer` to the **low-level `ModelDefinition` + `Optimiser` +
> `run::train`** path (which `ValueTrainer` builds internally, so it's a known-good
> template): `matmul` for from×to, `softmax_crossentropy_loss`, operator-overload loss
> combine, a named `policy_target` data input; ~3–6 days, and **the gating risk is the
> policy-target corpus/`DataReader` (§12.8, task #27), not the graph.** So the build
> order stands: #27 (data) gates #28 (train); the eval-only nets are low-risk, the
> policy nets ride the low-level path.

### 12.8 Data-format extension — eval + policy per position (the curation work)

The single new corpus requirement (the user's "focus on data curation"): every
record carries **both** an eval/WDL label **and** the Lc0 policy target, so all
three nets train from one corpus (eval-only ignores the policy column).
`bulletformat::ChessBoard` (32 B) cannot hold a policy distribution, so either a new
fixed+sparse record (board + eval + wdl + a capped list of `(move, prob)` pairs) or
a parallel policy-target stream keyed by position. Lc0 data already has the
policy[1858] vector → the converter (`data/lc0_nnue/`) emits both eval and policy;
SF-teacher data (design/15) supplies eval + a best-move (a degenerate one-hot
policy) for the eval-only and as a mix. Curation/filter rules (§15 §3.5) extend to
keep only positions with a usable policy target for the policy nets.

### 12.9 Inference plan — reuse the Reckless head, build the SF front-end

- **Head (reuse):** port the `experiment/reckless-net` forward (pairwise-mul +
  NNZ-sparse int8 L1 + f32 L2/L3 + 8 buckets) to `src/eval/nnue/` on main — it is
  our own code, bit-exact against a real SOTA-shaped net. Only the L1 input width
  changes if the accumulator ≠ 768.
- **Front-end (new):** HalfKAv2_hm + FullThreats index computation; the 768-wide
  two-transformer accumulator with the HalfKA Finny cache + threats mirror-rebuild;
  the threats dirty-diff observer (≤80/move) on the `MakeMoveHooks` seam. This is
  the bulk of the new engine code.
- **Adopt from Reckless** (D40): pairwise-mul activation, int8-FT/f32-tail hybrid,
  the `DEQUANT = 2^shift/(QA²·QB)` bridge, the bit-exactness checklist
  (`f32::mul_add` where SIMD uses FMA; bias-seeded accumulators; one fixed
  reduction order; explicit scalar clamps), and — for the D16 fast build only —
  compile-time cfg-module SIMD dispatch (no runtime fn-pointer indirection),
  keeping the `LazyLock<Kernels>` path for the portable default.
- **Parity gate:** the Rust forward must match the bullet-side forward to a fixed
  tolerance across a fixture set (the §3.4.6 scalar/SIMD discipline, now also
  Rust-vs-trainer), before any strength claim.

#### 12.9.1 Incremental FullThreats update (the ≤80/move dirty-diff observer)

The one genuinely heavy new machine (study 2026-07-13, from SF + Reckless — never
copied). A move produces a raw list of perspective-agnostic
`ThreatDelta{attacker, from, victim, to, add}` events, **built once** in the
make/unmake hook and **resolved per perspective** at eval time via the same
`threat_index` the from-scratch path uses (excluded pairs return the sentinel and
are dropped — so the enumeration may safely over-generate). Adopt **Reckless's
structure, SF's content** (kings excluded as attacker/victim; pawn-push relations
included — already true of our from-scratch `pawn_push_or_attacks`).

**Core primitive `record_square_touch(P, s, add, occ)`** — emit every relation a
piece `P` conceptually appearing/disappearing at `s` participates in:
1. **outgoing**: `piece_attacks(P, s, occ) & occ` → a `(P, s, victim, t, add)` per
   occupied target (pawns use `pawn_push_or_attacks`).
2. **x-ray ripple**: for each slider `S` (bishop/rook/queen) attacking `s`, the
   first occupant strictly beyond `s` on the `S→s` ray = `ray_pass(S.sq, s) & occ &
   slider_attacks(s)` (**at most one** — `ray_pass` at `src/attacks.rs:396` is
   exactly this "discovered-attack" query). Emit that discovered relation with the
   **sign flipped** (`!add`: removing `P` *unblocks* `S`), plus `S`'s direct
   relation to `P@s`.
3. **incoming**: short-range attackers of `s` (pawns via the color-symmetry
   `pawn_attacks(!c, s) & c_pawns`; knights) → `(A, victim=P, s, add)`.
   Kings are **not** emitted as attackers (SF). For `P`=king, run **only** the ripple.

**Three drivers**, called from the mutation points (board already reflects the move):
- `on_move(P, from, to)`: one shared occupancy `occ ^ to` for both
  `record_square_touch(P, from, false)` and `(P, to, true)` — this makes the
  along-a-ray discovered victim cancel (equal index, opposite sign), Reckless's
  elegant replacement for SF's `noRaysContaining` guard.
- `on_change(P, s, add)`: full touch (used for captures' removed piece, ep pawn,
  castling rook).
- `on_mutate(old, new, s)`: relabel `s`'s relations old→new **with no ripple**
  (occupancy at `s` unchanged) — promotions and the captured→mover swap.
  Per-move-kind wiring: quiet=`on_move`; capture=`on_change(mover,from,false)` +
  `on_mutate(victim,mover,to)`; ep=`on_move`+`on_change(pawn,capsq,false)`;
  promo=`on_move`+`on_mutate(pawn,promoted,to)`; castling=rook `on_change`×2 + king
  `on_move` (two ripples). A simpler first-cut (correctness A/B): `on_change` every
  `DirtyPiece` sub (removes first) then every add — auto-cancels, costs extra events.

**Buffers** (stack, heap-free): raw deltas `[ThreatDelta; 128]` (the `(8+16)·3+8 =
80` bound + pad; debug-assert `len ≤ 80`); per-perspective `added`/`removed` index
lists `[u16; 128]` each. **Undo** swaps `added↔removed`.

**Refresh trigger**: a threat index depends on the perspective's king **mirror half
only** (not the bucket), so a perspective's threat accumulator fully refreshes when
*its own* king crosses the d/e boundary — a strict subset of the HalfKA refresh, so
**piggyback: refresh threats whenever the PST (HalfKA) accumulator refreshes**
(superset-safe; a separate mirror flag is a later optimization). No Finny cache for
threats (mirror flips are rare) — reuse the from-scratch `append_active`.

**The correctness gate** (non-negotiable): a random-game replay test asserting
`incremental_threat_accumulator == from_scratch_refresh` at *every* node — the same
make/unmake-equivalence discipline as `assert_consistent`. The incremental
enumeration and the from-scratch `append_active` are two encodings of one feature
set; any content choice (kings, pushes, queen-victim rules) is made in lockstep in
both, or this test fails.

### 12.10 Build plan / phasing (full 83k from day one, lean head)

1. **Trainer** (§12.7): bullet FullThreats feature + configurable policy head +
   combined loss + extended data loader; produce the first eval+policy corpus and
   train the first triple (eval-only / small / full).
2. **Inference** (§12.9): port the Reckless head to main; build the SF feature
   front-end (features + accumulator + Finny cache + threats observer); CENN-v2
   loader; the quantized forward; the Rust-vs-trainer parity gate. All behind a
   feature/`EvalFile`-versioned path — default build + depth-6 bench stay neutral
   until ship.
3. **Screen** (D39): fixed-node SPRT of all three nets vs net-8 (best scaffold) to
   isolate net quality from inference speed; the eval-only ≥ eval+policy check
   (D40 control).
4. **Ship gate** (D39): real-TC SPRT of the winning eval net vs net-8; then the
   policy nets' in-search value (ordering/pruning) tested separately (the policy
   line is a research appendage — the eval-only net is shippable on its own).
5. Accumulator width / bucket count (§12.2/§12.3) and `λ_policy` are SPRT/SPSA
   knobs tuned after the loop is green (D39 rule f: tune on the shipped net).

### 12.11 Acceptance criteria (extends §8)

- **Parity:** Rust forward == trainer forward within tolerance on the fixture set;
  scalar == SIMD bit-exact (§3.4.6).
- **Bench neutrality:** default build byte-identical, depth-6 fingerprint 548978,
  until the net is shipped as the default eval.
- **Feature fidelity:** HalfKAv2_hm + FullThreats indices reproduce the SF
  reference index math on a cross-check set (the two OrientTBL polarities, the
  threat LUT arithmetic).
- **Strength:** fixed-node SPRT (screen) then real-TC SPRT (ship gate) vs net-8;
  the D40 control (eval-only not worse than eval+policy on eval) holds or the
  regression is attributed.
- **Provenance:** every net → a ledger row (D38) with arch tag, data snapshot,
  policy_kind, nps, and SPRT Elo.
