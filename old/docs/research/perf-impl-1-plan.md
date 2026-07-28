# perf-impl-1 — the plan, written before the implementation

Date: 2026-07-27. Branch `perf-impl-1-threats`, worktree
`.claude/worktrees/perf-impl-1`. Mandate: implement the top three
recommendations of perf study #7. This file is the plan and the record of what
was **verified** rather than assumed, written before any engine code changed.
The results live in `perf-impl-1-<slug>.md`.

## 0. What I verified first (all reproduced by me, none taken on trust)

| claim | source | my check | verdict |
|---|---|---|---|
| study binary + net identity: 800-FEN eval hash `dd08810005c92f9e…` | study §10 | reran `nnue-eval` on `archon-d47v-pgo` | **reproduced exactly** |
| wl50/d13 node total 2,118,123 | study §10 | reran the pinned bench | **reproduced exactly** |
| HEAD (783d142) still reproduces both | — | fresh `sota,arch-d47v` native build of the worktree | **reproduced exactly** — so the cuckoo commit is behaviour-neutral and I have a clean A/B baseline |
| study §3.2 threat Ir/node split | study §3.2 | own callgrind, `bench 16 1 8 wl50`, exact exclusive Ir | **confirmed within 6%** (mine 344/161/2540/657 vs 323/159/2354/638) |
| depth-6 bench pin 385329 gates item 2 | mandate | read `Cargo.toml`: there is **no `default` feature set**, so the pinned bench binary is the **PSQT** build and `sota_threats.rs` is not compiled into it | **FALSE as a gate for item 2** — see §1 |

## 1. The bench pin does not gate item 2 (correction to the mandate)

`tools/check.sh` runs `./target/release/chessengine bench 16 1 6 default depth`
on a **default-feature** build. `Cargo.toml` has no `default = [...]`, and the
entire SOTA NNUE path (including `sota_threats.rs`) sits behind the `sota`
feature. So the depth-6 bench pin is a PSQT-eval number: item 2 cannot move it,
and its staying at 385329 is *necessary but not sufficient* evidence.

The real equivalence gates for item 2 are therefore:

1. `nnue-eval` sha256 over the 800-FEN battery = `dd08810005c92f9e…`
   (bit-exact eval, every position, both perspectives, all buckets);
2. `bench 16 1 13 wl50.fens depth` = **2,118,123** nodes (node identity through
   a real search, 50 positions);
3. the existing per-node `assert_matches_refresh` oracle under
   `debug_assertions`, over the whole test suite and a debug search;
4. `cargo test --release --features sota -- --ignored threats_replay_large`
   (thousands of random games, incremental == refresh at every node);
5. and yes, 385329 as well.

Item 3 *is* gated by 385329, because it touches default-build code.

## 2. Item 2 — what the measurements say, and why the study's framing is wrong

### 2.1 Our threat delta is already minimal

New instrument (`measure_fold_row_counts`, committed): for each move it computes
the true minimal delta — the symmetric difference of the from-scratch active
threat feature sets before and after — and compares it with the number of rows
our fold actually touches.

```
perspective-plies      : 76651
resolved rows / p-ply  : 5.99
MINIMAL rows / p-ply   : 5.80
overfold factor        : 1.034
```

**There is no 3× row surplus.** The hypothesis that our fold is expensive
because we generate too many events is refuted. Any correct incremental scheme
must fold essentially the same rows we already fold.

### 2.2 `at_full` is 15.7% of cycles because the fold is inlined into it

Study §7 option 2 reads "`at_full` is 15.7% of deep cycles; SF's equivalent
stage costs ~0". That conflates two things the study's own §3.1 already shows:
`at_full`'s top opcodes are `vpmovsxbw` 21.2 / `vpaddw` 9.1 / `vpsubw` 8.6 —
that is the **fold kernel**, inlined by PGO into the symbol. §3.2 of the same
study gives the honest split: `at_full` self is 638 Ir/node against
`i8_deltas_into` 2,354.

I confirmed this directly by putting `#[inline(never)]` on
`apply_i8_deltas_into` and re-profiling the native build:

| symbol | share of instructions (native, non-PGO, `bench 16 1 200000 wl50`) |
|---|---:|
| `apply_i8_deltas_into` (the fold) | **6.28%** |
| `at_full` (plumbing + `build_deltas` + resolve + PSQ1 fold) | 6.18% |
| `record_square_touch` | 3.82% |
| `ripple` | 0.65% |
| `apply_at` (the snapshot) | ~0.70% |
| `refresh_threat` (king cross) | ~0.56% |
| **threat block total** | **≈18.2%** |

Of `at_full`'s 6.18%, ~1.3pp is the PSQ1 skip-channel fold (`psqt_add`/`psqt_sub`
row loads), which is not replay either.

So the **addressable replay surface is ≈10.0% of instructions**, not 15.7% of
cycles; and the fold (≈7.6pp with PSQ1) is irreducible in any scheme.

### 2.3 What the search actually does (throwaway counters, 201,815 nodes, d8 wl50)

```
push 202,707    at_full 83,005 (0.411/node)   compute_work 82,946
span_children 119,331  (1.44 children folded per compute; 24.1% of computes span >1 ply)
persp_folds 233,219    add_rows 407,889  sub_rows 545,189  => 4.09 rows / perspective-fold
raw_events 521,724     => 4.37 raw events per generated child
king refreshes 4,597 (2.3% of pushes)   one-sided folds 5,443 (4.6% of children)
```

Cross-validated: `persp_folds − rows_hist[0] = 233,219 − 51,298 = 181,921`,
exactly the callgrind call count for `i8_deltas_into`. Two independent
instruments agree to the unit.

Note `span_children/compute = 1.44`, not ~1.06 as the study's call ratios
suggest — spans of 2+ plies are 24% of folds. That matters: it means an
"only ever fold the live position" design is not available.

### 2.4 Therefore: the scheme is not the problem, the *representation* is

Our incremental scheme is already (a) minimal in rows, (b) lazy (only 46% of
pushed plies are ever folded), (c) forward-incremental along the search path
from the nearest computed ancestor. SF's forward/backward hop machinery solves
a problem we do not have.

What we actually pay for is that event generation resolves **piece identity
through a mailbox, one square at a time**, because the deferred design snapshots
a mailbox rather than bitboards:

- `record_square_touch` asks "which sliders attack `s`?" by iterating every
  occupied square on the rook and bishop rays and calling `Replay::resolve`
  (mailbox load + `Option` unwrap + linear scan of an 8-slot override array)
  and then a type `matches!`. Up to 8 identity resolutions to find 0–2 sliders.
- Same for knights (`knight_attacks(s) & occ`, then resolve + type test each)
  and pawns (two masks, resolve + colour + type test each).
- The from-scratch enumeration `for_each_threat` does none of this: it already
  filters by **victim-type bitboard masks** (`pawn_targets`,
  `minor_slider_targets`, `queen_targets`).

Line attribution inside `record_square_touch` supports this: `option.rs` alone
is 23.7% of the symbol, the attack-table `LazyLock` guard (`atomic.rs` +
`once.rs`) another 6.4%.

### 2.5 The design I will implement

**Replace the mailbox snapshot with a bitboard one, and make generation
bitboard-filtered — mirroring the from-scratch enumeration exactly.**

```rust
struct ThreatPosition {          // 72 B, replaces the 138 B mailbox snapshot
    by_type: [Bitboard; 6],
    by_color: [Bitboard; 2],
    kings: [Square; 2],
}
```

- `Replay` collapses to a mutable `ThreatPosition`: an edit is 2 XORs, so the
  `over_sq`/`over_pc`/`find()` override machinery disappears entirely.
- Slider discovery becomes `ratt & (rooks | queens)` and
  `batt & (bishops | queens)` — one AND each, replacing up to 8 mailbox
  resolutions and type tests.
- Knight/pawn discovery becomes `knight_attacks(s) & knights`,
  `pawn_attacks(!c, s) & pawns & by_color[c]`.
- Outgoing victims iterate per victim type over `att & by_type[vt]` for the
  types `threat_pair_kept` keeps, so both the mailbox resolve and the
  per-event `threat_pair_kept` test vanish.
- Hoist the `attacks::tables()` `LazyLock` deref once per `build_deltas`.

**Why this is bit-exact.** The fold is wrapping i16 add/sub and the PSQ1 fold is
wrapping i32 add/sub, both commutative and associative, and each raw event
resolves to an index independently of the others. So the accumulator depends
only on the **multiset** of resolved indices, never on their order. I am free
to reorder and regroup generation as long as the multiset is preserved. (I
verified there is no chunking hazard: `i8_deltas_into` chunks at CAP=96 and the
observed maximum is 26 rows.)

**Honest expected gain: +2.5 to +4% NPS**, not the study's +5–8%. I will report
the measured number whatever it is, and I will not claim the study's target.

### 2.6 What I will NOT do, and why

- **I will not build SF-style backward hops.** Nothing in our profile is the
  problem they solve. Adding a second update direction doubles the correctness
  surface of the single most dangerous subsystem in the engine (a missed update
  is a silently wrong eval) for no measured gain.
- **I will not remove the replay path.** It stays as the `debug_assertions`
  oracle, as the mandate requires, and as the from-scratch `refresh_threat`
  used on king crosses.
- **I will not touch the fold kernel.** It is 7.6pp and already tuned.

## 3. Item 3 — icache / code-size diet

Must be node-identical on the default build (bench 385329) and is measured by
icache misses, not wall clock. Plan: measure the static instruction counts of
`next_move` and `search::<NonPv>` in *our own* builds first (the study's numbers
are from a PGO binary at a different commit), find what the size is made of
(`objdump` + source-line attribution), and apply only changes that provably do
not move the node count.

## 4. Item 1 — hka2048 audit

Audited under a separate work item; findings and fixes are in the results
report. No launch: the GPU is occupied by `archon-d48x` until ~21:30.

## 5. Protocol

- Everything `nice -n 19`, `CARGO_BUILD_JOBS=2`, one fat-LTO build at a time.
- Conclusions carried by **deterministic** instruments: `instructions:u` totals,
  callgrind exclusive Ir, node counts, eval hashes. Cycles and NPS are reported
  as bands from interleaved pinned rounds, never as the basis of a claim.
