# ARCHON perf study #35 — the `DirtyPiece` duplication, deleted from the OTHER side of the make

**Landed:** `DirtyPiece::after_move` — the accumulator's placement record is now
derived from the board the move PRODUCED, not the one it was made in.
**−25.0 instructions/node (−0.442%), −3.02 branches/node (−0.513%), −0.30 to
−0.49% L1d load ops** across three protocols whose byte-identical nulls sit at
−0.06% to +0.02%. Node-identical (5,001,562) and eval-hash-identical
(`5ca978e012da1bb0`), D24 perft ritual and `CHECK_NNUE_GATE=1 tools/check.sh`
both green.

**Retracted, in the useful direction:** perf study 13 §5.1 rejected fusing the
derivation into `make_move_with` at **+41.5 i/n**. Rebuilt against today's
engine that arm measures **−11.36 i/n** — a win, not a loss. The rejection does
not reproduce. It is still not the change that landed, because the *other* way
of removing the same duplication is worth 2.2× as much.

**And the instrument failed, measurably.** The commission's gate was cycles.
Three protocols later — sequential-rotated, SMT-sibling-pinned, and
simultaneous-on-two-cores — a **byte-identical copy of the control** still sits
+0.38% to +5.27% away from it, against an arm whose mechanism predicts ~0.3–0.4%.
`cycles:u` on this box, with the D49 trainer live, cannot resolve this change,
and §3 is the evidence rather than the excuse.

---

## 0. HEADLINE

### 0.1 The arms

All five binaries: release LTO / CGU=1, `-C target-cpu=native`,
`--features sota,arch-d48x`; frame
`bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn`
= 50 FENs × 100k nodes = **5,001,562 nodes**, identical to studies 29/31/33 so
the numbers compose with theirs.

| arm | what it does | i/n | Δ i/n | br/n | Δ br/n | loads Δ% (paired) |
|---|---|---:|---:|---:|---:|---:|
| `base` | main @ `5e0393e` | 5659.760 | — | 589.06 | — | — |
| **`after`** | **record derived AFTER the make, from `state.moved`/`state.captured`** | **5634.760** | **−25.000** | **586.04** | **−3.02** | **−0.30 … −0.49%** |
| `passmover` | record still derived before the make, but the mover threaded in from `do_move`'s existing `moved_piece` argument | 5640.760 | −19.000 | 588.06 | −1.00 | −0.32% |
| `hookc` | study 13 §5.1 rebuilt: record built inside `make_move_with` from its own step-3 actors, handed out through Hook C | 5648.400 | −11.360 | 586.17 | −2.89 | −0.04% |
| `after2` | `after`, with the en-passant pawn CONSTRUCTED rather than read out of `state.captured` — identical output, no `expect` | 5636.760 | −23.000 | 586.04 | −3.02 | −0.15% |

`i/n` is exactly reproducible — the control returned **5659.760 on all 14
rounds**, a 0.000% null, which is why a 25.000 difference is quotable at all.
Node total and wl50 eval hash are **identical on every line**: these are
refactors of how the same record is derived, and if either had moved nothing
else in the table would be comparable.

### 0.2 The six findings

1. **The duplication is real and it is cheapest to delete from the far side.**
   `DirtyPiece::from_move` re-derived three facts `make_move_with` derives for
   its own Zobrist work — the side to move, the mover (`piece_on(from)` plus a
   panic edge), the victim (`piece_on(to)`). Two of them the make already
   *stores*, into the child `StateInfo` as `moved`/`captured`, put there by perf
   study 14 §4.4 so `unmake_move` would not have to re-read them. Reading that
   store forward instead of the mailbox backward is the whole change.
2. **Perf study 13's +41.5 i/n rejection does not reproduce (§4).** The same
   fusion, rebuilt, is −11.36 i/n today. Two things changed underneath it:
   study 14 removed the `mem::take`/224-byte-temp shape whose register pressure
   was 13's stated mechanism, and this rebuild unifies the normal and
   en-passant arms through `(captured, capsq)`, which the make computes anyway.
3. **…and it is still the wrong shape.** `hookc` wins 11.4 i/n but moves L1d
   loads by −0.04% — nothing. `after` wins 25.0 i/n and −0.30…−0.49% of loads.
   Deriving from registers inside the make deletes *instructions*; deriving from
   the make's stores deletes *loads*, and study 33 §5.3 says loads are what this
   site is priced in.
4. **The commissioned currency could not be measured, and I can prove it (§3).**
   Byte-identical control-vs-control, paired within round: **+1.585%** and
   **+1.650%** (sequential, n=14), **+5.266%** (SMT-pinned, n=4 — and
   confounded, §3.3), **+0.383%** (simultaneous two-core, n=6 pairs). The
   position-in-round artifact that nearly shipped study 29 is present and worth
   **9.4%** here (§3.2).
5. **One protocol bug found and retracted in flight (§3.3):** pinning a spinner
   to the measurement CPU's SMT sibling is a good idea, but I ran the spinner
   from `bin35/base` — and SMT siblings share L1I and L2 on Zen 4, so the
   control was measured with its own text pre-warmed by an identical process and
   every other arm was not. Its byte-identical null read +5.27%.
6. **Study 13 §5.5's "the cost of a change is only weakly related to the work it
   does" reproduced, twice, by accident (§5.2).** Respelling four lines *inside
   a `#[cold] #[inline(never)]` en-passant arm* moved the hot path by
   **+2.0 i/n** — exactly, deterministically, and far more than that arm's own
   instructions could account for (en passant is well under 1% of made moves).
   And adding fourteen lines of DOC COMMENT changed the binary's sha256, because
   panic `Location` constants carry line numbers.

---

## 1. THE SITE, AND WHY IT WAS NEXT

Perf study 33 localised `search::<NonPv>` (19.90% of the profile) and found
that only 32.87% of the PGO symbol is search's own code. The largest inlined
component is **`do_move` at 26.12% / ~198.4 cycles/node**, and study 33's §9
named this exact item as the one it would run next:

> the biggest single addressable block is §4's `do_move` … whose two hottest
> lines are both `DirtyPiece::from_move` deriving mover, victim, ep square and
> castling rook geometry from the pre-move board immediately before
> `make_move_with` re-derives the same facts.

Study 33 §5 also fixed what kind of intervention is allowed. The symbol's share
of engine stall cycles (8.80%) equals its share of retired load ops (8.84%) to
two decimals, while it holds 1.69% of the engine's misses and 91.4% of its loads
hit L1. **Locality interventions are forbidden; reducing the NUMBER of loads is
licensed, with a linear payoff.**

That is the mechanism this study was commissioned against, and it is the one
that decided between the two shapes in §0.1.

---

## 2. WHAT LANDED

### 2.1 The change

`Worker::do_move` used to be:

```rust
let dirty = self.eval_ctx.nnue_active()
    .then(|| DirtyPiece::from_move(&self.board, m));   // pre-move board
if let Some(dp) = dirty { self.eval_ctx.push_dirty(dp, ply); }
self.board.make_move_with(m, gives_check, &mut TtPrefetch { tt });
#[cfg(feature = "sota")]
if let Some(dp) = dirty { self.eval_ctx.push_sota_dirty(dp, ply, &self.board); }
```

and is now:

```rust
self.board.make_move_with(m, gives_check, &mut TtPrefetch { tt });
if self.eval_ctx.nnue_active() {
    let dp = DirtyPiece::after_move(&self.board, m);   // post-move board
    self.eval_ctx.push_dirty(dp, ply);
    #[cfg(feature = "sota")]
    self.eval_ctx.push_sota_dirty(dp, ply, &self.board);
}
```

`DirtyPiece::after_move` takes the mover from `board.state.moved` and the victim
from `board.state.captured` — the two fields step 11 of the make writes from the
values it already had in registers — and the side to move from the mover's own
`color` field rather than the board's. Everything else a record needs is either
in `m` (from / to / kind / promotion piece) or in `castling_rook_square`, which
no move mutates, so the post-move board carries the whole derivation.

**Nothing in the moved block wanted the pre-move board.** `push_dirty` takes no
board at all, and `push_sota_dirty` deliberately wants the POST-move one (its
threat x-rays read the child). The comment at the old call site said the
derivation "must run before `make_move_with`" — true of the derivation as it was
written, not of the information.

### 2.2 Why this is a load deletion and not a code move

| fact | before | after |
|---|---|---|
| side to move | `board.side_to_move` (load) | `mover.color` (register) |
| mover | `piece_on(from)` → `mailbox[from]` (load) + `Option` test + `expect` panic edge | `board.state.moved`, a store the make just made |
| victim | `piece_on(to)` → `mailbox[to]` (load) | `board.state.captured`, likewise |
| null-move sentinel | `if !m.is_ok()` — dead in the search, which never makes a null move through here | `debug_assert!` |

The two `StateInfo` reads are store-to-load forwards, not memory traffic, because
`make_move_with` is inlined into `do_move` which is inlined into `search::<NT>` —
which is exactly why study 33's `do_move` frame exists as an inline frame at all.

### 2.3 Correctness

`DirtyPiece::from_move` stays in the tree even though the search no longer calls
it (its remaining callers are all `#[cfg(test)]`), because it is now the
**independent oracle**: a new test walks depth 3 from six roots chosen to reach
every move kind and asserts

```rust
assert_eq!(DirtyPiece::from_move(pre, m), DirtyPiece::after_move(post, m));
```

field for field. A whole-record `assert_eq!` pins the EMISSION ORDER, not just
the multiset — the accumulator folds these in sequence and the Finny backward
walk indexes `subs()` positionally. The test asserts per-kind coverage
(`seen[kind] > 0` for all four) so a future refactor that stops reaching en
passant fails rather than silently testing less.

Gates run before landing:

* `cargo test` — green (the fast perft tier is in it).
* `cargo test --release -- --ignored perft_full_suite` — green (D24 ritual;
  this is board-mutation-adjacent code).
* `cargo test --release -- --ignored perft_extended` — green (~18 G nodes,
  61.1 s). Not strictly owed: `make_move_with` and `unmake_move` are
  byte-identical to their previous form and the whole `board.rs` diff is two
  additions inside `impl DirtyPiece` plus a test. Run anyway because the change
  reads state that make/unmake maintain.
* `cargo clippy --all-targets` — clean.
* `bash tools/check.sh` — **`Bench: 587683` reproduced bit-for-bit.**
* `CHECK_NNUE_GATE=1 bash tools/check.sh` — eval hash
  `5ca978e0…8b38` and d13 node total `1922947`, both unmoved.

---

## 3. THE NULL, WHICH FAILED — three protocols

The commission's instrument rule was: gate on cycles, median of ≥6 rounds,
paired within round, null first. The null ran first and it kept failing. This
section is what it cost to establish that, because the next study should not
pay for it again.

### 3.1 Protocol 1 — sequential, rotated, `nice 19`, CPU 2 (n=14)

Six labels per round: `base`, **`null`** (a byte-identical `cp` of `base`),
**`null2`** (a *source-perturbation* null — step 3's `captured`/`capsq` branch
polarity inverted, which LLVM canonicalised into a **byte-identical binary**;
all three have sha `2281a79b0ca816c7…`), plus the three arms. Round order
rotated. `docs/research/perf-artifacts-35/panel.txt`.

| | i/n Δ% | **cyc/n Δ%** | loads/n Δ% | stall/n Δ% |
|---|---:|---:|---:|---:|
| `null` (byte-identical) | +0.000 | **+1.585** [−11.30, +9.14] | −0.060 | +2.107 |
| `null2` (byte-identical) | +0.000 | **+1.650** [−27.07, +19.69] | +0.014 | +2.699 |
| `after` | −0.442 | +3.845 | **−0.491** | +6.520 |
| `passmover` | −0.336 | +3.612 | −0.324 | +5.659 |
| `hookc` | −0.201 | +0.849 | −0.044 | +1.313 |

The control's own round-to-round spread: `i/n` **±0.000%**, `cyc/n`
**±20.341%**, `stall/n` **±30.591%**, `loads/n` ±0.829%.

Two binaries that are the same bytes disagree by 1.6% in the paired median.
Every arm reads *slower* than the control, including the ones that provably
execute fewer instructions. This is rule 1 of the standing method: the
instrument cannot answer the question.

A min-over-rounds reduction (the right estimator when interference is one-sided)
does not rescue it — it puts the byte-identical nulls at **+4.86%** and
**+6.08%**, i.e. the control simply drew one lucky round.

### 3.2 The position-in-round artifact is real and worth 9.4%

`cyc/n` pooled over labels, by position within the round:

| position | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| median cyc/n | 5227.8 | 5220.5 | **4776.6** | 4842.8 | 4888.0 | 5034.2 |

A 9.4% swing from position alone. This is the effect that nearly shipped study
29's −1.67%; rotation covers it in expectation, but at n=14 each label visits
each position 2.3 times, and the residual is comparable to everything this study
is trying to see.

### 3.3 Protocol 2 — SMT sibling pinned, `nice 0` — **RETRACTED, my bug**

Two diagnoses from protocol 1: the measured run was at `nice 19` (maximally
preemptible by the noise it is trying to see through), and nothing occupied CPU
2's hyperthread twin, so each run rolled a die on whether it shared the core.
Protocol 2 fixed both — measurement at `nice 0`, and a spinner pinned to CPU 10
for the whole panel so SMT contention is constant instead of bimodal.

The spinner ran **`./bin35/base`**. On Zen 4 the two threads of a core share
L1I, L1D and L2, so the control was measured with its own text and rodata
pre-warmed by an identical process on the sibling, and no other arm was. Its
byte-identical null read **+5.266%**.

The panel was killed and the numbers are not used for anything. I did not run
the control experiment (spinner from a third, never-measured copy), so the
text-sharing mechanism is inferred from the shape of the failure, not proven.
**The idea is still right; the spinner must be a binary that is never on the
measurement list.**

### 3.4 Protocol 3 — the two arms run SIMULTANEOUSLY on two cores (n=6 pairs)

The reason a sequential design cannot hold the box constant is that `cycles:u`
moves with things that change on a few-second timescale: core boost frequency (a
fixed-latency DRAM access costs more *cycles* at a higher clock) and the D49
trainer's memory-bandwidth phase. Running control and arm **at the same time**
on two physical cores of the same CCD holds both constant by construction; the
new confound, core asymmetry, is cancelled by running every pass both ways and
averaging. `tools/perf35/conc35.sh`, `docs/research/perf-artifacts-35/conc.txt`.

| | i/n Δ% | **cyc/n Δ%** | loads/n Δ% |
|---|---:|---:|---:|
| `null` (byte-identical) | +0.000 | **+0.383** [−1.18, +2.30] | −0.019 |
| `after` | −0.442 | +1.348 [−1.07, +2.53] | −0.329 |
| `hookc` | −0.201 | −0.233 [−3.69, +3.56] | −0.036 |

This is the best null of the three — 4× tighter than protocol 1 — and it is
**still ±1.7 points wide around a +0.38% median**, against an arm the mechanism
prices at 0.3–0.4%. Two more halvings would be needed and I do not have a fourth
idea that gets them.

### 3.5 Protocol 1 again, on a calmer box (n=15) — same verdict

Re-run at the end of the session with `base`, `null`, `after`, `after2`, when
the box happened to be quieter (`docs/research/perf-artifacts-35/panel2.txt`).
The control's `cyc/n` spread halved to ±9.688% and the position-in-round table
went nearly flat (medians 4702.7 / 4712.7 / 4692.2 / 4612.9). It made no
difference to the verdict:

| | i/n Δ% | **cyc/n Δ%** | loads/n Δ% |
|---|---:|---:|---:|
| `null` (byte-identical) | +0.000 | **+2.780** [−4.00, +16.50] | +0.003 |
| `after` | −0.442 | +0.076 [−7.07, +9.09] | −0.295 |
| `after2` | −0.406 | +2.879 [−3.22, +11.12] | −0.148 |

A quieter box tightened the *spread* and not the *bias*: the byte-identical null
went from +1.585% to +2.780%. Whatever separates two identical binaries here is
not simply "how loaded the machine is".

### 3.6 What the arm is therefore landed on

Three instruments that *do* resolve, all pointing the same way, on a change that
is node-identical and eval-hash-identical:

* **instructions/node: −25.000 (−0.442%)**, against a null of **exactly 0.000%**
  — the control returned 5659.760 fourteen times out of fourteen.
* **branches/node: −3.02 (−0.513%)**, likewise exact.
* **L1d load ops/node: −0.491% (protocol 1, n=14), −0.295% (protocol 1 rerun,
  n=15) and −0.329% (protocol 3, n=6 pairs)**, against byte-identical nulls of
  **−0.060%, +0.003% and −0.019%** — one to two orders of magnitude below the
  arm in all three.

The last is the one study 33 licensed by name, and its three independent
protocols agree on sign and rough size while their nulls sit at zero.

**What this does not do is prove the cycles win.** Study 33's index says the
payoff at this site is linear in load ops, which converts −0.4% of engine loads
into roughly −0.3 to −0.4% of engine cycles — a number this box could not have
resolved even if I had guessed the protocol right the first time. That is the
honest statement: the change deletes work, every deterministic counter agrees it
deletes work, and the cycles number remains unmeasured.

---

## 4. PERF STUDY 13 §5.1, RE-PRICED — the rejection does not reproduce

Study 13's line was:

```
a2-dirty-via-hookc     6573.8    +41.5    rejected (S5.1)
a2b-hookc-byvalue      6570.5    +38.2    rejected (S5.1)
```

against a baseline of 6537.1 i/n, with the mechanism stated as: `make_move_with`
"is a 650-instruction function that is already spilling, and the marginal cost of
materialising a `DirtyPiece` *there* exceeds the cost of deriving it from scratch
outside".

Rebuilt on today's `main` (`hookc`): an associated `const WANTS_DIRTY: bool` so
`NoHooks` — hence perft and `Board::make_move` — monomorphizes the construction
away, the record built from step 3's own `us`/`pc`/`captured`/`capsq` and handed
to Hook C by value, `do_move` reading it back out of the hook afterwards.

**−11.36 i/n. A win.**

Two caveats before this is read as "study 13 was wrong":

* **The frames are not comparable.** Study 13's baseline was 6537.1 i/n in its
  own fast probe frame; this study's is 5659.76. The engine has lost ~13% of its
  instructions since, across studies 14–33. I can only say the arm is a win *in
  today's engine*, not that 13's measurement was mis-taken.
* **Its stated mechanism has since been removed.** 13's spilling `make_move_with`
  is the one that took a `mem::take` of the state vector and built a 224-byte
  stack temp; perf study 14 replaced that shape wholesale with edit-in-place
  (`push_state`). The register pressure the arm was rejected for is not there any
  more.

This rebuild also differs in one substantive way, and it is worth naming because
it is where a few of the instructions come from: `(captured, capsq)` already
unifies the normal and en-passant cases, because step 3 computes `capsq` as the
ep pawn's square precisely so step 5 and step 10 need no special case. So
`from_actors` has two hot arms where `from_move` has three, and the ep arm — one
of the two `#[cold]` outlined helpers on the pre-move path — disappears entirely.

**And it is still not what landed**, for the reason in §0.2: it deletes
instructions but not loads. `hookc` moves L1d loads by −0.04% (protocol 1) and
−0.036% (protocol 3) — indistinguishable from its byte-identical null, while
`after` moves them −0.30…−0.49% in the same panels. That contrast is the
measurement; the mechanism I would offer for it is that building the record from
registers the make already holds saves no memory traffic in either direction —
the loads it avoids, `make_move_with` was going to issue anyway — whereas
reading the make's *stores* takes them off the node. I did not test that
explanation (a disassembly diff of the two `do_move` bodies would), and §9 item 1
records that the size of `after`'s loads delta is larger than a naive count of
the deleted mailbox reads predicts.

`hookc` also has a cost the table does not show: `WANTS_DIRTY` is a const on the
search's hook type, so the PSQT and ProbCut-probe paths construct a record they
never read. `after_move` sits behind `nnue_active()` and the classical path pays
nothing, which is what keeps the depth-6 bench bit-identical for the right reason
rather than by luck.

---

## 5. TWO THINGS THE STUDY LEARNED BY ACCIDENT

### 5.1 A doc comment changes the binary's sha256

`bin35/after3` — the exact source that landed — has a different sha256 from
`bin35/after`, which it differs from only in **comments and one line-wrap that
`cargo fmt` chose**. The reason is that Rust's panic machinery embeds
`Location { file, line, column }` constants, so inserting fourteen lines of doc
comment relocates every `expect` site below it in `board.rs` and rewrites that
rodata.

It reproduces every measured number exactly — `i/n 5634.38`, `br/n 586.04`,
5,001,562 nodes, hash `5ca978e012da1bb0`, identical to `after` on all four —
which is what makes the source↔binary correspondence provable despite the sha.
**Worth writing down because sha-identity is the obvious way to prove "I measured
what I landed", and in this crate it does not survive editing a comment.**

### 5.2 A `#[cold]` arm respell is worth 2.0 i/n

`after2` differs from `after` in four lines *inside*
`special_record_after`'s `#[cold] #[inline(never)]` en-passant branch: it
constructs `Piece::new(them, Pawn)` instead of reading `state.captured` behind an
`expect`. Same record, same tests, same node count, same eval hash.

**+2.000 i/n** (5636.760 vs 5634.760, exact and reproducible over 15 rounds),
and −0.148% loads instead of −0.295%.

En passant is well under 1% of made moves, so that arm's own instructions cannot
be more than a rounding error of the 5M-node total — the difference is LLVM
deciding something different somewhere else. This is perf study 13 §5.5's
finding ("*cold, never-executed code in an unrelated module can move the hot
path by 1.4%*") reproduced in miniature, and it is the reason the landed spelling
is the one with the `expect`: chasing the cleaner-looking cold arm was worth 8%
of the study's entire win, for zero change in behaviour. The `expect` is
unreachable by construction (`make_move_with` step 3 sets `captured`
unconditionally for `EnPassant`) and is the same class of assertion the
neighbouring `special_record` already carries.

---

## 6. THE DECOMPOSITION — which half of the duplication is worth what

`passmover` exists to split the win. It leaves the derivation where it was, on
the pre-move board, and only threads in the mover that `do_move` already holds:
`do_move` has taken `moved_piece: Option<Piece>` from its caller since perf study
23 arm 6, so `from_move`'s `board.piece_on(from).expect(...)` was the *third*
derivation of that one fact.

Δ loads% is protocol 1 (n=14), where all three arms were measured in the same
rounds.

| | Δ i/n | Δ br/n | Δ loads% | what it deletes |
|---|---:|---:|---:|---|
| `passmover` | −19.00 | −1.00 | −0.324 | `mailbox[from]`, `board.side_to_move`, the `expect` panic edge, the dead `is_ok` sentinel |
| `after` | −25.00 | −3.02 | −0.491 | all of the above **plus** `mailbox[to]`, plus the `Option<DirtyPiece>` the `.then()` closure carried across the make |

So threading the mover is 76% of the instruction win and the victim is the
remaining 24% — but the victim is the half that cannot be threaded, because
`do_move`'s caller does not have it and only `make_move_with` computes it. That
is the whole argument for taking the record off the far side of the make instead
of the near one.

---

## 7. THE INSTRUMENTS

`tools/perf35/`, all on studies 29/31/33's frame:

* **`build35.sh`** — one labelled arm into `bin35/`. Release LTO,
  `target-cpu=native`, `sota,arch-d48x`. 3000 MB availability gate,
  `MemoryMax=6G`, `MemorySwapMax=0`, `-j2`, `nice 19`, pinned to cores 0–1 and
  off the measurement CPU, behind `tools/oomlast.sh`.
* **`ident35.sh`** — i/n, br/n, node total, wl50 eval hash per arm. The gate that
  makes everything else comparable.
* **`panel35.sh` + `med35.py`** — the sequential rotated panel and its PAIRED
  WITHIN-ROUND reducer: median of per-round deltas (where common-mode drift
  cancels) rather than a difference of two independent medians (which cannot see
  a position effect at all — study 29's `med29.py` did the latter). Also prints
  min-over-rounds and the position-in-round table of §3.2.
* **`panelsmt35.sh`** — protocol 2. **Kept, with its bug documented in its own
  header**: the spinner must not be one of the measured binaries.
* **`conc35.sh` + `conc35.py`** — protocol 3, the simultaneous two-core pairing
  with core-swapped passes.
* **`waitpanel35.sh`** — block on a panel reaching *n* lines, so a turn is never
  held hostage on a long job.

Noise floors measured by this study on this box, with the D49 trainer live:

| metric | control's round-to-round spread | byte-identical paired null |
|---|---:|---:|
| `instructions:u` | ±0.000% | +0.000% |
| `L1-dcache-loads:u` | ±0.829% | −0.060% / +0.003% / −0.019% |
| `cycles:u` | ±20.341% | +1.585% / +1.650% / +2.780% / +0.383% |
| `ex_no_retire.load_not_complete:u` | ±30.591% | +2.107% / +2.699% |

`perf stat` reported **100.00% counter-enabled** for all five events, so none of
this is multiplexing.

---

## 8. BOX

**Not quiet, and it could not be made quiet.** The D49 trainer ran at ~143% CPU
throughout and must not be disturbed; perf study 34 was building and measuring
in a sibling worktree for part of the session (two `rustc` at ~100% during
protocol 3); the user's browser held 40–100%. `pgrep -x perf` and
`pgrep -x fastchess` were both empty at every measurement, so no other *perf*
session was competing for the PMU.

Measurements were pinned to CPU 2 (sibling 10) for protocols 1–2 and CPUs 2 and
4 for protocol 3, which is study 33's CPU and deliberately not study 32's CPU 6
(sibling 14 — where the trainer was in fact scheduled for most of the session).

Absolute cycle numbers here are inflated by contention and should not be quoted;
the ratios are what this study reads, and §3 is a measurement of exactly how far
even those can be trusted.

---

## 9. WHAT I AM LEAST SURE ABOUT

In order.

1. **The loads delta is bigger than the mechanism accounts for.** Deleting
   `mailbox[from]`, `mailbox[to]` and `side_to_move` is ~3 load ops per node
   against a total of ~3081 — about 0.10%. The measurement says 0.30–0.49%,
   three to five times that, consistently across three protocols with nulls at
   zero. Candidate explanations I did not test: `L1-dcache-loads` on Zen 4 counts
   *dispatched* loads including replays and mispredicted paths, and this arm also
   removes 3.02 branches per node; or LLVM restructured more than the three
   reads. **The sign is solid and the null is clean; the magnitude may be the
   instrument.** If someone needs the exact figure, IBS retired-load sampling
   (study 33's `ibs33.sh`) is the tool.
2. **The cycles win is inferred, never measured.** §3 says why, at length, and I
   would rather it had been measured. Everything in §0.1's table is a
   work-reduction counter; the conversion to cycles rests entirely on study 33's
   stall/load index of 1.00 at this site, which that study itself lists as the
   thing *it* was least sure about.
3. **Whether `hookc` would beat `after` under PGO.** Every number here is from a
   non-PGO build, and study 33 showed PGO changes the inlining at this exact site
   dramatically (`do_move` collapsing from 116.6 to 18.0 as its own symbol).
   `after`'s advantage is that its two `StateInfo` reads become store-to-load
   forwards *when make_move_with is inlined into do_move* — which PGO makes more
   certain, not less, so I expect the ordering to hold. But I did not build the
   PGO pair.
4. **The SMT text-sharing diagnosis of §3.3 is inferred.** The failure shape is
   very suggestive and the sharing is architecturally real, but the clean control
   — spinner from a third, never-measured copy — was not run, because by then I
   judged protocol 3 the better use of the box.
5. **The `after` vs `after2` ordering (§5.2) may not be a real ordering.** The
   +2.000 i/n is exact and reproducible, but the loads half of it (−0.295% vs
   −0.148%) is the same size as the protocol-to-protocol spread of `after`'s own
   loads delta (−0.295% to −0.491%). I landed the spelling with the better
   numbers; if a future reader prefers the `expect`-free one, the cost is at most
   0.035% of engine instructions and it is not worth a rebuild to find out.
6. **`passmover` is a real, unlanded 19 i/n.** It is strictly dominated by what
   landed, so it is not a loss; but if a future change ever needs the record on
   the *pre*-move board again, that is where 76% of this win still lives.
7. **No game testing.** This is node-identical by construction and verified so,
   so its Elo is exactly whatever ~0.4% of engine work is worth — but "0.4% of
   work" and "0.4% of time" are the same thing only if the deleted work was on
   the critical path, and §3 is why I cannot say that from measurement.
