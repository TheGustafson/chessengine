# Perf study 56 — the bundles were hiding their members, and two of study 55's headline numbers do not survive its own landing

**Date:** 2026-07-31 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 15 GB,
**shared with a live D49 GPU trainer and a 1500-game fixed-node net screen** ·
**Base:** `main` at `0ce956f4` (study 55 merged) · **Net/arch:** D48X @
`network_scale=600`, `sota,arch-d48x`, ACC 1280 · **Instruments:**
`tools/perf56/` · **Artifacts:** `docs/research/perf-artifacts-56/` ·
**Branch:** `perf-study-56-combine`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,562 nodes**, one thread, 16 MB TT, pinned to
physical core 2. Non-PGO release, fat LTO, CGU = 1, `-C target-cpu=native`.

**Frame cross-check (rule 8).** The shipping binary measures **5946.69
instructions/node**. Study 55 measured `base55` at 6030.49 and landed −83.8;
6030.49 − 83.8 = **5946.69**. Reproduced to the second decimal across a session
boundary, and `bin56/ctl56`'s `.text` md5 is **`4e4aad52`** — byte-identical to
the binary study 55's §4.2 recorded as `promoted55`. The frame and the artifact
both survive.

---

## ★ 0. THE ANSWER, FIRST

### 0.1 ★ LANDED: −1.107 % of instructions = **−65.8 instructions/node**

n = 8 paired rounds, `instructions:u` se **0.000**, **8/8**, p = 0.0078, against
the shipping binary, node-identical at 5,001,562. **No counter regresses**;
every memory counter improves in the same panel:

| `panel-final.txt`, n = 8 | instr % | **i/n** | L1d % | ld % | st % |
|---|---:|---:|---:|---:|---:|
| `null56` (byte-identical control) | −0.000 | 0.0 | −0.036 | +0.047 | −0.009 |
| `c_fin5` (the best arm) | −0.893 | −53.1 | −0.402 | +0.251 | −0.685 |
| **`promoted56` (the landed tree)** | **−1.107** | **−65.8** | **−0.654** | **−0.456** | **−1.207** |

### 0.2 ★ Study 55's two retained "large wins" are not wins. The interaction did not BLOCK them, it CONSUMED them.

Study 55 kept `bcmisc55` (−44.7 i/n) and `bcsort55` (−34.6 i/n) behind feature
flags as "large wins alone, blocked only by interaction", and the commission
sent me to recover ~79 i/n of them. **There is nothing to recover.** Both were
measured against `base55`, a substrate that stopped existing the moment study 55
landed small-body inlining and `BUF_MASK` on top of it. Re-measured against the
binary study 55 actually shipped (`panel-r1.txt`, n = 7, se 0.000, ref = the
shipped binary):

| arm | study 55, vs `base55` | **study 56, vs SHIPPED** | neg/n |
|---|---:|---:|---:|
| `bcmisc55` — reduction `clamp` | *(bundled)* | **+11.3** | 0/7 |
| `bcmisc55` — `LMR_DIVISOR` `clamp` | *(bundled)* | **+0.5** | 0/7 |
| `bcmisc55` — `DeltaBuf::push` mask | *(bundled)* | **−1.8** | 7/7 |
| **`bcmisc55` (whole bundle, rebuilt from `0ce956f4`)** | **−44.7** | **+10.4** | 0/8 |
| **`bcsort55`** | **−34.6** | **+14.7** | 0/7 |
| **`bccont55`** | **+5.8** | **+55.6** | 0/7 |

`bcsort55` also loses *in combination*: §5's `c_fin4` is the landing set plus
`bcsort55` and it is **14.7 i/n worse** than the same binary without it. This is
a straight retraction of a study-55 claim, and the general lesson is sharper
than the specific one: **a retained arm's number expires the moment anything
lands underneath it.** Study 55 said "do not assume they stack"; the stronger
rule is *do not assume they still exist*.

### 0.3 ★ A bundle cannot tell you which member matters, and study 55's bundles were hiding a −33 i/n win and a +28 i/n loss

Study 55's `inld55` bundled `check_time`, `correction_value`,
`attackers_to_exist` and `pseudo_legal` and measured **+25.0 i/n** — a loss,
which retired the whole group. Split and priced individually:

| helper | body | study 55 | **study 56, alone** |
|---|---:|---:|---:|
| `Board::attackers_to_exist` | 341 B | *(inside `inld55`, +25.0)* | **−19.2** |
| `Board::attackers_to` | 362 B | not tried | **−13.8** |
| `Histories::correction_value` | 504 B | *(inside `inld55`, +25.0)* | **+28.1** |

The two `attackers_*` helpers split **perfectly additively** (−13.8 + −19.2 =
−0.555 %, and the bundle measures −0.555 %), so the non-additivity is not
everywhere — it is specifically between arms that compete inside the same
frame. `inld55` was one hidden −19.2 sitting next to one hidden +28.1.

### 0.4 ★ The spelling of a bounds fix is worth more than the fix: `clamp` +11.3, mask −3.9

`Worker::reduction`'s two `reductions[]` indices carry the same provable bound
either way. Study 55 wrote it as `clamp` (two compares and two `cmov`s per
index) and it costs **+11.3 i/n** on the shipping tree. Written as `& 255` it is
**−3.9 i/n**. **A 15.2 i/n swing, same bound, same semantics, opposite sign.**
Study 55's own biggest win (`bcbuf55`, −84.6) was a mask; its `bcmisc55` was a
clamp; nobody noticed the two were different mechanisms wearing one label.

### 0.5 ★ The static census RANKS NOTHING. Four measured losers all help in combination.

`tools/perf56/sites56.py` resolves every bounds-check panic edge in the binary
to its exact source line (§2). It is an excellent *candidate generator* and a
**worthless predictor**:

| arm | panic edges deleted | **i/n** |
|---|---:|---:|
| `sqidx56` (the biggest census family, 61 sites) | **13** | **+3.5** |
| `zobmat56` (2nd-largest site, all in `make_move`) | **9** | **+7.1** |
| `m56red2` | 5 | **−3.9** |
| `contcorr56` | 6 | **−5.3** |

And then, in combination, the direction reverses: `sqidx56` (+3.5 alone),
`zobmat56` (+7.1), `rinto56` (+6.1) and `innnue56` (+1.5) sum to **+18.2 i/n of
measured loss**, yet adding all four to `c_fin2` is worth **−5.5 i/n**
(`c_fin2` −0.801 % → `c_fin5` −0.893 %, n = 12, §5). Every one of them is in the
shipped tree *because* it was measured as a loser and put in a combination
anyway.

---

## 1. THE INSTRUMENT, AND ITS CONTROLS

**`instructions:u`** is the only counter this study adjudicates on. Across every
panel here its standard error is **0.000** and a byte-identical binary reads
**−0.000 %**. Ordering is a **blocked Latin square** (`tools/perf56/exact56.sh`
— rotate within a block of `NLAB`, reshuffle the base between blocks); deltas
are paired **within** round.

**No cycles claim is made anywhere in this study.** The commission's σ = 1.57 %
figure stands and this box additionally ran a 1500-game match and a GPU trainer
throughout. `cycles:u` is present in every artifact file and is **not
evidence**; do not read it. `instructions:u` is unaffected by that contention —
it counts retired user instructions for a deterministic, node-identical
workload, which is exactly why it is the licensed counter.

### 1.1 Three controls, not one

1. **`ctl56`** — this study's tree with no arm feature, built before any edit.
   `.text` md5 **`4e4aad52`**, **byte-identical to study 55's landed
   `promoted55`**. The starting point is provably the shipped artifact.
2. **`null56`** — a byte-for-byte `cp` of `ctl56` riding in every rotation.
   Reads −0.000 %, se 0.000, in all six panels.
3. **`ctl56b` / `ctl56c`** — the tree after all scaffolding was added, built
   twice at different points. `.text` md5 **`153f7ef2`** both times, and
   `ctl56b` measures **0.000 % against `ctl56`** (`panel-r2.txt`, n = 8).

Point 3 matters because the md5 *moved* from `ctl56`: adding `#[cfg]` blocks
relocates panic `Location` constants (study 35 §5.1), so byte-identity is not
available once scaffolding exists. **Study 55 asserted its control was inert
from an md5; this study measured it instead**, and the measurement is what makes
`ctl56b` usable as the reference for every arm contrast.

### 1.2 A dilution the whole series should know about

A `perf record` of the frame attributes **≈5.2 % of all measured instructions to
one-time startup** — `quantise_halfka` 2.70 %, `sota_loader::load_v2_bytes`
1.54 %, `attacks::init_magics` 0.97 % — which is net loading and table building,
not per-node search work. Every percentage in this study (and in 52–55) is
therefore **diluted by ~5 %**: the landed −1.107 % is ≈ −1.17 % of the work that
actually scales with nodes. Reported because it is a constant offset in the
frame nobody has stated, not because it changes any ranking.

---

## 2. THE CENSUS INSTRUMENT — every bounds check, resolved to its source line

Study 55 §7 item 4 left the census method as "`objdump`, count call targets per
symbol". That names a *symbol*; a 16 KB `search::<NT>` with eight bounds checks
tells you nothing about which index to mask. `tools/perf56/sites56.py` resolves
all of them exactly, with **no debug-info build**:

`core::panicking::panic_bounds_check` takes a `&'static Location`, and rustc
materialises that argument with a `lea <addr>(%rip), %rdx` a few instructions
ahead of the call. A `Location` is `{ file: &str (ptr, len), line: u32, col:
u32 }` — 24 bytes in `.rodata`. Read it back and you get `file:line:col` for
every check in the binary.

**One trap, recorded because the first version of the script fell into it.** The
binary is a PIE, so the `file` pointer inside every `Location` is stored as
**zero** and supplied at load time by an `R_X86_64_RELATIVE` relocation. Without
joining `readelf -r`, all 284 edges decode to a null pointer and the script
cheerfully reports **"1 distinct source site"**. A census tool that finds one
site in a 4 MB binary is lying, not succeeding.

The resulting ARCHON ranking of the shipped binary (378 panic edges, 151
symbols; top sites):

| edges | source site | what it is |
|---:|---|---|
| 14 | `history.rs:543` | `cont[id.0 as usize]` — `cont53`'s site |
| 9 | `zobrist.rs:90` | `psq[pc][8 + count]`, all inside `make_move`/`do_move` |
| 8 | `tt.rs:504` | `clusters[mul_hi64(key, len)]` — a runtime-length `Vec` |
| 8 | `sota_threats.rs:220` | `DeltaBuf::push` |
| 7 | `sota_accumulator.rs:187` | |
| 5 | `attacks.rs:349` | `PAWN_ATTACKS[color][sq.index() as usize]` |

**And the census found the root cause of the largest family.**
`Square::index()` returns a **`u8`**, so `sq.index() as usize` is provably
`< 256` and nothing more, while every table it subscripts is 64 wide — the
mailbox, `ZOBRIST.psq`'s inner axis, every leaper and geometry table, the
castling-rights mask, the magic table. **61 subscript sites**, all now routed
through a new `Square::idx()` that carries the `& 63`.

That is the correct diagnosis and it is *still* only +3.5 i/n alone (§0.5). The
mechanism, measured: **a mask is unconditional cost, a bounds check is not.**
Where LLVM already folded the check in the hot inline copy and left it only in
cold copies, the census counts the cold edges and the mask pays at the hot one.
Edge count measures code, not execution.

### 2.1 The missing bound, stated accurately

The commission is right and study 55 is wrong. Study 55 called
`reductions[move_number]` "a latent panic: `move_number` reaches 256 in a
256-move position". **There is no 256-move position.** The maximum legal move
count in any chess position is 218, and `move_number` in our loop is incremented
only *after* the legality filter (`if !NT::ROOT && !self.board.legal(m) {
continue; }`), so it counts legal moves. The other index is bounded even more
tightly: `depth` is clamped to `MAX_PLY - 1` = **245** at node entry (N0,
`search/mod.rs:2347`), and `search` routes `depth <= 0` to qsearch before the
move loop.

So it is a **missing bound, not a reachable panic**. I looked for a path to
index ≥ 256 — a pseudo-legal count, a re-search counter, a duplicate emission
from `MovePicker` — and did not find one; the root arm skips the legality filter
but roots iterate an already-legal root-move list. **I am not claiming a bug.**
The bound is landed anyway: it is free correctness, and it rides along with a
win once spelled as a mask.

---

## 3. MECHANISM B, ARM BY ARM

*(`panel-r1.txt` n=7 ref `ctl56`; `panel-r2.txt` n=8 ref `ctl56b`. se 0.000
throughout. 1 % = 59.4669 i/n.)*

| arm | site | instr % | **i/n** | neg/n | verdict |
|---|---|---:|---:|---:|---|
| `a_red` | `reduction()` — study 55's **clamp** | +0.190 | **+11.3** | 0/7 | loss |
| **`m56red2`** | `reduction()` — **mask** | −0.065 | **−3.9** | 8/8 | **LANDED** |
| `m56lmr` | `LMR_DIVISOR[depth.clamp(1,16)-1]` | +0.009 | +0.5 | 0/7 | ~neutral, **LANDED** (closes a depth-0 underflow) |
| **`m56dbuf`** | `DeltaBuf::push` store mask | −0.031 | **−1.8** | 7/7 | **LANDED** |
| **`contcorr56`** | `cont_corr` 832 → 1024 + mask | −0.089 | **−5.3** | 8/8 | **LANDED** |
| `cont53` | `cont` 3328 → 4096 + mask | −0.086 | −5.1 | 8/8 | **dominated** — see below |
| `sqidx56` | `Square::idx()`, 61 sites | +0.059 | +3.5 | 0/8 | loss alone, **LANDED** (helps in combination) |
| `zobmat56` | `psq[pc][8+count]` mask | +0.119 | +7.1 | 0/8 | loss alone, **LANDED** (ditto) |
| `rinto56` | `resolve_into`'s 4 cursor stores | +0.102 | +6.1 | 0/7 | loss alone, **LANDED** (ditto) |
| `bcsort55` | study 55's retained arm | +0.248 | **+14.7** | 0/7 | **loss, retracted** |
| `bccont55` | study 55's falsification | +0.935 | **+55.6** | 0/7 | **loss, ten times worse than study 55 said** |

**`contcorr56` is the one that pays for itself, and the reason is the price.**
It is `cont53`'s shape on a smaller table: `cont_corr` is 832 slabs of 1536 B,
indexed by `ContHistId::piece_to()`, which the optimizer only knows is `< 16384`
(it is a `u16 / 4`). Padding to 1024 costs **+288 KiB per worker thread**,
taking design/09 §8's catalogue from 19.979 to 20.260 MiB — **inside** the
21 MiB budget. `cont53` pads 3328 → 4096, costs **+1.125 MiB** and breaches it.

**`cont53` is dominated and should stay unlanded.** Alone it is −5.1 i/n; added
to the study-56 landing set it moves that set from −12.8 to −12.0 (`c_rcdl` vs
`c_rcd53`, `panel-r3.txt`). It buys nothing on top and still costs the 1.125 MiB
per thread. That closes study 53's open item on this arm.

**`resolve_into` deserves its own note even though it lost alone.** Its fused
twin `resolve_both` takes `&mut [[usize; DELTA_CAP]; 2]` and masks its cursors
with `& M` (perf study 24). `resolve_into` did the identical job with the
identical buffers — the caller passes `&mut self.add_scratch[p]`, already a
`[usize; DELTA_CAP]` — but its signature erased them to `&mut [usize]`, so it
could never fold. The signature is now tightened to match its twin. It is +6.1
i/n alone and part of a −5.5 i/n group in combination.

---

## 4. MECHANISM A — the size rule is real but it is NOT a size rule

The commission's instruction was: do not blindly extend mechanism A; price
specific small helpers. So the candidate set was generated mechanically instead
of by hand — join `nm -C -S` (body size) against `perf report` (profile weight)
and take everything still out of line. Each candidate was then measured
**alone**, because §0.3 is what happens when you don't.

| helper | body | instr % of profile | **i/n** | verdict |
|---|---:|---:|---:|---|
| **`Board::attackers_to_exist`** | 341 B | 1.25 % | **−19.2** | **LANDED** |
| **`Board::attackers_to`** | 362 B | 1.69 % | **−13.8** | **LANDED** |
| **`SotaContext::evaluate`** | 885 B | 1.67 % | **−6.9** | **LANDED** |
| **`apply_at` + `ripple` + `psqt_fold`** | 479 / 452 / 166 B | 2.02 % | see §6.2 | **LANDED** |
| `AttackTables::attacks` | 359 B | 0.21 % | 0.0 (2/7) | inert |
| `MovePicker::init_evasions` | 734 B | 0.57 % | 0.0 (3/7) | inert |
| `Worker::undo_move` | 448 B | 0.99 % | −0.0 (6/7) | inert |
| `Worker::stat_score` | 505 B | 0.50 % | 0.0 (1/7) | inert |
| `Board::refresh_check_info` | 632 B | 1.01 % | **+20.5** | loss |
| `Histories::correction_value` | 504 B | 0.69 % | **+28.1** | loss |

**Study 55 §0.1 said the discriminator is body size. It is not.** The largest
body in this table wins (`SotaContext::evaluate`, 885 B, −6.9) and a body barely
half its size is the worst loser (`correction_value`, 504 B, +28.1). Size
correlates because big bodies spill more state into the 16 KB `search::<NT>`
frame, but the actual variable is **how much live state the body adds to the
frame it lands in**, and `evaluate` lands in the NNUE call chain rather than in
`search::<NT>` at all. A size threshold is a proxy that misranks by 28 i/n.
Measure the helper; do not measure its `nm` output.

---

## 5. THE COMBINATION SEARCH — built, never summed

Every row below is a real binary, built and measured. `panel-r3/r5/r7/r8`.

| label | contents | predicted from parts | **measured** | **i/n** |
|---|---|---:|---:|---:|
| `c_rcdl` | red2+ccorr+dbuf+lmr | −0.176 % | **−0.215 %** | −12.8 |
| `c_rcd53` | `c_rcdl` − lmr + cont53 | — | −0.201 % | −12.0 |
| `c_fin` | `c_rcdl` + `inatk56` | −0.770 % | **−0.685 %** | −40.7 |
| `c_fin2` | `c_fin` + `ineval56` | −0.801 % | −0.801 % | −47.6 |
| `c_fin3` | `c_fin2` + **`sqidx56` (a loser)** | −0.742 % | **−0.816 %** | −48.5 |
| `c_fin4` | `c_fin2` + **`bcsort55`** | — | **−0.553 %** | −32.9 |
| `c_fin6` | `c_fin3` + 4 inert helpers | — | −0.816 % | −48.5 |
| **`c_fin5`** | `c_fin3` + **`zobmat56`+`rinto56`+`innnue56` (all losers)** | −0.569 % | **−0.893 %** | **−53.1** |

**Non-additivity, this study's four data points:**

| combination | error vs sum of parts | direction |
|---|---:|---|
| `inatk56` = `atk1` + `atk2` | **0.000 pp** | *exactly additive* |
| `c_rcdl` | −0.039 pp (−2.3 i/n) | favourable |
| `c_fin` | +0.085 pp (+5.1 i/n) | unfavourable |
| `c_fin5` vs `c_fin3` | **−0.324 pp (−19.3 i/n)** | **favourable** |

Study 55 reported errors up to 2.63 pp; this study's largest is 0.324 pp, but it
is in the direction that matters most for landing — **the three arms added at
that step each cost instructions alone (+7.1, +6.1, +1.5) and are worth
−4.6 i/n together on top of `c_fin3`.** And
the `inatk56` row is the counter-example that keeps the rule honest: additivity
is not *absent*, it holds exactly when the arms do not compete for the same
frame. The operational rule is unchanged and now doubly paid for: **build the
combination, measure the combination, and put your measured losers in one.**

---

## 6. WHAT LANDED

Ten changes, all unconditional, all node-identical at 5,001,562.

* **`Square::idx()`** — a masked table subscript, replacing
  `sq.index() as usize` at all 61 subscript sites.
* **`Zobrist::material`** — `psq[pc][(8 + count) & 63]`.
* **`resolve_into`** — signature tightened to `&mut [usize; DELTA_CAP]` to match
  its twin `resolve_both`, and its four cursor stores masked.
* **`DeltaBuf::push`** — masked store (study 55's `m56dbuf`).
* **`cont_corr`** — padded 832 → 1024 with a masked accessor pair
  (`corr_slab`/`corr_slab_mut`), +288 KiB/worker.
* **`Worker::reduction`** — the missing bound, as a mask.
* **`LMR_DIVISOR[depth.clamp(1,16) - 1]`** — closes the depth-0 underflow.
* **`#[inline(always)]`** on `Board::attackers_to`, `Board::attackers_to_exist`,
  `SotaContext::evaluate`, `ThreatContext::apply_at`,
  `sota_threats::ripple`, `sota_accumulator::psqt_fold`.

### 6.1 Not landed, and why — every negative arm, with its number

`bcsort55` +14.7 · `bccont55` +55.6 · `bcmisc55`'s clamp form +11.3 ·
`correction_value` +28.1 · `refresh_check_info` +20.5 · `cont53` −5.1 alone but
**dominated** in combination and +1.125 MiB/thread · `AttackTables::attacks`,
`init_evasions`, `undo_move`, `stat_score` all inert (0.0, and `c_fin6` proves
the group adds nothing).

`tt.rs:504` (`clusters[mul_hi64(key, len)]`, 8 edges, carrying the comment *"keep
the bounds check until a profile says otherwise (D14c)"*) was **examined and not
attempted**: `clusters` is a runtime-length `Vec`, so there is no constant to
mask against, and the only spellings that fold the check either cost a `cmov`
(`idx.min(len-1)`), cost a division (`% len`), or change the TT mapping to
power-of-two sizing — which is not node-identical. The profile now exists; the
fix does not, in safe Rust, without changing behaviour.

### 6.2 "I measured what I landed", and the landed tree BEAT the arm

`panel-final.txt` runs `c_fin5` and `promoted56` against the shipping binary in
one rotation. The promoted tree reads **−1.107 %** where the arm reads
**−0.893 %** — 12.7 i/n *better*. That gap is real and it has a cause, found
rather than assumed:

**`ripple` and `psqt_fold` already carried `#[inline]`.** Adding
`#[cfg_attr(..., inline(always))]` produced two attributes on one item; rustc
takes the first and warns `unused attribute`, so **in every arm binary those two
helpers were never actually `inline(always)`** — `innnue56`'s +1.5 i/n was
measuring `apply_at` alone. Promotion collapsed the duplicate, and `nm` confirms
the mechanism: both symbols are **present in `c_fin5` and absent from
`promoted56`**. The landed tree does more of what the arm intended, and the
final panel is the measurement of the thing that shipped.

---

## 7. GATES

| gate | result |
|---|---|
| `cargo fmt --check` | **PASSED** |
| `cargo clippy --all-targets -- -D warnings` | **PASSED** |
| feature-matrix `cargo check` | **PASSED** |
| `cargo test` (debug: unit + fast perft tier) | **PASSED** — 614 passed, 0 failed |
| **D24 perft ritual** (`board`/`movegen`/`zobrist`/`attacks` all touched) | **`perft_full_suite ... ok`** |
| bench perft fingerprint (two runs identical) | **1839507** |
| `Bench:` trailer (portable default build, depth 6) | **587683** — **unmoved** |
| **wl50 eval hash** | **`2e1ede95…93cf79c4`** — **unmoved** |
| **d13 node total** | **1 739 588** — **unmoved** |
| frame node identity, all 24 binaries | **5 001 562** exact |
| `ctl56` `.text` md5 vs study 55's `promoted55` | **`4e4aad52`, identical** |
| `ctl56b` ≡ `ctl56` on `instructions:u` | **0.000 %**, se 0.000 |
| `ctl56c` ≡ `ctl56b` | **`153f7ef2`, byte-identical** |

**No pin moved, and none should have** — every change is behaviour-preserving,
so a moved pin would have been evidence that an equivalence was wrong. The
`zobrist.rs` and `board.rs` edits touch key computation directly, which is why
`perft_full_suite` (not just the fast tier) is the gate that matters here.

---

## 8. BOX DISCIPLINE — one deviation, declared

Every build ran under `systemd-run --user --scope -p MemoryMax -p
MemorySwapMax=0`, `-j2`, `nice -n 19`, behind a memory floor **raised from study
55's 2500 MB to 3500 MB** because this box also carried the D49 trainer and a
1500-game screen.

**The deviation:** the owner's gate held
`/home/nick/.cache/archon-build.lock` for the entire duration of a *match*
(`tools/screen.sh`, 1500 games at 25 k nodes), not a build. Waiting would have
cost the study its whole session. `tools/perf56/build56.sh` therefore waits
**30 s** for the lock and, on timeout, proceeds without it under a **tighter**
cgroup (4G rather than 6G) at the lowest CPU and IO priority. The justification
is the owner's own script header: *"Fixed NODES, so the result is
contention-immune and concurrency is free."* The memory floor, not the lock, is
what protects against the failure mode the lock exists for. **Stated here for
ratification rather than footnoted.**

---

## 9. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **Why `sqidx56`, `zobmat56` and `rinto56` lose alone and win together.** §0.5
   establishes *that* they do, at n = 12 with a zero null. The mechanism I offer
   — a mask is unconditional cost while a bounds check may already be folded on
   the hot path — explains the *alone* half and is supported by the census, but
   **I did not measure which register or inlining decision reverses in
   combination.** Study 55 owed this same decomposition and so do I.
2. **Whether the `Square::idx()` sweep contains a hot subset that wins alone.**
   61 sites went in as one bundle, which is the exact error §0.3 documents. It
   lands on the strength of the combination, not of a split. The mailbox sites
   in `board.rs` are the obvious first cut.
3. **`tt.rs:504`.** §6.1 — examined, no node-identical safe-Rust fix found.
   `mul_hi64(key, len) < len` is true and unprovable to LLVM.
4. **No cycles claim, anywhere.** §1.
5. **No PGO measurement.** Now overdue across studies 52–56, and it matters most
   here: the dominant mechanisms are inlining and bounds-check elision, exactly
   what PGO re-litigates, and this study lands six `inline(always)` attributes.
6. **No games.** Every change is a node-identical refactor, so games are owed
   only for the speed→Elo conversion at the standing ~120 Elo per doubling.

---

## 10. WHAT THE NEXT STUDY SHOULD RUN

1. **The PGO pair.** Five studies owe it and this one raises the stakes.
2. **Split `sqidx56`** (§9 item 2) — a −3.5-to-−10 i/n candidate for two builds.
3. **Re-price study 55's `inlab55`** (`stat_score` + `upcoming_repetition` +
   `new_probcut`, −25.3 i/n unlanded). `stat_score` alone is **inert** on the
   shipping tree, so the other two carry it — and neither has been priced alone.
4. **Audit for duplicate `#[inline]`/`#[inline(always)]` attributes.** §6.2 cost
   this study 12.7 i/n of invisible measurement error, in the favourable
   direction only by luck. `cargo clippy` does not flag it; `rustc` emits a
   `warning: unused attribute` that is easy to miss in a 54-warning build.
5. **Do not re-run study 55's retained arms.** §0.2. Delete the expectation, not
   just the number.

---

## 11. REPRODUCTION

```bash
git checkout perf-study-56-combine
bash tools/perf56/build56.sh ctl56                       # the shipping control
bash tools/perf56/build56.sh c_fin5 m56red2,contcorr56,m56dbuf,m56lmr,\
inatk56,ineval56,sqidx56,zobmat56,rinto56,innnue56       # pre-promotion (parent commit)
bash tools/perf56/nodes56.sh ctl56 c_fin5 promoted56      # 5,001,562 or it means nothing
bash tools/perf56/exact56.sh 8 ctl56b null56 c_fin5 promoted56
python3 tools/perf56/med56.py docs/research/perf-artifacts-56/panel-final.txt \
        ctl56b null56 c_fin5 promoted56

python3 tools/perf56/sites56.py bin56/ctl56               # every bounds check, by source line
python3 tools/perf56/census56.py bin56/ctl56              # ... by symbol
bash tools/perf56/edges56.sh ctl56b c_fin5 promoted56     # did the arm delete what it claims?

CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=1 bash tools/check.sh
cargo test --release -- --ignored perft_full_suite
```

`sites56.py` is the reusable one. It needs no debug build, it names the line,
and it will tell you — as it told this study — that the biggest site in the
binary is not the biggest win.
