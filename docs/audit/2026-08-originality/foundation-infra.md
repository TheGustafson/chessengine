# Foundation + Infrastructure Domain — License & Originality Audit

Date: 2026-08-09. Scope: `src/{types,bitboard,board,fen,moves,attacks,movegen,perft,zobrist,cuckoo,uci,bench,main,lib}.rs`, `src/bin/`, `benches/`, and infrastructure (`tools/check.sh`, `tools/build_pgo.sh`, `tools/spsa/`, `tools/ledger/ledger.py`, `tools/nnue/` scripts, `tools/ops/`), plus the three repo-wide checks this domain owns: licensing posture, the distilled-notes audit, and the transcribed-comment sweep.

References compared: Stockfish at **eca43a97** (via `git show`; checkout at sf_18 used for shape only), Reckless (AGPLv3, clone present), Integral (all-rights-reserved; clone ABSENT, notes deleted from tree — audited from git history at `afcf029d`).

Rubric: L0 independent · L1 idea-level adoption, independently expressed · L2 structural echo (same decomposition/names/call shapes) · L3 near-translation · L4 verbatim. Tags: COMMUNITY-STANDARD / SF-SPECIFIC / RECKLESS-SPECIFIC / INTEGRAL-SPECIFIC / ARCHON-ORIGINAL.

**Headline: no L3 or L4 code anywhere in this domain.** The domain's license exposure is confined to (a) small copied CONSTANT SELECTIONS from Stockfish (magic-search seeds, the Zobrist seed, the bench position suite — all disclosed in the design docs under D20/D27), and (b) L2 structural echoes of SF mechanisms that the design docs adopt by name. Nothing in this domain matches Reckless or Integral above L1.

---

## 1. Verdict table

| Module / mechanism | Level | Tag | One-line basis |
|---|---|---|---|
| `src/types.rs` (Square/Color/Piece newtypes) | L1 | COMMUNITY | LERF mapping is chessprogramming-wiki standard; expression is original Rust newtypes |
| `src/types.rs` value-constant family | L2 (constants) | SF-SPECIFIC | SF names AND values verbatim by D20: `VALUE_MATE 32000`, `VALUE_INFINITE 32001`, `VALUE_NONE 32002`, `MAX_PLY 246`, TB bands, piece values (`types.rs:466–530` vs `eca43a97:src/types.h:156–200`) |
| `src/bitboard.rs` | L1 | COMMUNITY | Set-ops/popcount/lsb are common property; expression diverges from Reckless on every axis (see §4.2) |
| `src/moves.rs` (`Move(u16)`) | L1 | SF-SPECIFIC idea | SF's 16-bit scheme, documented as such; from/to fields SWAPPED vs SF (`moves.rs:6–11` from=bits 0–5; `eca43a97:src/types.h:440` `(from << 6) + to`) |
| `src/moves.rs` (`MoveList`) | L1 | COMMUNITY | Fixed array + len; `MaybeUninit` backing and `Deref<[Move]>` are our own; SF uses `ExtMove*` cursors, Reckless a different list |
| `src/board.rs` API surface | L2 | SF-SPECIFIC | SF `position.h` method names wholesale: `pieces`, `checkers`, `blockers_for_king`, `gives_check`, `attackers_to`, `non_pawn_material`, `see_ge`… (documented, design/04) |
| `src/board.rs` state model | L1 | ARCHON-ORIGINAL twist | Inline live `StateInfo` + `Vec` history (D8 as amended, perf study 14) — NOT SF's caller-owned linked list, not Reckless's stack |
| `src/board.rs::upcoming_repetition` | L2 | SF-SPECIFIC | Same algorithm + window; ours DROPS SF-current's progressive `other` key check and ADDS a mover-color guard SF-current lacks (`board.rs:2325–2381` vs `eca43a97:src/position.cpp:1549–1591`) |
| `src/fen.rs` | L1 | COMMUNITY | FEN grammar is a spec; structured `FenError` enum is ours (SF returns a different error surface, Reckless's parser different shape) |
| `src/attacks.rs` const geometry (between/line/ray_pass) | L1–L2 | SF-SPECIFIC semantics | SF's exact table semantics incl. `between` ∪ {s2} and `ray_pass` (deliberate, doc 03 §3.4); built by our own const-fn oracle architecture |
| `src/attacks.rs` `Prng` (xorshift64\*) | L1 | COMMUNITY (published) | Vigna's published algorithm; the shift triple and multiplier ARE the algorithm (`attacks.rs:435–448`; SF `eca43a97:src/misc.h:457–484` is the same paper) |
| `src/attacks.rs` `MAGIC_SEEDS` | **copied constants** | SF-SPECIFIC | All 8 seeds verbatim from SF's 64-bit row — see finding F1 |
| `src/attacks.rs` `init_magics` | L2 | SF-SPECIFIC + COMMUNITY | Carry-Rippler + epoch trick are wiki-standard; SF-specific pieces adopted: per-rank seeding, sparse-candidate prefilter `popcount((magic*mask)>>56) >= 6` (`attacks.rs:719–724` vs `eca43a97:src/attacks.cpp:131–136`). Real deviations: u32 offsets vs pointers, `TABLE_ALLOC` mask + exhaustive reachability proof (ARCHON-original, study 53), zero `unsafe` |
| Magic constants themselves | derived ≡ SF's | SF-SPECIFIC | NOT copied as a table; re-derived at every launch — but from SF's seeds + SF's PRNG + SF-identical consumption order, so the found multipliers equal SF's by construction (finding F1) |
| `src/zobrist.rs` | L1–L2 | SF-SPECIFIC | SF's seed 1070372 + all five hashing tricks (promotion-rank zeroing, whole-mask castling keys, `8+count` material slots, rule50 folding); OWN draw order ⇒ our keys ≠ SF's keys (`zobrist.rs:33,106–111`; `eca43a97:src/position.cpp:120–138`, `src/position.h:322–325`). `make_key` constants are Knuth's MMIX LCG (public) |
| `src/cuckoo.rs` | L2 | COMMUNITY (published) + SF params | Van Kervinck's published algorithm, cited; SF's exact parameter set: 8192 slots, H1 = low 13 bits, H2 = bits 16..29, 3668-entry assert (finding F3) |
| `src/movegen.rs` | L2 | SF-SPECIFIC | SF's GenType architecture (CAPTURES/QUIETS/EVASIONS/NON_EVASIONS), SF's function names (`legal`, `pseudo_legal`, `capture_stage`), target-mask parameterization — adopted by D27/design/05 "mirrors current Stockfish semantics exactly"; expression independent (finding F4) |
| `src/perft.rs` | L1 | COMMUNITY | Bulk counting is standard; `divide` prints SF's output FORMAT (deliberate tool parity); `perft_checked` oracle is ARCHON-ORIGINAL |
| `src/uci.rs` | L1 | COMMUNITY (protocol) | UCI grammar is the protocol; `d` output reuses SF's three field labels `Fen:`/`Key:`/`Checkers:` (`uci.rs:1805–1818` vs `eca43a97:src/position.cpp:83–84`) — de minimis |
| `src/bench.rs` code | L1 | SF-SPECIFIC grammar | CLI argument ORDER `bench [ttMB] [threads] [limit] [fens] [type]` is SF's, documented ("so muscle memory and scripts transfer") |
| `src/bench.rs::BENCH_POSITIONS` | **copied data selection** | SF-SPECIFIC | 49/49 entries byte-identical (incl. `moves …` suffixes) to SF's 54-entry `Defaults` — finding F2 |
| `src/main.rs`, `src/lib.rs`, `src/bin/*`, `benches/*` | L0/L1 | ARCHON-ORIGINAL | Own tooling; criterion benches original |
| `tools/check.sh`, `tools/build_pgo.sh` | L0 | ARCHON-ORIGINAL | Own merge gate / PGO pipeline; no adapted code found |
| `tools/ledger/ledger.py` | L0 | ARCHON-ORIGINAL | Stdlib-only content-addressed provenance DAG; no external lineage |
| `tools/spsa/` | L1 | COMMUNITY (published) | Spall's SPSA with the canonical α=0.602/γ=0.101 schedules, "fishtest/OpenBench-STYLE" parameterization (c_end/r_end) — formulas, not code; drives fastchess as an external binary |
| `tools/nnue/`, `tools/ops/` (skim) | L0/L1 | ARCHON-ORIGINAL | Own scripts; external tools (fastchess, bullet trainer, SF binary for data-gen scaffold per D28) invoked as dependencies, none vendored into the tree; adaptation-marker greps clean |

---

## 2. License-risk findings, most severe first

Nothing here reaches "violation." The findings below are ranked by how much explaining each would need if the tree were published tomorrow.

### F1 — Magic-search seeds copied verbatim from SF; derived magics therefore equal SF's (constants; disclosed)

`src/attacks.rs:639`:

```rust
const MAGIC_SEEDS: [u64; 8] = [728, 10316, 55013, 32803, 12281, 15100, 16645, 255];
```

`eca43a97:src/attacks.cpp:101–102`:

```cpp
int seeds[][RANK_NB] = {{8977, 44560, 54343, 38998, 5731, 95205, 104912, 17020},
                        {728, 10316, 55013, 32803, 12281, 15100, 16645, 255}};
```

Our row is SF's 64-bit row, all 8 values, and the doc comment says so ("SF-current 64-bit values, D20 … tuned against SF's exact PRNG consumption order, so the candidate loop below must consume the PRNG identically"). Because the PRNG is the same published xorshift64\* and the candidate loop deliberately consumes draws in SF's order, **the magic multipliers ARCHON finds at startup are SF's magic multipliers** — re-derived, not pasted, but numerically identical by construction. The answer to the commissioning question "were the magic numbers copied from SF's tables or generated independently?" is precisely: *generated, but not independently.*

Legal weight: 8 integers whose only function is to make a search converge fast; individually unprotectable, and as a "selection" they are the output of SF's own tuning run, i.e. facts about which seeds work (D20's stated policy). Low risk. Originality weight: this is a genuine SF lineage marker — an engine that wanted zero SF lineage here would re-tune its own seeds (cheap: the seeds only affect init time, not play).

### F2 — The bench position suite is SF's, byte-for-byte (data selection; disclosed)

`src/bench.rs:389` `BENCH_POSITIONS` (49 entries) vs SF `eca43a97:src/benchmark.cpp:31` `Defaults` (54 entries): **49 of 49 entries are byte-identical, including the two entries carrying `moves ...` continuation suffixes.** Our suite is a 49-element subsequence of SF's curated list. design/06 §2 cites `benchmark.cpp` as the source ("the default bench FEN suite"), so this is a documented adoption made so that node-count fingerprints exercise an SF-comparable position mix.

Legal weight: chess positions are facts; the thin-copyright question is only about the *selection*, which SF itself publishes under GPLv3 as part of its source. This is the same category as the D20 tunables transcription (owned by the search-domain audit): defensible as uncopyrightable facts/method, but it is the second-clearest piece of copied SF *text* in the repo. If a maximally-clean posture is ever wanted: swap in an own-curated suite and re-pin the bench (one commit; the `Bench:` trailer moves by definition).

### F3 — cuckoo.rs: published algorithm, SF's exact parameterization, closest structural correspondence in the domain

The algorithm is Marcel van Kervinck's, published, and credited in our module header (`src/cuckoo.rs:1`); SF cites the same paper. Ours adopts SF's specific implementation parameters: table size 8192, `h1 = k & 0x1fff`, `h2 = (k >> 16) & 0x1fff`, and the 3668-entry count used as a build assert (`cuckoo.rs:37–55, 42, 133` vs `eca43a97:src/position.cpp:112–117, 162`). The insertion loop is the closest 6-line correspondence found anywhere in this domain:

Ours (`src/cuckoo.rs:119–127`):
```rust
loop {
    std::mem::swap(&mut keys[slot], &mut k);
    std::mem::swap(&mut moves[slot], &mut mv);
    if k == 0 { break; }
    slot = if slot == h1(k) { h2(k) } else { h1(k) };
}
```
SF (`eca43a97:src/position.cpp:152–159`):
```cpp
while (true) {
    std::swap(cuckoo[i], key);
    std::swap(cuckooMove[i], move);
    if (move == Move::none()) break;
    i = (i == H1(key)) ? H2(key) : H1(key);
}
```

Assessment: L2, not L3. Cuckoo displacement-by-swap is the textbook algorithm (merger: there is essentially one way to write it), the empty-slot test differs (key==0 vs move==none), and the surrounding enumeration is restructured (explicit color×piece-type loops with pawns excluded up front vs SF's all-pieces loop with an inline pawn test; `Option`-returning `lookup`; our own doc rationale throughout). The 3668 pin is used as an *oracle against SF as a fact*, which is the D27 pattern.

### F4 — movegen.rs / board.rs: SF's architecture and names, adopted on purpose (L2, the domain's originality floor)

design/05 says it plainly: "The design mirrors current Stockfish semantics exactly (D27)." The GenType split, the CAPTURES-includes-queen-promos / QUIETS-includes-underpromos convention, the target-mask parameterization, and the function names `legal` / `pseudo_legal` / `capture_stage` / `blockers_for_king` / `attackers_to_exist` are SF's decomposition, recognizably. The board API surface is SF's `position.h` vocabulary.

Why this stays L2 and not L3, with a receipt: our `Board::legal` (`src/movegen.rs:499–597`) vs SF's `Position::legal` (`eca43a97:src/position.cpp:660–697`) share the three-case logic (castling path attacked / king steps with mover removed from occupancy / pin-ray alignment) but not the text: ours cold-outlines a `legal_special` with measured register-pressure rationale, tests castling via fixed square pairs where SF walks `to → from`, and carries an EP arm SF-current no longer has inline. The same pattern holds everywhere sampled (`upcoming_repetition` drops SF's progressive `other` accumulator and adds a mover-color guard; `generate` uses `u8` const-generics + a color const-generic where SF uses `Type`/`Color` templates). Same skeleton, own flesh, heavy own documentation. Legal risk low (structure/method); originality cost real and acknowledged — this codebase is *architecturally* an SF re-derivation in its foundation, by pinned decision (D1-era "Stockfish-class architecture").

### F5 — Value/zobrist constant families (constants; disclosed)

`VALUE_MATE 32000 / VALUE_INFINITE 32001 / VALUE_NONE 32002 / MAX_PLY 246` + the TB band arithmetic and SF's constant NAMES (`src/types.rs:466–530` vs `eca43a97:src/types.h:156–199`); Zobrist stream seed `1070372` (`src/zobrist.rs:33` vs `eca43a97:src/position.cpp:122`); rule50 folding thresholds 14/8 (`src/zobrist.rs:184–205` vs `eca43a97:src/position.h:322–325`). All D20 ("adopt Stockfish's current values as starting points, recorded verbatim"). Individually trivial; listed because together with F1/F2 they form the honest answer to "how much SF is in the foundation": *the numbers, the names, and the architecture — never the code.*

### F6 — Repo hygiene, not license: tracked Reckless build artifacts

`reckless_reference/pgo-prof-28/*.profraw` + `merged.profdata` (16 MB) are git-TRACKED — PGO profile counters from a perf-study build of Reckless (AGPLv3). Profile data is facts-about-execution (contains AGPL symbol names, no code); legally inert, but it is 16 MB of third-party-derived binary clutter that would ship with a published repo. Delete-and-gitignore candidate. (The clones themselves are properly gitignored: `.gitignore:5` and `:8`, verified via `git check-ignore`.)

### 2.b — The distilled-notes audit (all 29 tracked notes)

Method: every fenced code block in `stockfish_reference/*.md` (12 files) and `reckless_reference/*.md` (17 files) was line-matched (whitespace-normalized exact) against the FULL reference sources at **three SF revisions** (sf_16, sf_18 checkout, eca43a97) and Reckless HEAD, then re-swept with a token-Jaccard fuzzy matcher (J ≥ 0.75) to catch reformatted/pseudocode quotes.

**Stockfish notes: CLEAN.** Zero verbatim lines and zero fuzzy runs ≥ 3 across all 12 notes at all three revisions. The blocks that exist are paraphrase/condensed pseudocode.

**Reckless notes: three short quoted excerpts, largest 5 lines.** All embedded in analytical commentary; none is a standalone reproduction:

| Note | Location | Size / character | Matches |
|---|---|---|---|
| `reckless_reference/07-search-node-pruning.md` | block @ L140 | **5-line near-verbatim excerpt** of the TT-cutoff condition (reformatted; 3 lines exact, 2 at J=0.86) | `Reckless/src/search.rs:341–347` |
| `reckless_reference/07-search-node-pruning.md` | block @ L318ish (probcut) | ~8 pseudocode lines with token-identical content (Python-style `:` syntax, so not textually verbatim) | `Reckless/src/search.rs:603–622` |
| `reckless_reference/05-movepick-history.md` | block @ L266 | 5-line paraphrase (identifiers drift: `is_present` vs source `is_some`) of `update_continuation_histories` | `Reckless/src/search.rs:1328–1334` |
| `reckless_reference/13-nnue-inference-simd.md` | block @ L294 | 13-line structural pseudocode of the cfg-module tree (explicitly compressed, `...` elisions) | `Reckless/src/nnue.rs:17–80` |

Assessment: quotation-for-study scale (≤ 5 contiguous near-verbatim lines, with commentary), not distribution of the work. If a zero-quote posture is wanted, the 07 TT-cutoff block is the only one worth rewriting into pseudocode.

**Integral notes (git history only).** The 13 notes committed at `afcf029d` were deleted from the tree at `1f8c111c` and are not in `old/` — they exist only in git history, which still ships if the repo is published with history. Audited from `afcf029d`: **all code blocks are pseudocode/paraphrase — zero verbatim C++** (per-block scan: largest blocks are a 38-line pseudocode datagen loop and a 32-line time-management formula sketch; ≤ 4 C++-flavored lines per file, all fragments). They do transcribe Integral's tuned TM constants (0.4193, 0.0575, …) as facts. Given Integral is all-rights-reserved, pseudocode + constants is the right side of the line; nothing found that would require history rewriting.

### 2.c — Transcribed-comment sweep (repo-wide, both directions)

Method: exhaustive intersection — not sampling — of comment text between our tree (724 files: all `.rs`/`.py`/`.sh` under `src/`, `tools/`, `benches/`) and the reference comment corpus (SF at eca43a97 + sf_18 checkout, all `.cpp`/`.h`; Reckless all `.rs`): (1) whole-line normalized-exact matching (2,241 reference comment lines ≥ 5 words); (2) all shared 6-word shingles (13,122 reference shingles). Intersection is symmetric, so this covers "our comments in theirs" and "theirs in ours" in one pass. (Known small gap: C-style `/* */` block comments aren't parsed; SF/Reckless use `//`-style almost exclusively outside license headers.)

**Whole-line verbatim matches: 0.**

**Shingle hits: 9, of which 8 are generic English** ("must be a power of two" ×3, "the side to move is mated", "if the side to move has", a digit run in a table, the FEN field enumeration `fen.rs:148` ≈ SF's "5-6. Halfmove clock and fullmove number" — convergent descriptions of the same spec). The 9th is a **deliberate, attributed quotation**: `src/search/params.rs:598` quotes SF's one-line comment `(*Scaler) Making IIR more aggressive scales poorly` (`eca43a97:src/search.cpp:1029`), introduced as "SF's own source carries … — this is that comment, measured on ARCHON." A cited one-line quotation for engineering commentary; keep, it is the model of how to do this.

**Verdict: no transliteration tells.** A tree that had been translated line-by-line from SF or Reckless would light this sweep up; ours doesn't.

---

## 3. Licensing posture (repo-wide check a)

**Current state, verified:** no `LICENSE`/`COPYING` file at the root; no `license` field in `Cargo.toml`. Default: **all rights reserved** (the code is copyrighted to the author; nobody else may legally copy or redistribute it).

**What this audit domain contributes to the posture question:** no GPL/AGPL-derived *expression* found (no L3/L4). The SF-derived material in this domain is constants, data selections, names, and architecture — the categories D27 deliberately claims as unprotectable facts/methods. The Reckless comparison came back essentially L0/L1 everywhere. If the other two domain audits come back the same way, the whole tree is a clean idea-level reimplementation and the license choice is genuinely free; if either finds L3/L4 against SF or Reckless, distribution without a GPL-compatible license becomes a live violation (competitions distribute binaries; rating lists redistribute them).

**The decision the owner faces** (needed only at first distribution — private development requires nothing):

1. **Stay private / closed.** Legally safe *given this domain's findings* (facts-and-methods adoption is not infringement, and unpublished code distributes nothing). Costs: most rating lists (CCRL/CEGT/SPCC) test binaries you send them — that IS distribution, so the moment ARCHON enters a list, a posture is needed even if the source stays closed. Closed engines are accepted on lists and in events (Komodo Dragon precedent), so "closed + explicit permission-to-test grant" is workable. Publication ambitions (papers) don't force source release either.
2. **GPLv3.** The natural home: SF's own license, the community default for SF-architecture engines, and it retroactively moots every constants/selection question in this report (GPLv3 material may incorporate GPLv3 material). One-way door: source must ship with binaries forever. Recommended if the repo goes public at all.
3. **AGPLv3.** Only forced if some domain finds Reckless-derived *expression* (AGPL code can be combined into an AGPL project; GPLv3 cannot absorb AGPL obligations in reverse). This domain found none — so AGPL would be a voluntary extra restriction (network-use copyleft) with no compelling driver. Not recommended absent such a finding.
4. **Permissive (MIT/Apache-2).** Defensible ONLY on the strong claim that zero SF-derived protectable expression exists tree-wide. Given the D20 tunables transcription (260 constants, search domain) and F1/F2 here, this invites the exact argument GPLv3 forecloses. Not recommended.

Two adjacent invariants worth restating because they hold: the shipped NET is provenance-protected by D28 (SF-labeled data is dev scaffold, "never shipped"), and no SF/Reckless binary or network is tracked in the repo (`engines/canonical/*` are symlinks — path strings — with the bytes gitignored; verified).

**Recommendation:** decide *at first external distribution*, and default to **GPLv3** at that moment. Before any public release: drop F6's tracked profdata, optionally re-tune F1's seeds and swap F2's suite if a no-SF-text posture is wanted, and add the license header/`Cargo.toml` field in the same commit.

---

## 4. Originality census + differentiation

### 4.1 What is genuinely ARCHON-original in this domain

- **The oracle-first attack-table architecture** (`src/attacks.rs`): a `const fn` naive ray generator is the module's correctness oracle; the compile-time geometry tables are *defined in terms of it*, the runtime magic init *verifies against it* on every launch, and the test suite cross-checks both against a third, coordinate-walk model. SF initializes; ARCHON initializes-and-proves.
- **Magic tables with zero `unsafe` and a machine-checked bounds proof** (perf study 53): the shared table is allocated to a power of two, every reachable index is exhaustively asserted `< TABLE_SIZE` at init (through the hot path's own `Magic::index` arithmetic), and the single access site masks with `TABLE_MASK` so LLVM discharges the bounds check. A safe-Rust equivalent of SF's raw-pointer scheme with a stronger invariant than either SF or Reckless carries.
- **The StateInfo hybrid** (D8 as amended by perf study 14): live ply inline in `Board` + `Vec<StateInfo>` history — neither SF's caller-owned `StateInfo*` chain nor a clone-per-node board; measured into its current shape.
- **`perft_checked`** (`src/perft.rs`): a perft that runs the FULL board invariant oracle at every node — all placement layers, all six Zobrist keys vs from-scratch recompute, check bookkeeping, unmake round-trip identity. A movegen debugging instrument neither reference ships.
- **Zobrist determinism pins verified against an independent Python model** of the xorshift64\* stream (`zobrist.rs` tests) — the fill order is pinned as a tested contract, and our key stream is deliberately NOT SF's (different draw order).
- **`tools/ledger/` (D38)**: a content-addressed, append-only provenance DAG (nets/corpora/experiments, sha256 identity, regen recipes, annotation overlays, referential-integrity verify) — research-infrastructure originality well beyond chess-engine norms.
- **`tools/spsa/param_provenance.py`**: a tool that mechanically audits which SF revision our constants were transcribed from, with canary values that catch reading the wrong reference era. As far as known, unique to this project — and it is precisely the tool a license/provenance auditor wishes every SF-derived engine had.
- **The measurement discipline as infrastructure**: `check.sh`'s feature-matrix gate + the two-number NNUE behavioral gate (eval hash + node pin), the `Bench:` trailer contract, `build_pgo.sh`'s eval-file-required PGO policy, and the `engines/canonical/` symlink discipline (one indirection point for the release net/binary/arch triple). Individually small, collectively an unusually rigorous reproducibility story.
- **Self-documenting negative results in code**: hot-path doc comments carry the measured rejection of their own tempting alternatives (e.g. `attacks::tables()`'s "MEASURED AND REJECTED, TWICE OVER" banner with instruction-level numbers). Foundation code that argues from receipts is rare anywhere.

### 4.2 The Reckless null, stated with receipts (highest expression-risk pairing)

Exhaustive normalized line intersection between our 14 foundation files and all of Reckless `src/` found **only convergent single-line Rust** (enum variant lists, `let mut i = 0;`, derive attributes, obvious signatures like `pub const fn is_empty(self) -> bool`) — zero multi-line logic runs. Structurally the designs *diverge* at every decision point sampled: our `Bitboard` keeps a private field, a separate `BitboardIter` struct via `IntoIterator` (Reckless implements `Iterator` directly on `Bitboard`), `Direction`-enum masked shifts (Reckless: raw `i8` offset shift), `Option`-returning `lsb/pop_lsb` (Reckless: unchecked), `FromIterator` (absent there); our `Move` uses SF's 2-bit-kind + 2-bit-promo scheme (Reckless: 4-bit flag nibble with capture bit, decoded via `unsafe transmute` — ours has no `unsafe` in the entire foundation). Verdict for the pairing the brief flagged as highest-risk: **L1, no expression convergence found.**

### 4.3 The Integral null

Clone absent; notes deleted from the working tree (history-only, audited clean in §2.b); `grep -ril integral` over `src/` finds zero code citations — Integral references live only in design docs (D36/D37 study framing) and one nnue match script name. **No Integral-derived anything in this domain.**

### 4.4 Honest originality summary

The foundation is a *disclosed SF re-derivation at the level of architecture, names, semantics, and constants* (D20/D27 are the pinned policy, and the design docs cite chapter and verse), executed in independently written, more heavily specified, more heavily proven Rust. The originality of this domain lives less in the chess code — commodity by design — than in the verification and provenance infrastructure wrapped around it, which is presentable as a contribution in its own right.
