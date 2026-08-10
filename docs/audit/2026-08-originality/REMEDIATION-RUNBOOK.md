# Remediation runbook — post-training validation pass (2026-08 audit)

All remediation code was written and committed in a worktree while the box
was mid-training-run under a **hard no-cargo constraint** (a compile beside
the trainer has OOM-killed production jobs). Nothing here has been compiled.
This is the exact ordered pass to run once the training run is down.

Branch: `worktree-agent-add6887e326caacd0`. Commits, in order:

| commit | item | what |
|---|---|---|
| `46e884f4` | — | audit docs imported (they were untracked) |
| `1424f8c9` | 3, 5 | `old/tools/nnue/patches/` + `reckless_reference/pgo-prof-28/` deleted; `*.profraw`/`*.profdata` gitignored |
| `4540388c` | 4 | Reckless-note AGPL fragments → pseudocode (05, 07, 13) |
| `89c3dcf9` | 2 | scalar `activate`/`find_nnz` restated from the math + backend property test |
| `e3a8b78f` | 1 | `OUTPUT_BUCKETS_LAYOUT` → derived ARCHON layout + OBL1 container section + trainer mirror |
| `823c1287` | 6a | magic-seed harness (`#[ignore]`) + REMEDIATION-PENDING notice |
| `34af97b7` | 6b | ARCHON-original 49-position bench suite, staged NOT activated |
| (this)     | 7 | runbook + synthesis §2 statuses |

## Step 0 — merge, carefully

The LIVE checkout has **uncommitted modifications** to files this branch
also touches (seen at session start: `jax/archon_jax/{features,losses,model,
spec,train}.py`, `jax/tools/pump_train.py`, `src/eval/nnue/sota_context.rs`,
`src/eval/nnue/sota_head.rs`, `src/eval/mod.rs`, `src/search/mod.rs`).
Commit or stash those FIRST, then merge this branch (through
`chessengine-mainwt` per the standing worktree rule) and resolve overlaps —
the likely collision points are `spec.py` (new constants near the top),
`pump_train.py` (`build_cfg`), and `sota_head.rs` (the const block above
`SotaHead`).

No commit on this branch carries a `Bench:` trailer (per commissioning
instructions — the pin is recomputed here). Every change is designed to be
search-behavior-neutral, so the merge commit should reproduce the CURRENT
pins bit-for-bit; that reproduction is step 2/3's whole point.

## Step 1 — full test suite (the new tests ride in it)

```bash
cargo test
cargo test --release -- --ignored perft_full_suite   # attacks.rs was touched (D24 ritual)
cargo clippy --all-targets
```

New tests that MUST pass, and what each one decides:

- `sota_simd::tests::restated_scalar_kernels_match_every_backend_on_random_inputs`
  — **decides item 2.** The SIMD backends were left byte-untouched; the old
  scalar was pinned to them; the restated scalar matching them bitwise on 64
  seeded random rounds (contract domain, edges planted) proves the
  restatement changed nothing. The pre-existing `dispatched_*` tests
  corroborate. If this fails, the restatement has a real behavioral bug —
  fix the scalar, never the backends.
- `sota_head::tests::bucket_layouts_expand_their_ceilings_exactly` — the
  legacy ceiling expansion equals the historical 33-value table edge-for-edge
  (legacy-default preservation), and the ARCHON expansion equals the
  committed derivation receipt.
- `sota_head::tests::forward_reads_the_per_net_bucket_layout` — new-layout
  selection through the real forward pass; shared buckets bit-identical.
- `sota_loader::tests::obl1_round_trip_and_legacy_byte_identity`,
  `obl1_rides_after_the_other_trailers`, `obl1_rejects_bad_sections` —
  the OBL1 container contract (legacy nets serialize as a strict byte
  prefix; round-trip; validation).
- `bench_positions_archon::tests::staged_suite_is_valid_and_playable` —
  every staged bench FEN parses, is consistent, playable, unique.
- `attacks`: the whole existing battery re-proves `init_magics` after the
  `init_magics_seeded` refactor (determinism test included).

## Step 2 — the NNUE gate: the OBL1 design's acceptance test

```bash
CHECK_NNUE_GATE=1 bash tools/check.sh
```

**Expectation: NOTHING moves.** Eval hash `751d0f12…7a1b` and the d13 node
total `1815389` must be reproduced exactly — the canonical net's container
has no OBL1 section, so it must load under the legacy default and evaluate
bit-identically. The depth-6 `Bench:` line must print the current pin
(`351273`). If the eval hash moves, the legacy-default path is wrong —
that is a STOP, not a re-pin: no commit in this branch is allowed to move
eval behavior.

## Step 3 — trainer parity smokes (cheap, already half-verified)

The Python side was executed during remediation (no-cargo only binds Rust):
spec expansion == historical table, OBL1 emitter exact, netpack splitter
across 9 trailer cases. Post-merge, additionally:

```bash
.venv/bin/python jax/tools/train_smoke.py        # end-to-end smoke, legacy default
```

Nothing passes `arch_bucket_layout` yet, so every existing run json and
re-export is byte-identical by construction.

## Step 4 — magic seeds (item 6a, execution-gated half)

```bash
cargo test --release -- --ignored find_own_magic_seeds --nocapture
```

Paste the printed `MAGIC_SEEDS` row into `src/attacks.rs`, drop that
constant's REMEDIATION-PENDING notice (update the attribution to name the
harness + sweep parameters), rerun `cargo test` + the fast perft tier.
Seeds affect init time only, never search: the commit is non-functional and
must reproduce the existing `Bench:` bit-for-bit (verify via check.sh).

## Step 5 — bench suite activation (item 6b; moves the Bench: pin BY DESIGN)

When ready (any time before distribution; it invalidates historical
fingerprint comparability, so bundle it with a natural pin move if one is
pending):

1. In `src/bench.rs`, point the default suite at the staged one:
   replace `BENCH_POSITIONS`'s body with a re-export of
   `crate::bench_positions_archon::BENCH_POSITIONS_ARCHON` (and move the
   provenance doc comment accordingly); delete the SF-provenance note.
2. `bash tools/check.sh` → take the NEW depth-6 `Bench:` line.
3. `CHECK_NNUE_GATE=1 bash tools/check.sh` — the gate's eval hash and d13
   total are computed on `tools/nnue/calib/wl50.fens`, NOT the bench suite,
   so they must NOT move. Only the depth-6 default-suite trailer moves.
4. One commit: the swap + the new `Bench:` trailer + CLAUDE.md's bench-pin
   paragraph updated (check the CURRENT slot moved, not its history — the
   `859a442` lesson) + design/06 §3.6's suite provenance note.
5. Update `design/14` §3.2's fingerprint history note: trailers before this
   commit were recorded on the SF-derived suite and are not reproducible by
   the new one.

## Step 6 — first net under the ARCHON bucket layout (next training gen)

In the next big run's json: `"arch_bucket_layout": "archon"`. That single
key drives the loader's batch labels (TrainConfig via `build_cfg`) AND the
quant export's OBL1 section (ArchSpec via `from_run_spec`). On export,
`netpack.py` passes the section into the `.cenn`; the engine selects the
net's own buckets. Gate the first such net with
`jax/tools/engine_parity.py` — NOTE the small follow-up below.

## Deferred / follow-ups (each with its reason)

1. **Everything Rust is compile-deferred** (items 1, 2, 6a-harness, 6b-tests):
   the box could not run rustc beside the trainer. Steps 1–2 discharge it.
2. **Magic-seed replacement** (6a): requires executing `init_magics` to
   price seed rows — step 4. The shipped row is untouched and marked.
3. **Bench-suite activation** (6b): moves the D24 pin, which must be
   recomputed in the activating commit — step 5.
4. **Parity tools + OBL1 nets**: `jax/tools/engine_parity.py` /
   `forward_parity.py` call `map_batch` without `bucket_layout`, i.e. legacy
   — correct for every net that exists today. Before gating the FIRST
   archon-layout net, thread the run json's `arch_bucket_layout` into their
   `map_batch` calls (a two-line change each), or their JAX-side bucket
   selection will disagree with the engine's.
5. **`netpack.py strip_d40_threats`** already refused any container with
   trailers (pre-existing); OBL1 does not change that. Noted, not fixed.
6. **License file choice** (synthesis §2 item 7, GPLv3 at first
   distribution) is a distribution-time owner decision, out of scope here.

## Honesty ledger (divergences from the audit, and open claims)

- **Bit-identity of the restated kernels** is argued by construction
  (operation-for-operation identical arithmetic, reshaped iteration) and is
  DECIDED by step 1's property test against the untouched backends — not by
  this document.
- **The derived bucket layout differs from Reckless's** on 17/33 piece
  counts (ceilings `[8,10,13,15,18,21,25,32]` vs `[8,12,16,19,22,25,28,32]`);
  the first ceiling coincides at 8 (that is where 15.0% of our sample mass
  lands) — flagged in-source per the audit's optics rule.
- **Audit accuracy**: findings matched the tree everywhere it mattered.
  Line numbers were off by a handful (OUTPUT_BUCKETS_LAYOUT at 221 not 226;
  `activate` at 1172 not 1181) — cosmetic. The note-fragment count was
  stated as "five fragments (05, 07)"; in reality 05 had 4 fences, 07 had 3
  quoted/near-quoted blocks (TT-cutoff, probcut, lerp), and 13 had 1
  rust-tagged structural sketch — all 8 were converted, a superset of the
  checklist item.
- **The bullet-side copies** of the Reckless table
  (`tools/nnue/bullet/d40_{features,policy}.rs`) were NOT in the audit's
  checklist but carried the same literal text; restated as computed ranges
  with attribution (behavior identical — legacy).
