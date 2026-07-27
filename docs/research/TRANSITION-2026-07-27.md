# TRANSITION — 2026-07-27 (session handoff)

Written at the end of a long session. Read this first; it is the authoritative
state. Everything below is committed on `main` unless marked otherwise.

---

## 1. LIVE STATE — what is running right now

| thing | state |
|---|---|
| **D48X training** | **sb324/520**, `archon-d48x.service` active, ~175k pos/s, heldout EMA ~0.00678 |
| LR schedule | flat **5.5625e-4** to sb400, then linear anneal to 0 at sb520 |
| ETA | ~22:55 |
| Restart policy | `Restart=on-failure`, resume point AUTO-DETECTED |
| TensorBoard | `archon-tb-server.service` on **:6006**, shim `archon-d48x-tb` active |
| SPSA | **STOPPED** (died in the 12:27 reboot at k=1130/2400). Deliberately not restarted — user set it aside for the reasoning-based approach. |
| c3 chain | gone with the reboot |
| Loop | cron job `00813083`, every 15 min, session-only |

**To restart the trainer after any death:** `bash tools/nnue/d48x_launch.sh`.
It auto-detects the newest complete checkpoint pair and relaunches. Do NOT
hand-roll the invocation — plain `uv run` FAILS on `jax/` (the `notebook`
extra will not resolve under py3.13); the working form is
`uv run --no-project --with numpy==2.5.1 --with "jax[cuda12]==0.11.0"
--with optax==0.2.8`, which is what the launcher does.

---

## 2. THE HEADLINE — the eval-frame investigation is a CLOSED NEGATIVE

We spent the session on one hypothesis, tested it two independent ways, and
**refuted it**. This is a real answer, not an abandonment.

**The measurement stands and is solid.** Our static eval is **≈0.58× SF18's
internal frame**, confirmed three independent ways: 0.55 winner-slope from
game outcomes (2026-07-26), 0.579 on wl50 and 0.605 on WAC from static eval
(`tools/eval_scale_study.py`). And our cp-denominated constants really do sit
at Stockfish's **raw** values — ten checked, capture-futility base 231 vs 232
and aspiration delta 5 vs 5 among them. So the margins genuinely are ~1.7×
larger relative to eval than Stockfish's are.

**What is refuted is that exploiting this is worth Elo.** Two formulations:

| approach | result |
|---|---|
| scale ~40 margins DOWN by 0.58 (`FRAME_S`) | **negative** — 10/10 tactical arms worse, no positive game arm |
| PRUNE-family only | **wash**, +3.5 ± 31.7 |
| HISTMG family | **wash**, +9.27 ± 17.73 over 600 games |
| scale the EVAL UP by 1.72 (`EVAL_SCALE`) — the dual | **negative** — mate_in_3 1211 → 1194 → 1177, monotone |

The dual matters because it also repaired **class 4** (material-vs-eval), the
one part no margin rescale could reach. So "class 4 was the missing piece" is
tested and is not the explanation either.

**The cleanest statement of the refutation:** the correction degrades tactics
and Elo at *essentially unchanged search volume* (the global tree moves only
+3.6% on NNUE). It is not trading tree size for accuracy — it is making worse
decisions at the same cost. The constants are already near a genuine optimum,
which is consistent with our tree already being smaller than Stockfish's at
equal depth (perf study 7 §1.2). There is no under-pruning to recover.

**Useful corollary:** search-margin mis-calibration is now ELIMINATED as the
explanation for the −33 Elo equal-node deficit. Remaining effort belongs on
eval quality and instruction count, per
`docs/research/where-the-sf16-gap-actually-is-2026-07-27.md`.

Full detail: `docs/research/eval-frame-scale-model-2026-07-27.md` (read §6.5's
REFUTED banner, §9, §10, §10.1).

---

## 3. THE BUG THAT COST THE MOST — read this before trusting any node count

`cmd_bench_session` **parsed bench's optional net argument and silently
ignored it**. The argv front door loaded it; the UCI-session form did not.

```
argv:         bench 16 1 13 wl50 depth <net>  ->  2,068,701  (NNUE)
UCI session:  bench 16 1 13 wl50 depth <net>  ->  3,536,295  (PSQT, dropped)
```

Fixed in `36966dd`. But **every node-count measurement taken through a UCI
pipe before that fix was PSQT**, including a full day of ablations and the
entire frame-scale sweep. Re-measuring flips signs:

```
                  PSQT      NNUE
RAZOR_BASE       -13.26%   -5.74%
PFUT_BASE        -14.12%   +8.46%   sign flip
QS_FUTILITY      +3.77%    -5.92%   sign flip
```

**Rule going forward:** always activate the net with `setoption name EvalFile`,
never rely on bench's 6th argument. `tools/frame_ablate.py` is the reference.

---

## 4. RETRACTIONS — claims made and withdrawn today

Recorded because each was confidently stated and each was wrong. The pattern
is uniform: **every one was caught by measuring a second way, not by reasoning
harder.**

1. **"SF16→SF18 constant ratios show a 38% frame transfer rate."** Killed by
   the co-movement test — `singularBeta` moved 82→53 beside 65→75 in the same
   expression. A frame shift is a scalar multiply and must be uniform within a
   formula. My sample was selection-biased (I picked razoring and probcut,
   which grew; never checked singularBeta or optimism, which shrank).
2. **The node-count direction test as a classification criterion.** Node count
   is not monotone in pruning aggressiveness.
3. **"`correction_value` is untouched by our eval frame"** (§5.1). False —
   `history.rs:743` feeds it our own eval errors. The verification pass
   derived the gravity stationary point and settled it as class 5.
4. **`RFP_THREAT_MARGIN`'s +11% ablation was "anomalous."** It enters its
   margin with a MINUS sign; shrinking it enlarges the margin. Correct
   first-order behaviour I misread.
5. **`frame_scale` truncated instead of rounding.** `SE_DOUBLE_BASE = −3`
   became −1 (ratio 0.33, not 0.58) and `ASPIRATION_BASE_DELTA` was 2 at BOTH
   s=594 and s=470 — two sweep arms silently identical.
6. **The implementing agent's own §9.1 mechanism** (tree explodes → depth lost
   → tactics fall) rested on a PSQT +78% that is +3.6% on NNUE. Withdrawn by
   the agent itself.

---

## 5. WHAT LANDED (all gated, all on main)

- **`perf-impl-1` merged** — bitboard threat events + movepick stage-init
  outlining. **−5.65% instructions / −8.65% L1d**, node-identical, eval
  bit-exact. Worth ~+6 to +10 Elo. Both wins were mechanisms perf study 7
  named *wrongly*.
- **Upcoming-repetition** (P9 item 19's last piece) — cuckoo tables finally
  have a caller. Bench 385329 → 445174; +15.5% is almost entirely bench
  position 37 (artificial, pawnless), −28.8% across the other 48. Mate-in-3
  safety check clean (1211 vs 1210).
- **Loader bulk-read** — payload readers looped per element with bounds AND
  capacity checks on each of ~84M. **1.677 G → 0.591 G instructions (−64.8%)**.
  Worth ~8.6% more SPSA/screen games per box hour.
- **The NNUE behavioural gate** (`CHECK_NNUE_GATE=1`) — the depth-6 bench pin
  is a PSQT number and is BLIND to NNUE work. Caught two real pin moves today.
- **Triple-extension deep gate fixed** — SF18 uses two different predicates
  (`ply > rootDepth` for double, `ply*2 > rootDepth*3` for triple); we passed
  one to both. d13 NNUE 2068701 → 2041687.
- **`EVAL_SCALE`** — exists, inert at 1024, swept and refuted (§2).
- Reference clone moved to **`sf_18`** (was `sf_16`, which silently returned
  SF16 constants).

**Current pins:** bench **445174**, eval hash **cd190d1b…7f3d**, d13 NNUE
**2041687**.

---

## 6. OPEN THREADS

1. **The params refactor — never started.** `params.rs` is 1127 lines of ~238
   flat tunables. The agreed ordering, which must be preserved: refactor
   proven inert FIRST (bench + eval hash + d13 + `cargo test --features
   sota`), factors wired at identity SECOND, measurement THIRD — each its own
   commit, so every change stays attributable.
2. **The unmerged frame-scale branch** — `worktree-agent-a9b8c2ccac18ab5bf`,
   6 commits. Contains the 238-row classification (read from source, still
   valid), the `FRAME_S`/`FRAME_SX`/`FRAME_SC` wiring, and
   `tools/frame_family.sh`. Given the negative verdict, merging is optional;
   the classification table is the part worth keeping.
3. **Class 4 is real but unexploited** — `PIECE_VALUES` are Stockfish's
   exactly while our eval is 0.58× theirs, so a captured piece weighs 1.72×
   too much against eval. `EVAL_SCALE` tested the repair and it did not pay,
   but the *mismatch* is still real and may matter elsewhere.
4. **SMP scaling still unmeasured with NNUE.** `tools/smp_scaling.sh` is
   written and has a hard idle-box guard. Time-to-depth, not NPS — lazy SMP
   duplicates work, so 6.94× NPS is compatible with a 2–3× real speedup.
5. **Strength testing owed** on `perf-impl-1` and upcoming-repetition.

---

## 7. TRAPS THAT COST TIME TODAY

- **`pkill -f` matches your own shell.** Killed the turn twice (exit 143/144).
  The `$$`/`$PPID` filter does NOT fix *detection* either — a `pgrep` in a
  subshell still self-matches. Working form: exclude anything whose cmdline
  mentions `shell-snapshots`, or read `/proc/<pid>/cmdline` directly. Better:
  `systemctl --user is-active <unit>`.
- **I OOM-killed the trainer** by running a PGO build (two LTO rustc jobs)
  beside it. 15 GB total, trainer at ~8 GB. `free -g` before ANY build;
  `-j 4` max; never `build_pgo.sh` beside the trainer. A cgroup `MemoryHigh`
  does NOT protect against a global OOM.
- **`bench` writes its whole report to stderr** — capture with `2>&1 >/dev/null`.
- **/tmp is tmpfs (RAM)** — the trainer log died with the reboot.
