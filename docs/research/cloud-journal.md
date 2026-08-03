# Cloud journal

Running log of everything ARCHON has run on GCP. **Append a row when a job is
submitted and edit it when the job reaches a terminal state.** Record the
expectation BEFORE the result, so a surprise is visible as a surprise rather
than rationalised afterwards.

Tooling: `tools/ops/cloud/` (README there). Design: this file is the operational
record; `docs/research/cloud-data-campaign.md` is the older design doc and its
quota figures are **stale** — see Standing facts below.

---

## Standing facts (verified 2026-08-03)

| item | value |
|---|---|
| project | `archon-502500` |
| region | `us-east1` |
| bucket | `gs://archon-main-bucket` (corpora 242.65 GiB, sf18-relabel 79.45 GiB, nets 2.39 GiB) |
| worker identity | `archon-batch-worker@archon-502500.iam.gserviceaccount.com` |
| worker roles | project: `batch.agentReporter`, `logging.logWriter` · bucket: `objectCreator`, `objectViewer` |
| base image | `gcr.io/google.com/cloudsdktool/google-cloud-cli:stable` + apt `python3 zstd` |
| machine | `c2d-highcpu-16`, **SPOT** |

**Quota — the design doc is wrong about this.** It plans around 17,600 vCPUs and
records four quota requests as "accepted and reconciling". Actual:

| quota | doc assumes | actual |
|---|---|---|
| `CPUS` | 20,000 | **300** |
| `C2D_CPUS` | 20,000 | **300** |
| `PREEMPTIBLE_CPUS` | 20,000 | **0** |
| `INSTANCES` | 1,250 | 1,250 ✓ |

Only the instance request landed. `PREEMPTIBLE_CPUS = 0` turned out to be
harmless: modern Batch `SPOT` draws on the regular `CPUS` quota, proven by
canary 1 reaching RUNNING.

**⚠ THE QUOTA ROWS ARE NOT THE EFFECTIVE CEILING. MEASURED CAP = 96 vCPU.**
`CPUS`/`C2D_CPUS` read 300, and I planned job #4 at parallelism 18 (288 vCPU) on
that basis. It settled at exactly **6 concurrent tasks / 96 vCPU** — 6 live VMs,
67 tasks PENDING, stable over minutes, not ramping. Owner confirms 96 is the cap.
Whatever binds (Spot-specific ceiling, Batch scheduling limit, or regional Spot
capacity) is not the row the tooling was reading, and reading it made every
wall-clock estimate **3x optimistic**. Pass `--vcpu-limit 96`, and confirm
against a live job before trusting a new number.

**Cost model** (worst case; prices are unverified list figures):
`$0.012/vCPU-h` compute, `$0.12/GB` egress, **2.90 vCPU-h per 1M kept
positions** measured locally. ~73 M positions/h at parallelism 18.

**Known ceiling:** oracle-vs-oracle is deterministic, so a replayed opening
yields a byte-identical game. The UHO book (2,632,036 lines) therefore caps a
campaign at **~150 M unique positions**. See "Opening supply" below.

---

## Jobs

| # | job | submitted | shape | expected | outcome |
|---|---|---|---|---|---|
| 1 | `oracle-canary-canary-20260803-184238` | 18:42 | 1 task, 100k pos | prove Spot/quota/GCS/image | **FAILED** exit 127 — `gcloud: command not found`. Entrypoint used `bash -lc`; a login shell sources `/etc/profile`, resets PATH, drops `/usr/lib/google-cloud-sdk/bin`. Did prove Spot works under `PREEMPTIBLE_CPUS=0` (reached RUNNING). |
| 2 | `oracle-canary2-canary-20260803-184735` | 18:47 | 1 task, 100k pos | same, with `-c` fix | **FAILED** exit 3 — `Illegal instruction (core dumped)`, rc=132 in stage 1. Binaries built `-march=native` on Zen 4 **with AVX-512**; c2d is Zen 3 with none. fastchess 5,302 AVX-512 insns, Stockfish 1,363. **A local `docker run` cannot catch this** — the container uses the host CPU. |
| 3 | `oracle-canary3-canary-20260803-185921` | 18:59 | 1 task, 100k pos | same, with portable binaries | **SUCCEEDED** in ~4 min. 101,331 records, 58.27 rows/game, 22.58 % D46 drop. Fetched back, sha256 verified on arrival. |
| 4 | `oracle-uho1-run-20260803-191039` | 19:10 | **73 tasks**, par 18 requested, 145 M target | ~2.1 h wall, **$6.86** worst case; 73 shards × ~2 M. Sized to 73 of the book's 75 available shards so it never wraps. | **RUNNING** — but only **6 tasks run at once** (96 vCPU cap), so the real wall clock is ~12 waves × 0.43 h ≈ **5.2 h**, ETA ~00:20. The 2.1 h estimate assumed parallelism 18 and was 3x optimistic. Cost is unchanged (same total vCPU-h). |

---

## Opening supply, and why it stops being the constraint

A campaign cannot exceed `unique_openings x ~58 kept positions/game` **while
play is deterministic**. Both players are Stockfish at a fixed node budget, so a
replayed opening reproduces the game byte for byte. With the UHO book
(2,632,036 lines) that caps a campaign at **~150 M unique positions**, which is
why job #4 is sized at 145 M.

**OWNER RULING 2026-08-03: solve this with TIMED GAMES, not more openings.**
Wall-clock play is non-deterministic — search depth varies with whatever the
machine does that millisecond — so the *same* opening yields a *different* game
every replay. That turns 2.6 M openings from a hard ceiling into a renewable
supply and removes the position-sourcing problem entirely.

**The variation IS the product, not a tolerance to be minimised** (owner,
2026-08-03). A few percent of timing jitter changes which move gets played at a
handful of decision points per game; downstream that is a different game, a
different result, and a different set of positions. The engine is unchanged and
still enormously strong at this budget, so every one of those games is still
oracle-quality signal.

**MEASURED 2026-08-03, and it is a BOUNDED multiplier, not a renewable supply.**
An earlier revision of this file claimed timed games "removes the
position-sourcing problem entirely". That was written from the argument, before
the measurement, and it is wrong. Replaying ONE UHO opening:

| variant | replays | distinct games | new vs earlier variants |
|---|---|---|---|
| `st=0.05` both sides | 48 | **4** | 4 |
| `st=0.05` v `st=0.04` | 32 | **9** | +5 |
| `st=0.05` v `st=0.03` | 24 | 3 | **0** |
| `st=0.04` v `st=0.045` | 24 | 4 | **0** |
| `Threads=2`, `st=0.05` | 32 | 8 | (costs 2x CPU) |

Three findings:

1. **It saturates almost immediately.** Distinct games after N replays at
   `st=0.05`: 4->3, 8->3, 16->3, 24->3, 32->4, 48->4. The most common line
   appeared **25 times in 48 replays**.
2. **Asymmetric time is the best single lever** (owner's suggestion) — 9
   distinct from 32 replays against 4 from 48 symmetric, i.e. ~3.4x the
   divergence per replay.
3. **Different pairings do NOT stack.** The 3rd and 4th time pairings
   contributed **zero** new games; every line they produced was already in the
   union. There appears to be a small fixed set of reachable continuations per
   opening that the time budget does not enlarge.

Practical capacity: ~3 distinct games per opening at good efficiency (~80 %),
~9 at very poor efficiency (~19 %). So the ceiling moves from ~150 M
(deterministic) to roughly **450 M**, not to infinity. That is ample for an
8-hour run (~215 M at `st=0.05`) but it is a ceiling and should be planned as
one.

**CAVEAT: the saturation curve is n=1 opening.** A 3-opening check at 8 replays
gave 3/4/4 distinct, which is consistent, but the plateau height itself is
measured on a single position and may vary with how forcing the opening is.

If more headroom is needed later, the lever that is known to work is the one the
PREVIOUS cloud generator used and this one dropped: **MultiPV with a temperature
on move choice** (`native_game_generator.py` ran `--multipv 4 --temperature-cp
80`). That samples among near-equal moves instead of hoping the clock flips one,
and it is not bounded by search stability.

Read `gen_games.sh`'s existing `nodes=N tc=inf` header note in that light. It
warns that a wall-clock run "would silently produce weaker games whenever the
box got busy, and the WDL labels would inherit that variation" — written for
THIS shared box, where load swings by an order of magnitude and would push play
strength around unpredictably. A Batch task owns its vCPUs at pinned
parallelism, so the residual jitter is small, bounded, and the thing we want.

Design points for the first timed campaign:

* **Equivalent time control.** Measured 2026-08-03: eca43a97, 1 thread, on this
  box under load = **368k nps**, which is close to what a cloud task sees with
  16 SF processes on 16 vCPUs. The current 5,000-node operating point is
  therefore **~13.6 ms/move**. An idle cloud vCPU will be faster, so re-measure
  on the target machine and set the movetime to match 5k nodes there.
* **Prefer `st=` (fixed time per move) over a full `tc=`.** `st=` keeps
  positions-per-game and shard sizing predictable while still varying the node
  count per move; a full time control adds the engine's own time management,
  which is more realistic but makes yield harder to size.
* Label quality is unaffected: labels come from the in-game score at whatever
  the engine actually searched, and `label_inline.py` records it either way.
* Reproducibility moves from "same inputs, same corpus" to "same inputs, same
  *distribution*". The shard-level byte-equivalence check in job #3 retires and
  becomes a statistical comparison.

**Investigated and then SCRAPPED per the ruling** — recorded so it is not
re-derived later. Local `data/nnue/binpacks/uho_frs_filtered.tsv` holds
131,157,609 full-state FENs (castling, ep, clocks, plus result/ply/score);
filtering to `|score| <= 200 and ply <= 40` keeps 7.60 % => **~10 M contested
openings, 0.0 % duplicates in a 3 M-row sample, 0 malformed**, which would have
lifted the ceiling to ~580 M. Also available and unmeasured:
`gs://.../corpora/lichess-chess-evaluations/` (342,059,879 unique positions,
7.91 GiB) and `gs://.../corpora/official-stockfish/master-binpacks/` (232 GiB of
full games). None of this is needed if games are timed.

## Operating rules learned here

1. **`DEST` may be a local directory.** The worker was exercised end to end on
   this box before any cloud access existed, which found six defects for free.
   Keep that property.
2. **A local container cannot test ISA portability** — it runs on the host CPU.
   `build_payload.sh` now refuses any binary containing `%zmm`.
3. **Canaries earn their cost.** Two of four jobs failed on things no amount of
   local testing would have surfaced. Always `--canary --i-mean-it` first.
4. **Record the expectation before the result** in the table above.
