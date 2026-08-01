# Perf study 66 — the compacted row is BUILT and BIT-EXACT and it loses; the indirection is the cost, and locality converts NOTHING

**Date:** 2026-08-01 · **Box:** 7700X (Zen 4), AVX-512 VBMI2, SMT on, 32 KB
L1d, 1 MB L2/core, 32 MB L3, 15 GB · **Base:** `main` at **`41f3b41b`**
(post-study-65 handoff, after the SF `ebcea3ef` tunable refresh) · **Net/arch:**
D48X @ `network_scale=600`, `sota,arch-d48x`, ACC 1280 · **Branch:**
`worktree-agent-a8f4de0543dff70ae`

**Frame, quoted with every number below:**

```
bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
```

50 FENs × 100 k nodes = **5,001,560 nodes**, one thread, 16 MB TT,
`-C target-cpu=native`, non-PGO, pinned to core 2. A GPU training run and a
1500-game match shared the box throughout. **Every speed claim is made against
the arm's own null in the SAME BINARY** — one `.text`, five data modes selected
by environment variable — so the box's load cancels in every contrast. No
`cycles:u` appears anywhere in this document (studies 43/63's standing refusal);
`task-clock` does, because it is what the brief asked to be judged on and
because it is quoted only as a within-round paired median with its sign count.

---

## ★ 0. WHAT LANDED, AND THE THREE ONE-LINE ANSWERS

**Landed and gate-green** (`CHECK_NNUE_GATE=1 bash tools/check.sh` → exit 0,
`Bench: 486897`, eval hash `2e1ede95…79c4`, d13 `1743308`, **all three pins
unmoved**):

1. **`SotaNetwork::threat_rowmap`** — a per-feature **byte-offset map** (bit 0 =
   stored narrow) replacing study 65's narrow BITMAP. A bitmap can only say
   *whether* a row is packed, which is what forced study 65 to keep the
   1280-byte slot; a map says *where*, which is what makes compaction possible
   at all.
2. **`pack_threat_plane`** — three physical layouts as three DATA modes of one
   binary (`ARCHON_NARROW=0` flat i8; `ARCHON_PACK=0` study 65's pack-in-place;
   `ARCHON_PACK=1` study 66's compacted two-plane), plus two more
   (`ARCHON_PERM=id|1`) for the locality arm.
3. **`foldcount::note_fold_redundancy`** — the intra-fold census (§1): exact
   cancellation, repeats, and the address SPAN of a fold's rows.
4. **`threat_rows_equal`** — `SotaNetwork` equality now compares the threat
   table logically, row by row through each net's own map. Byte comparison made
   `load ∘ serialize` look non-identical over a 64-byte pad.
5. **`tools/perf66/`** — build, frame, hash, panel, pairwise, identity, derive.

**The compaction answer (deliverable 1, study 65 §4 item 2's "first thing a
successor should do"):**

> **Built, bit-exact, and it does not rescue the route.** Removing the dead
> 320-byte tail *entirely* converts **−0.81 % of the +19.74 % L2 misses study 65
> blamed on it** (6/8 rounds), and `l2_pf_hit_l2.all` is still **+18.09 %** with
> no tail left to fetch. **Study 65's dead-tail-plus-prefetcher mechanism is
> refuted.** The prefetcher over-fetches past the end of any shortened row;
> whether the bytes beyond it are dead or belong to the next row barely matters.

**The cost answer (§3), which is the useful part:**

> Decomposed inside one binary and one set of rounds, the narrow-row route's
> cost is **not the decode and not the tail — it is the INDIRECTION**. The row
> map alone, with no packing and no permutation, costs **+6.02 % instructions,
> +6.49 % task-clock and +15.28 % L2 misses** (0/8 rounds negative on every
> counter). The 6-bit pack on top of it buys **−2.21 % L2 misses** for
> **+7.63 % instructions and +2.47 % task-clock**. So even with a *free* map the
> route still loses on wall clock, and study 64 §6.1's "243 KB permutation
> table" — the thing study 65 designed the whole slot-preserving layout to avoid
> — was the correctly identified cost all along.

**The locality answer (deliverable 2, the brief's routes 1 and 3):**

> **A whole-table row permutation that cuts the share of a fold's row PAIRS
> sharing a 4 KB page by 3.32× (2.794 % → 0.842 %, physical offsets, measured)
> converts NOTHING**: +0.03 % instructions, **+0.41 % task-clock at 4/8 rounds —
> a coin flip** — +1.51 % L2 misses at 3/8, +0.30 % L1d misses at 3/8. It is
> bit-exact and would cost zero at run time if baked into the resolver's LUTs.
> The perturbation landed in the **worse** direction (§4.2 — I predicted better
> and the census says otherwise), which makes it a *two-sided* null: moving a
> fold's rows substantially further apart changes no counter, so moving them
> closer cannot be worth much either. Locality within a fold is not the binding
> constraint, and "visit rows in address order" / "make the refresh's rows
> contiguous" are closed with a number rather than an argument.

---

## ★★★ 1. THE CENSUS NOBODY HAD: INSIDE ONE FOLD

Every census before this one is per-ROW (studies 51/61/65) or per-FOLD-LENGTH
(studies 61/62). Neither can answer the two questions a layout or ordering arm
has to ask *before* it is built. `foldcount::note_fold_redundancy` charges both
at the fold, whole frame, node-identical and eval-hash exact:

| channel | folds | rows | **cancel** | repeat | already ascending |
|---|---:|---:|---:|---:|---:|
| `hka` | 4,958,599 | 11,305,854 | **0** | 0 | 1,905,541 (38.4 %) |
| `thr` | 4,316,728 | 19,714,419 | **379,800 (1.93 %)** | 0 | 1,328,344 (30.8 %) |

**Redundancy.** A feature in both the add and the sub list contributes
`+w − w = 0` to every lane, so the pair is removable and the fold stays
bit-identical (wrapping i16 addition commutes). It happens **1.93 % of the time
in the threat channel and never in HalfKA**, and no row is ever listed twice.
1.93 % of threat rows is **0.076 rows/node of 7.241** — 1.05 % of all folded
rows — against an `O(NA·NS)` detection at ~5 comparisons per fold. **Priced out,
not built**, and reported so nobody prices it from a guess.

**Address spread**, `log2` of the byte distance from a fold's lowest to its
highest row base:

| channel | modal bucket | share | row PAIRS within one 4 KB page |
|---|---|---:|---:|
| `hka` | **2¹⁴ = 16 KB** | 34.2 % | 811,429 / 7,735,911 = **10.5 %** |
| `thr` | **2²⁶ = 64 MB** | 31.1 % | 1,727,734 / 61,834,924 = **2.8 %** |

> ### The threat fold's rows are as scattered as the 77.7 MB table allows; HalfKA's sit inside its per-king-bucket slice.
>
> HalfKA's 16 KB modal span is the geometry: for one perspective and one king
> bucket the live rows are 704 contiguous rows (901 KB), and a fold's two rows
> are the same piece on two squares. The threat channel has no such locality —
> §4 explains why from the index's own shape, and §4.3 measures what fixing it
> is worth.

**Temporal reuse** (`FOLDROWREUSE`, study 61's instrument, re-read here because
§4 needs it). A fully associative cache holding the last `W` distinct rows gets
at least `CDF(W)` hits:

| channel | touches | distinct rows | `W < 32` (≈ L1) | `W < 1024` (≈ L2) | `W < 32768` (≈ L3) |
|---|---:|---:|---:|---:|---:|
| `thr` | 19,670,258 | 44,161 | **33.4 %** | 84.7 % | 97.4 % |
| `hka` | 16,078,879 | 20,090 | 30.8 % | 79.1 % | 97.1 % |

The whole frame folds **36,216,247 rows = 7.241 rows/node = 9,268 B/node =
144.8 cache lines/node**, split `thr_into` 54.44 %, `hka_into` 31.22 %,
`hka_dual` 13.23 %, `thr_refresh` 1.11 %, `hka_inplace` 0.00 %.

---

## ★★★ 2. THE COMPACTED LAYOUT: BUILT, BIT-EXACT, AND STUDY 65'S MECHANISM IS REFUTED

### 2.1 What it is

Study 65 packed every ≤ 6-bit FullThreats row into 960 B **inside its own
unshrunk 1280 B slot**, so `RowOffset::of(f) = f · ACC` survived and the route
cost no offset table. The price was a dead 320 B tail in every narrow row, which
its §2.3 blamed for the loss, and its §6 item 1 named the compacted layout as
the surviving version.

`ARCHON_PACK=1` builds it: **wide plane at 1280 B stride, then narrow plane at
960 B stride, then a 64-byte pad.** `960 = 15·64` and `1280 = 20·64`, so every
row still starts on a cache line and a narrow row occupies **exactly 15 lines
with nothing dead behind it**. The pad absorbs the decode's 8-byte tail
overread on the last narrow row. Addressing goes through
`threat_rowmap[f]` — a `u32` byte offset with the narrow flag in bit 0, which is
free because both strides are even.

Measured table: **77,721,600 B → 60,930,624 B (−21.6 %)**, i.e. **52,472 narrow
rows (86.42 %) and 8,248 wide (13.58 %)** — reproducing study 65 §1's static
86.4 % exactly, from a completely different computation.

### 2.2 It is bit-exact

| gate | result |
|---|---|
| node total, all five data modes | **5,001,560** — exact |
| wl50 eval hash, all five data modes | **`2e1ede95…79c4`** — exact |
| `cargo test --features sota,arch-d48x,narrow65` | **724 passed, 0 failed, 8 ignored** |
| `narrow_rows_fold_bit_exactly` | now folds the same rows from **both** packed layouts with a mixed map and demands the same accumulator |
| `serialize ∘ load` | the identity again — the four round-trip tests were failing on a 64-byte pad until `threat_rows_equal` compared the table LOGICALLY (§5) |
| `CHECK_NNUE_GATE=1 bash tools/check.sh` | **exit 0**, `Bench: 486897`, eval hash `2e1ede95…79c4`, d13 `1743308` — all three pins unmoved |

### 2.3 And it loses — 8 shuffled rounds, one binary

Reference is `ARCHON_NARROW=0`: the same `.text`, the same dispatch, the flat
i8 table. Medians of within-round ratios, with the sign count.

| exact counter | ref (median) | `pack0` (study 65) | `pack1` (study 66) |
|---|---:|---:|---:|
| `instructions:u` | 27,791,772,409 | **+14.091 %** 0/8 | **+14.102 %** 0/8 |
| `task-clock` | 4193 ms | **+11.898 %** 1/8 | **+8.667 %** 0/8 |
| `L1-dcache-load-misses` | 2,793,960,292 | +4.035 % 0/8 | +3.736 % 0/8 |
| `l2…dc_access_in_l2` | 2,906,830,496 | +3.914 % 0/8 | +3.693 % 0/8 |
| `l2…ic_dc_miss_in_l2` | 218,739,496 | **+19.739 %** 1/8 | **+18.302 %** 0/8 |
| `l2_pf_hit_l2.all` | 1,123,323,418 | +18.788 % 0/8 | +18.090 % 0/8 |

**`pack1` against `pack0` — compaction against the slot-preserving layout, which
is the contrast study 65 asked its successor for:**

| counter | Δ | sign |
|---|---:|---:|
| `instructions:u` | +0.010 % | 0/8 |
| `task-clock` | −0.863 % | **4/8 — a coin flip** |
| `L1-dcache-load-misses` | −0.329 % | 6/8 |
| `l2…dc_access_in_l2` | −0.250 % | 6/8 |
| `l2…ic_dc_miss_in_l2` | **−0.811 %** | 6/8 |
| `l2_pf_hit_l2.all` | −0.135 % | 5/8 |

> ### THE DEAD TAIL WAS NOT THE MECHANISM
>
> Study 65 measured +13.23 % L2 misses and +18.90 % `l2_pf_hit_l2.all` and
> attributed both to the L2 streaming prefetcher fetching the dead 320-byte
> tail. **This study removes the tail completely and collects −0.81 % of the
> L2 misses.** `l2_pf_hit_l2.all` is still +18.09 % when there is no tail in
> existence. The prefetcher over-fetches past the end of *any* shortened
> sequential run; it does not care whether what lies beyond is dead space or the
> next live row, because either way it is bytes the fold did not want yet.
>
> Study 65's *generalisation* — "a byte you do not read is only saved if nothing
> else fetches it" — survives intact. Its proposed **fix** does not.

---

## ★★★ 3. THE DECOMPOSITION: THE INDIRECTION IS THE COST

The panel above cannot separate three effects that arrive together (the map
lookup, the decode, the byte saving). A second panel runs **four modes in the
same rounds of the same binary**, adding `ARCHON_PERM=id` — an identity map,
no packing, no permutation — which isolates the map lookup by itself.

**8 shuffled rounds, reference `ARCHON_NARROW=0`:**

| counter | ref (median) | `permid` (map only) | `perm1` (map + transpose) | `pack1` (map + 6-bit pack) |
|---|---:|---:|---:|---:|
| `instructions:u` | 27,791,774,374 | +6.016 % 0/8 | +6.050 % 0/8 | +14.102 % 0/8 |
| `task-clock` | 4152 ms | +6.487 % 0/8 | +7.102 % 1/8 | +10.945 % 0/8 |
| `L1-dcache-load-misses` | 2,793,889,052 | +3.826 % 0/8 | +3.811 % 0/8 | +3.729 % 0/8 |
| `l2…dc_access_in_l2` | 2,904,556,864 | +3.661 % 0/8 | +3.675 % 0/8 | +3.690 % 0/8 |
| `l2…ic_dc_miss_in_l2` | 215,652,762 | **+15.281 %** 0/8 | +16.151 % 0/8 | +15.651 % 0/8 |
| `l2_pf_hit_l2.all` | 1,114,009,309 | +8.825 % 0/8 | +8.823 % 0/8 | +17.602 % 0/8 |

**`pack1` vs `permid` — the 6-bit pack ALONE, decode cost minus byte saving,
with the map on both sides:**

| counter | Δ | sign |
|---|---:|---:|
| `instructions:u` | **+7.628 %** | 0/8 |
| `task-clock` | **+2.469 %** | 2/8 |
| `L1-dcache-load-misses` | −0.125 % | 4/8 |
| `l2…dc_access_in_l2` | −0.101 % | 4/8 |
| `l2…ic_dc_miss_in_l2` | **−2.210 %** | 5/8 |
| `l2_pf_hit_l2.all` | +8.629 % | 0/8 |

> ### The map, on its own, costs +6.02 % instructions, +6.49 % task-clock and +15.28 % L2 misses — for ONE `u32` load per folded row.
>
> That is 3.94 threat rows/node reading a 243 KB table, and it adds **6.6 L2
> misses per node** on a 43.1/node budget: **~1.7 L2 misses per lookup.** More
> than one, because the read misses AND displaces a weight line that was going
> to be used. The map is randomly accessed and 243 KB, while 9.3 KB/node of
> weight rows stream through a 1 MB L2 continuously — nothing that size stays
> resident.
>
> **Study 64 §6.1 priced this route with exactly that 243 KB table. Study 65
> designed the whole slot-preserving layout to avoid paying it, blamed the loss
> on the prefetcher, and named "pay the table instead" as the survivor. The
> table is 2.6× more expensive in L2 misses than the byte cut is worth.**

And the byte cut, isolated, is real but small: **−2.21 % L2 misses** (5/8, so
weak) for **+7.63 % instructions and +2.47 % task-clock**. **Even with a free
map the route still loses on wall clock**, so the obvious follow-up — shrink
the map to a 15 KB block prefix-popcount index — cannot rescue it and was not
built.

### 3.1 Why the decode cannot get cheaper

1280 lanes × 6 bits is 960 B, and turning 24 packed bytes into 32 `i16` lanes
needs a bit-level shuffle. Study 65's kernel is already minimal —
`vpermb` → `vpmultishiftqb` → `vpternlogd` → `vpmovzxbw`, **four vector ops per
32 lanes against the wide path's one `vpmovsxbw`**. None is removable:
`vpmultishiftqb` needs each group's 6 bytes in its own qword (hence `vpermb`),
its 8-bit windows carry 2 bits of the next field (hence the mask), and the
result is bytes (hence the widen). Byte-aligning the fields to drop the permute
puts the row back at 1280 B.

> **A sub-byte feature-transformer format costs 4× the fold's vector-op count to
> save 25 % of its bytes.** On a machine where the weight stream's latency is
> already overlapped — 33.4 % of threat row touches are inside an L1-sized
> window and 97.4 % inside an L3-sized one (§1) — that trade cannot pay, in any
> layout. **The ≤ 6-bit narrow-row family is closed.**

---

## ★★★ 4. LOCALITY: THE TRANSPOSE IS BIT-EXACT, FREE, AND WORTH ZERO

### 4.1 What the index looks like, and why the span is 64 MB

The FullThreats index is
`lut1[attacker][victim][from<to] + from_to[attacker][from][to]`, i.e.

```
index = cum_off[att] + sub · geom[att] + ft(from, to)
```

with `sub` the victim relation (colour × type ordinal) and `ft` the `(from, to)`
geometry ordinal. **The victim relation is the SLOWEST axis of each attacker
block and the geometry the fastest.** So two rows of one fold that share
`(attacker, from)` but differ in victim sit `geom[att]` rows apart — **336 rows
(430 KB) for a knight, 1456 (1.8 MB) for a queen** — and rows from different
attackers sit whole blocks apart. That is §1's 2²⁶ modal span.

**Transposing each block** (`cum_off + ft · num_subs + sub`) is a pure
permutation of the row space, so it is **bit-exact by construction** — the
accumulator sums the same vectors — and it costs **nothing at run time** if
folded into `build_threat_tables`' two LUTs, which already carry exactly these
two terms. It moves the within-attacker victim stride from `geom` rows to
`num_subs` rows, i.e. **≈ 34× closer for a knight and ≈ 146× for a queen.**

### 4.2 ⚠ IT MOVED LOCALITY THE OTHER WAY, AND THE CENSUS IS WHY I KNOW

I predicted the transpose would shrink a fold's address span. `FOLDREDUND` now
charges the **physical** offset (the row map's, not `feature · ACC`), so the
prediction is checkable, and it is wrong in the half that matters:

| threat channel, whole frame | identity | transposed |
|---|---:|---:|
| row PAIRS within one 4 KB page | **2.794 %** | **0.842 %** |
| folds spanning ≥ 2²⁴ bytes | 60.31 % | 56.28 % |
| folds already in ascending address order | 30.8 % | 30.8 % |

The span barely moves (60.3 % → 56.3 %) because a fold's rows mostly belong to
**different attackers**, and the transpose is within-block. What it does do is
**destroy the one piece of locality the identity layout had**: under
`cum_off + sub·geom + ft`, rows sharing `(attacker, victim, from)` and differing
only in the target square have **consecutive** `ft`, i.e. they are adjacent
1280-byte rows on the same page. The transpose separates them by `num_subs`
rows ≈ 12.8 KB. **Same-page row pairs fall 3.32×.**

So this is not the arm I set out to build. It is a better one: a **large,
measured, opposite-direction perturbation of exactly the quantity the brief's
routes 1 and 3 propose to improve.**

### 4.3 Measured, as a fifth data mode of the same binary

`ARCHON_PERM=1` applies the transpose through the row map, so its null
(`ARCHON_PERM=id`) pays the identical lookup and the contrast is the
**permutation alone**:

| counter | Δ (`perm1` vs `permid`) | sign |
|---|---:|---:|
| `instructions:u` | +0.033 % | 0/8 |
| `task-clock` | +0.407 % | **4/8** |
| `L1-dcache-load-misses` | +0.297 % | 3/8 |
| `l2…dc_access_in_l2` | +0.229 % | 3/8 |
| `l2…ic_dc_miss_in_l2` | +1.505 % | 3/8 |
| `l2_pf_hit_l2.all` | +0.481 % | 2/8 |

Node total 5,001,560 and eval hash `2e1ede95…79c4` exact, which is the check
that the permutation is a permutation.

> ### Cutting same-page row pairs by 3.32× converts nothing.
>
> Every counter sits inside ±1.5 % with a 2–4/8 sign count, and the direction is
> the *harmful* one, so this is a **two-sided** null on locality: if scattering a
> fold's rows 3.3× further apart costs nothing measurable, clustering them
> cannot be worth much either. That closes the brief's routes 1 and 3 —
> *"visit weight rows in ADDRESS order within a fold"* and *"restructure the
> Finny table so the 114 HalfKA lines per refresh are contiguous"* — with a
> measurement rather than an argument, and it does so at a scale no in-fold
> reordering of 2–5 rows could ever reach.
>
> The mechanism is the same one study 43 §6.5 found for prefetch: **the fold is
> miss-PARALLEL, not miss-serial.** One tile walk exposes ~80 independent line
> accesses inside a ~320-µop window, so the misses are all in flight already and
> limited by miss-handling registers, not by how far apart their addresses are.
> Moving the addresses does not change how many there are.
>
> **What it does NOT close:** the 4 KB page is the granularity this census
> measures, and TLB pressure is separately known to be small (study 60: 2.25
> page walks/node, huge pages a null). A permutation aimed at *cache-set*
> conflicts rather than pages would be a different arm, and nothing here
> measures it.

### 4.4 The corollary for prefetch, from the kernel's shape

Worth writing down because the brief lists *"software-prefetch the next row
while folding the current one"* as a route and it reads as new. It is not
buildable as stated: the tile walk visits **every row of the fold inside tile
0**, so there is no "next row" whose address is known long before its demand
load — the demand load for row `r+1` issues ~40 instructions after row `r`'s.
That is prefetch kill #8, by construction rather than by measurement, and it is
consistent with the seven that were measured.

---

## ★★ 5. THE NET-CORRUPTION TRAP, ONE LEVEL DEEPER THAN STUDY 65 FOUND IT

Study 65 caught that a packed table must be **unpacked** by `serialize`, or
`serialize ∘ load` stops being the identity and every net-surgery tool that
round-trips a net writes a corrupt file. Compaction breaks it again in a second
way: a row is no longer at `feature · ACC` **at all**, so the serialiser's walk
over `chunks_exact(ACC)` visits the wrong bytes. Walking the ROW MAP instead is
what makes one loop correct for all three layouts.

It also broke `SotaNetwork`'s equality, and the failure was instructive: four
round-trip tests failed because the compacted table is **64 bytes longer** (the
overread pad) than the flat one, even for a synthetic net with **zero** narrow
rows. A byte comparison of a table that has three legal physical encodings
answers a question nobody asked. `threat_rows_equal` compares row by row through
each net's own map, with a `memcmp` fast path so the check stays `O(bytes)` in a
debug build.

**The general form, for the next layout change:** *if a structure gains a
physical encoding, its equality has to move up one level of abstraction in the
same commit, or the gate starts reporting encoding differences as content
differences.*

---

## ★★ 6. THE OFF-PATH IS ESSENTIALLY UNMOVED

`narrow65` is not in any default feature set, so the shipped build takes
`pack_threat_plane`'s `#[cfg(not(feature = "narrow65"))]` arm, keeps the flat
table and an empty map, and `row_entry` falls back to `feature · ACC`. Measured,
`sota,arch-d48x`, three reps each:

| | `.text` md5 | `.text` size | nodes | eval hash | `instructions:u` |
|---|---|---:|---:|---|---:|
| `base` (`41f3b41b`) | `56c9fc27` | 758,213 | 5,001,560 | `2e1ede95…4dd8` | 27,652,282,848 |
| `ship66` (this tree) | `f753c74a` | **758,261** | 5,001,560 | `2e1ede95…4dd8` | 27,653,097,396 |

**+48 bytes of `.text` and +0.00295 % instructions = +0.163 i/node**, from
`row_entry` replacing `is_narrow` in the cold from-scratch rebuild and the
serialiser. Node-identical and eval-hash exact. Reported rather than claimed
byte-identical: study 65's §2.4 set that precedent and it is the honest gate.

---

## ★★ 7. WHAT I COULD NOT ESTABLISH

Most doubtful first.

1. **`task-clock` is a wall-clock statistic on a shared box and I quoted it.**
   Studies 63/64/65 refused wall clock entirely; the brief for this study asked
   for it explicitly. The mitigation is that every figure is a **within-round
   paired median of arm/null ratios in one binary** with its sign count printed,
   and the two verdicts that rest on it (the transpose at 4/8, compaction at
   4/8) are called coin flips *because of* the sign count, not despite it. The
   large ones (+6.49 %, +10.95 %, all 0/8) are far outside anything the sign
   counts suggest is noise. It is still not an idle-box number and no Elo is
   derived from it.
2. **The ~1.7-L2-misses-per-map-lookup figure is a division, not a
   measurement.** It divides the panel's +15.28 % (6.6 misses/node) by the
   census's 3.94 threat rows/node. A per-site counter would settle whether the
   extra 0.7 is displacement or a second line touched.
3. **The block-index map (15 KB instead of 243 KB) was not built.** §3 argues it
   cannot rescue the route because the pack loses on task-clock *with the map on
   both sides*; that is an argument from a measured contrast, not a
   measurement of the smaller map.
4. **The transpose was measured through the map, not baked into the LUTs.** Its
   own contrast is clean (both sides pay the same lookup), but a landed version
   would have a different instruction profile — a better one. Since the contrast
   is a null at 4/8, landing it was not worth the LUT surgery.
5. **I predicted the transpose's direction wrong** (§4.2) and only found out
   because `note_fold_redundancy` was changed to charge the PHYSICAL offset. The
   null it produced is a two-sided one and therefore stronger than the one I set
   out to measure, but that is luck, not design: had the census still reported
   virtual offsets, this document would have claimed a ~500× span collapse that
   never happened. **Charge a layout census to the address the hardware sees.**
6. **The 1.93 % threat-row cancellation is priced, not built.** The price
   (~5 comparisons per fold against 1.05 % of folded rows) says it is a wash;
   nobody has measured it.
7. **No games, and no Elo.** Everything here is bit-exact, so games could only
   arbitrate speed, and nothing landed that changes speed.
8. **`clippy --all-targets` under `sota,arch-d48x,narrow65` reports 9 warnings
   that pre-date this study** (in `simple_halfka.rs`, `protected_halfka.rs`,
   `aligned.rs`, plus study 65's two dead-code notes). `tools/check.sh` runs
   clippy with `-D warnings` on the DEFAULT feature set only, which is clean.
   Flagged, not fixed — they are not this study's and touching them would
   confound the off-path diff.

---

## ★ 8. WHAT THE NEXT STUDY SHOULD KNOW

1. **The ≤ 6-bit narrow-row family is CLOSED, in all three layouts, with the
   cost decomposed.** Slot-preserving loses; compacted loses by the same amount;
   the indirection is +6 % wall clock on its own and the decode is +7.6 %
   instructions for a −2.2 % L2-miss saving. Do not re-propose it, and do not
   re-propose the dead-tail explanation — §2.3 refutes it directly.
2. **Locality inside a fold converts nothing, measured at 3.32× in the harmful
   direction.** The fold is miss-PARALLEL: one tile walk exposes ~80 independent
   line accesses inside the out-of-order window, so the misses are all in flight
   already and their addresses' proximity is irrelevant. This is the same
   mechanism that killed seven prefetch arms, now measured from the other
   direction. **Any future "reorder / cluster / make contiguous" proposal has to
   explain why it is not this** — and note that §4.2 also found the identity
   layout's *existing* locality (rows sharing attacker + victim + from-square
   are consecutive), which is easy to destroy by accident and worth nothing to
   preserve.
3. **A randomly-accessed table of ~250 KB costs ~1.7 L2 misses per lookup in
   this engine.** That is a reusable price, and it should be applied before
   adding any indirection to a hot path. It also puts a question mark over
   study 59's `from_to` fusion (64 KiB of `u8` → 128 KiB of `u16` to save one
   add), whose own doc argues the size is "paid in address space, not in touched
   lines" — that reasoning is what this study measured to be wrong. **But the
   whole lookup is only ~0.9 % of the engine (study 59's own IBS figure), so
   halving the table can win at most a fraction of that; it is priced out, not
   recommended.**
4. **The row map machinery is landed and reusable.** Any future layout that
   needs "rows are not at `feature · ACC`" now has the map, the serialiser walk,
   the logical equality, and a five-mode in-binary null harness. What it does
   NOT have is a cheap lookup — see item 3.
5. **What is left on weight-row bytes, honestly:** row count is within 6 % of
   the deferral minimum (study 64 §3), width is an owner ruling, precision is
   spent, sub-byte packing is closed here, and layout is closed here. **The only
   multiplier left is study 64 §5's — evaluations per node, priced at ≈ 45 % of
   the engine's L1-miss latency at the free-predictor ceiling.** It is
   FUNCTIONAL and needs games, and study 65 §3 has the ROC that says the free
   predictors are not good enough and the live route is a trained HalfKA-only
   tier (break-even at 28.2 % coverage, `smallnet_*` machinery already shipping
   inert). That is a net decision, not a kernel one, and it is where the next
   order of magnitude is.
6. **Still open, inherited and untouched:** study 43 §9 item 2's subtractive
   Finny-line arm; study 57 §6's FNV-1a-64 payload checksum; study 48 §9 item
   3's AVX2 `activate` hoist; study 63 §7 item 1's re-run of the +3.14 Elo
   receipt on the repaired wall-clock tool.

---

## ★ 9. REPRODUCE

```bash
ln -s /path/to/chessengine/data data        # the worktree needs the net

bash tools/perf66/build66.sh base                            # shipped features
bash tools/perf66/build66.sh n66   narrow65                  # the five data modes
bash tools/perf66/build66.sh fc66  foldcount,rowprobe,narrow65

# §1 — the intra-fold census
ARCHON_FOLDCOUNT=/var/tmp/perf66/fc66.txt taskset -c 2 ./bin66/fc66 \
  bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
grep -E 'FOLDREDUND|FOLDROWREUSE|FOLDREGION thr_weights' /var/tmp/perf66/fc66.txt
python3 tools/perf66/derive66.py

# §2.2 — bit-exactness in every mode
bash tools/perf66/hash66.sh ./bin66/n66
P66_MODES="ARCHON_PERM=id ARCHON_PERM=1" bash tools/perf66/hash66.sh ./bin66/n66

# §2.3 — the layout contrast
bash tools/perf66/panel66.sh ./bin66/n66 8

# §3, §4.2 — the four-mode decomposition
P66_OUT=/var/tmp/perf66/all4.tsv bash tools/perf66/panel66.sh ./bin66/n66 8 \
  narrow0=ARCHON_NARROW=0 permid=ARCHON_PERM=id perm1=ARCHON_PERM=1 pack1=ARCHON_PACK=1
python3 tools/perf66/pairs66.py /var/tmp/perf66/all4.tsv perm1 permid
python3 tools/perf66/pairs66.py /var/tmp/perf66/all4.tsv pack1 permid

# §6 — the off-path, and the merge gate
bash tools/perf66/ident66.sh ./bin66/base ./bin66/ship66 3
bash tools/perf65/test65.sh sota,arch-d48x,narrow65
CHECK_NNUE_GATE=1 bash tools/check.sh
```

**Do not run a panel while anything else builds**, and do not park perf text on
`/tmp` — it is tmpfs.
