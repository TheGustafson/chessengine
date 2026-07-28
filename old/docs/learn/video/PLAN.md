# ARCHON Learning Videos — Production Plan

A repeatable pipeline for turning the 15-part first-principles course in
`docs/learn/01..15` into narrated, animated explainer videos (Manim + Piper TTS),
buildable by an agent that **cannot watch video or hear audio**. Everything lives
under `docs/learn/video/`; nothing else in the repo is modified.

The pilot (video 02, Board Representation) is fully built and is the reference
implementation — read its `scenes/pilot02.py`, `scripts/02-narration.json`, and
`pipeline/*` alongside this plan.

---

## 1. Why video 02 is the pilot (01 vs 02)

Both candidate docs are strong openers. **02 — Board Representation** was chosen
because it maximizes the one hard constraint of this brief: *the visuals must
carry the explanation, and an agent that can't watch must still verify them.*

| Criterion | 01 — Problem of chess | 02 — Board representation |
|---|---|---|
| Core visual object | game trees, Elo curves, block diagrams (abstract) | the **8×8 board** — Manim's ideal object, instantly legible |
| "Aha" per concept | minimax propagation, αβ cuts (good, but one big idea) | *every* concept is a concrete grid animation: LERF numbering, bit-split, set-ops lighting squares, knight mask, `pop_lsb` iteration, `<<8` shift |
| Risk of "talking slides" | higher (tables, formulae, prose) | low — squares light up, values compute on screen |
| No-eyes verifiability | harder (curves/trees need semantic reading) | easy — bounding boxes of a known grid; frames read trivially |
| Series role | natural episode 1 | self-motivating ("how do you store a board *fast*?") |

02 also stress-tests the pipeline hardest (boards, code panels, syntax coloring,
piece glyphs, staged reveals, an iteration animation), so proving it de-risks the
other 14. Video 01 is the natural *next* build and reuses everything here (its
game-tree/minimax visuals map onto the same `NarratedScene` + timing engine).

---

## 2. Toolchain (verified working on this machine)

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12.3 | venv at `docs/learn/video/venv` (uv-created) |
| Manim CE | 0.20.1 | Cairo renderer; LaTeX present (`dvisvgm`) but **not needed** — we use Pango `Text`, not `Tex` |
| manimpango | built from source | needed system `libpango1.0-dev libcairo2-dev pkg-config` (no wheel for cp312 here) |
| Piper TTS | `piper-tts` (onnxruntime 1.27, CPU) | voices in `voices/`: `en_US-ryan-high` (default), `en_US-lessac-medium` (fallback) |
| ffmpeg / ffprobe | 6.1.1 | concat + AAC mux + frame extraction |
| pydub | 0.25.1 | assembles the narration track at exact offsets |

**One-time environment setup** (already done; reproduce with):
```bash
apt-get install -y libpango1.0-dev libcairo2-dev pkg-config libgirepository1.0-dev espeak-ng
cd docs/learn/video
uv venv venv --python 3.12
VIRTUAL_ENV=$PWD/venv uv pip install --python venv/bin/python manim==0.20.1 piper-tts
venv/bin/python -m piper.download_voices en_US-ryan-high  --download-dir voices
venv/bin/python -m piper.download_voices en_US-lessac-medium --download-dir voices
```

**Box / tournament rules (mandatory while a match runs — check `pgrep -f fastchess`):**
- run every CPU-heavy step under `nice -n 19`;
- cap ffmpeg/render threads at 2 (`-threads 2`; `OMP_NUM_THREADS=2` for Piper/onnxruntime);
- test renders stay short + low-res (**480p15**); the single high-quality **1080p30**
  render is deferred to the very end;
- never touch the repo outside `docs/learn/video/`; never push;
- media/audio/venv bytes are gitignored — commit **only** scripts, plan, and the
  scripts-of-record.

---

## 3. Workspace layout

```
docs/learn/video/
  PLAN.md                     ← this file
  .gitignore                  ← ignores media/, venv/, *.mp4, *.wav, voices/*.onnx, checkpoints/*.png
  venv/                       ← python env (gitignored)
  voices/                     ← Piper .onnx models (gitignored)
  scripts/
    NN-narration.json         ← SOURCE OF TRUTH: title, voice, pacing, per-scene beats (committed)
    NN-narration.md           ← generated human-readable script (committed)
  manifests/
    NN-timing.json            ← derived timing contract (committed): per-beat audio_dur, beat_dur, offsets
  scenes/
    common.py                 ← visual system + timing engine + no-eyes guardrails (committed)
    pilotNN.py / videoNN.py   ← the Scene classes (committed)
  pipeline/
    synth.py                  ← narration.json -> per-beat wavs + timing.json
    build_audio.py            ← timing.json + wavs -> one full_track.wav (pydub, exact offsets)
    assemble.py               ← concat scene clips + mux narration -> final mp4
    verify.py                 ← no-eyes verification (timing, drift, ffprobe)
    gen_script_md.py          ← narration.json -> narration.md
    render_scenes.sh          ← render all scenes at a quality tier
  audio/NN/*.wav              ← per-beat clips + full_track.wav (gitignored)
  media/                      ← manim renders + final mp4 (gitignored)
  checkpoints/*.png           ← frame-read evidence (gitignored)
```

---

## 4. The pipeline (end to end)

```
scripts/NN-narration.json          (author: prose, segmented into scenes→beats)
        │  synth.py  (Piper, nice, OMP=2)
        ▼
audio/NN/<beat>.wav  +  manifests/NN-timing.json
        │                         │
        │ gen_script_md.py        │  scenes read beat budgets from here
        ▼                         ▼
scripts/NN-narration.md    scenes/videoNN.py  ──render──▶  media/videos/videoNN/<Q>/<Scene>.mp4
        (committed)                                              │
build_audio.py ──▶ audio/NN/full_track.wav                       │
        └──────────────────────────┬─────────────────────────────┘
                          assemble.py  (ffmpeg concat + AAC mux, -threads 2)
                                     ▼
                         media/pilot/archon-NN-*.mp4
                                     │
                          verify.py + frame reads (ffprobe, Read PNGs)
```

### 4.1 Audio-driven timing (why A/V stay locked)

The narration is synthesized **first**; the video is then built to fit it. This is
the core trick that makes sync exact without watching:

1. `synth.py` renders each beat and measures its wav length `audio_dur`.
2. The beat's on-screen **budget** is `beat_dur = ceil((audio_dur + pad) * fps) / fps`
   — quantized up to the frame grid so a 30 fps render is frame-exact.
3. Each Scene loads `manifests/NN-timing.json`. The `NarratedScene.beat(id)` context
   manager plays animations (whose `run_time`s I control and sum) and then **waits
   out the remainder** to hit `beat_dur` exactly. If animations exceed the budget it
   **raises** (a real overrun guard).
4. A scene's total duration is therefore the exact sum of its beat budgets (+ a
   small inter-scene tail). `build_audio.py` lays each clip on a silence bed at its
   `global_offset` (= cumulative sum). Video and audio are both built to the same
   numbers ⇒ **drift ≈ 0**, verified to the millisecond by `verify.py`.

Because `audio_dur ≤ beat_dur` always, narration never bleeds into the next beat;
the `pad` becomes a short breath between beats.

### 4.2 Commands (pilot 02)

```bash
cd docs/learn/video
V=venv/bin/python
nice -n 19 env OMP_NUM_THREADS=2 $V pipeline/synth.py 02          # wavs + timing.json
nice -n 19 $V pipeline/gen_script_md.py 02                        # narration.md
ARCHON_LAYOUT_ONLY=1 nice -n 19 venv/bin/manim render -r 320,180 --fps 2 scenes/pilot02.py <Scene>   # fast geometry gate
nice -n 19 bash pipeline/render_scenes.sh ql 15                   # 480p15 test renders
nice -n 19 $V pipeline/build_audio.py 02                          # full_track.wav
nice -n 19 $V pipeline/assemble.py 02 480p15 media/pilot/archon-02-480p.mp4
nice -n 19 $V pipeline/verify.py 02 480p15 media/pilot/archon-02-480p.mp4
# ... iterate on frames ...  then FINAL:
nice -n 19 bash pipeline/render_scenes.sh qh 30                   # 1080p30
nice -n 19 $V pipeline/assemble.py 02 1080p30 media/pilot/archon-02-board-representation.mp4
nice -n 19 $V pipeline/verify.py 02 1080p30 media/pilot/archon-02-board-representation.mp4
```

---

## 5. NO-EYES VERIFICATION PROTOCOL

An agent cannot watch the video, so correctness is established by four independent,
programmatic layers. **Run them in order; do not full-render before the geometry
gate is green.**

**(a) Fast geometry gate — `ARCHON_LAYOUT_ONLY=1`.**
`common.py` stubs all animations/waits when this env var is set, so `construct()`
runs in ~1 s and fires every guardrail without rendering frames:
- `assert_in_frame(m)` — nothing crosses the safe margin (x ±6.93, y ±3.82 of the
  8×14.22 frame);
- `assert_min_height(t)` — no text below ~0.26 u (readable at 1080p);
- `assert_no_overlap(a,b)` — declared region pairs (board vs panel) don't collide;
- the `beat()` overrun guard — animation `run_time`s never exceed the narration budget.
Render each scene at `-r 320,180 --fps 2`; a clean pass means the geometry is valid.
*This gate caught, in the pilot: question text 0.5 u past the right edge, a code
panel wider than the frame, a duplicate-color `t2c` crash, and several right-panel
overflows — all before any real render.*

**(b) Frame reads — the eyes substitute.** Render the scene at **480p15**, extract
PNGs at fully-built beat moments with
`ffmpeg -ss <t> -i scene.mp4 -frames:v 1 checkpoints/<scene>_<tag>.png`, and **Read
each PNG**. This is where a human-style look catches what geometry can't: stale
mobjects from a previous beat that were never faded, low-contrast glyphs, colour
clashes, awkward spacing. *This step caught, in the pilot: S6 accumulating stale
code panels across beats (overlapping text), the S4 pawns beat keeping the wrong
"A / B" labels, a faint single knight glyph, and lost struct indentation.* Leave
the checkpoint PNGs on disk as evidence.

**(c) Timing + drift — `verify.py`.** Probes each rendered clip's duration against
its manifest scene duration (tolerance ≤ 2.5 frames), asserts every beat's
`audio_dur ≤ beat_dur`, and checks the assembled track length equals the total
video length (drift target < 50 ms).

**(d) Final probe — `verify.py ... final.mp4`.** ffprobe confirms resolution
(1920×1080), fps (30), an audio stream is present, and audio≈video length. Then
Read ≥5 spread frames of the final muxed file as a last human check.

**Iteration loop:** storyboard → geometry gate → 480p15 render → Read frames → fix
→ repeat per scene → only then assemble + verify → only then the final 1080p30.

---

## 6. Visual design system (keep consistent across all 15)

- **Palette** (`common.py`): bg `#0d1117`; board squares muted slate
  (`#8b98ad`/`#3d4757`); **cyan `#3ec6ff` = set A / primary**, **amber `#ffc857` =
  set B / values**, **green `#4ade80` = result / success**, red `#ff6b6b` =
  caveat; text `#e6edf3`, muted `#9aa5b1`. Colour *means* something — don't
  decorate with it.
- **Fonts:** `DejaVu Sans` for prose/titles, `DejaVu Sans Mono` for code/values.
  Body ≥ font_size 24; titles 40–48; code 21–30.
- **Layout convention:** board (or code) on the **left** (≈ x −5.6…−0.6), a text
  panel on the **right** (centre ≈ x 3.35, max width 6.7). Use
  `NarratedScene.place_right(mob, top=…)` / `fit_width(…)` so panels always fit.
- **Header:** every scene shows `NN · <Doc title> — <section>` top-centre.
- **Motion discipline:** build a diagram in the first few seconds of a beat, then
  hold it (with a small `Indicate`/`Circumscribe` accent) while narration explains.
  Long dead holds are fine — the voice is talking.
- **Pitfalls baked into the tooling:** Pango collapses leading spaces (use nbsp for
  code indent); `t2c` keys must not be substrings of each other (ambiguous colour);
  multi-line `Text` glyph indices skip newlines (box per-line `Text` mobjects, not
  slices of one block); **fade a beat's whole panel before the next beat** (store it
  as `self._rcol`).

---

## 7. Per-video effort template (videos 01, 03–15)

For each doc `NN`:

1. **Read the doc** and pull a spine of 6–10 scenes, each 1–5 narrated beats. Keep
   claims faithful to the doc (cite file/const names as the doc does).
2. **Write `scripts/NN-narration.json`** — `voice`, `length_scale` (~1.06),
   `pad_seconds` (~0.55), `fps` 30, and scenes→beats of tight teaching prose.
   Target 1000–1500 words ⇒ ~7–9 min.
3. `synth.py NN` → check the printed total lands in 6–12 min; adjust prose/pad.
4. **Build `scenes/videoNN.py`** on `NarratedScene`, reusing `ChessBoard`,
   `code_block`, `piece`, `place_right`, the palette, and the beat pattern.
   Prefer the pilot's idioms; add new reusable mobjects to `common.py` if shared.
5. **Geometry gate** (5a) until green.
6. **480p15 render + frame reads** (5b), scene by scene, until composition is clean.
7. `build_audio.py` → `assemble.py … 480p15` → `verify.py` (5c) → Read spot frames.
8. **Final:** `render_scenes.sh qh 30` → `assemble.py … 1080p30` → `verify.py`
   final probe + Read ≥5 frames. Defer this step if a tournament is running; if it
   must run, keep `nice -n 19` and threads 2.
9. `gen_script_md.py NN` and commit scripts/plan/scenes/manifest (never media bytes).

New-doc-specific visuals to design (not needed for 02): **01** game-tree growth +
minimax/αβ propagation; **03** magic-bitboard ray masks; **05** αβ pruning cuts;
**07** TT probe/replace; **11–12** NNUE accumulator as a flowing vector. All sit on
the same timing engine.

---

## 8. Cost estimate to scale to the remaining 14

Wall-clock is dominated by **agent iteration** (writing + reading frames), not CPU.
Compute is modest and fully `nice`-able around a tournament.

**Per-video CPU/render budget (this machine, nice -n 19, threads 2):**

| Step | Cost |
|---|---|
| Piper synth (≈1.3k words) | ~30–60 s |
| Geometry gate (9 scenes @ 320×180 fps2) | ~1–2 min |
| 480p15 test renders (all scenes, ×N iterations) | ~4–7 min per full pass |
| Audio assemble + mux | < 20 s |
| **Final 1080p30 render (all scenes)** | **~20–35 min** (the long pole) |
| Assemble + verify final | ~1–2 min |
| **CPU subtotal** | **~30–50 min/video** (mostly the deferred final render) |

**Per-video agent-effort budget** (the real cost):

| Phase | Effort |
|---|---|
| Read doc + write narration JSON | moderate |
| Author scene code (reusing pilot) | the bulk — 1st-timer heavy, drops fast |
| Geometry-gate fix loop | small (fast, mechanical) |
| Frame-read + polish loop (Read ~15–25 PNGs) | moderate — the quality lever |
| Assemble + verify + final | small |

Estimate: **the pilot cost ~1 focused agent-session** (much of it building the
reusable `common.py` + pipeline, now amortized). With that infrastructure done,
budget **~0.4–0.7 session/video** for 01 and 03–15, i.e. the series is roughly a
**6–10 session** effort total, plus ~30–50 min nice'd CPU each (≈ 7–12 CPU-hours
for all 14 finals, trivially schedulable overnight / between tournament rounds).
Videos heavy on new visual metaphors (01 trees, 03 magic bitboards, 11–12 NNUE)
sit at the top of the range; representation-style docs (04, 09, 13, 14) at the
bottom.

**Storage:** ~15–35 MB per 1080p30 final; per-beat wavs + track ~40–80 MB
(gitignored). Keep only committed text + the final mp4s you choose to publish.

---

## 9. Known limitations / polish backlog

- Piper drops one syllabic-stress phoneme on rare words (harmless, inaudible).
- Chess piece glyphs are Unicode (DejaVu Sans); single dark-square pieces get a
  disc backing (see S5). For richer boards, SVG piece sets could replace glyphs.
- `assert_no_overlap` only checks pairs you declare; cross-beat stale-mobject
  overlaps are caught by frame reads, not the gate — always do step 5b.
- No captions/subtitles track yet; `srt` is a dependency of manim and the manifest
  already has per-beat text + timestamps, so a `.srt` export is a small add.
