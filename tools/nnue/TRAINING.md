# NNUE Tier-A training pipeline — reproducible commands + smoke validation

Status: validated 2026-07-11 (this note captures the **training half** of the
Tier-A bootstrap: `raw text shards -> 32-byte bulletformat bin -> GPU-trained,
quantised net`). The **datagen half** is validated separately in
`design/15-nnue-datagen.md §10`.

This is the "prove the machinery on a tiny corpus before spending datagen
core-hours" gate (`design/15 §4 step 3-4`, `design/16 §5`). Everything here is
**discardable scaffold** (D15/D28): `bullet`/`bulletformat` are external, cloned
read-only, never committed; the `.bin`/net/corpus artifacts live under
gitignored `data/`. Only the *tooling/docs* in `tools/nnue/` are tracked. **No
engine (`src/`) code is involved.**

Bottom line: **GO on the training half.** `gen -> convert -> train` runs on this
box's GPU end to end, emits a correctly-formatted quantised net, and is
compute-cheap (minutes-scale) — the cost of the real net remains **datagen**,
not training.

---

## 0. Environment this was validated on

| component | value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, 8 GB, reported by bullet as `sm_89` (Ada) |
| CUDA toolkit | 12.0 (`nvcc` V12.0.140); libs in `/lib/x86_64-linux-gnu` + WSL driver in `/usr/lib/wsl/lib` |
| driver | 595.97 (WSL2 passthrough; `libcuda.so.1` under `/usr/lib/wsl/lib`) |
| rustc | 1.92.0 (bullet needs edition 2024 => rustc >= 1.87) |
| OS | Linux WSL2 |

## 1. Pinned external tooling (open question §9.1 of doc 15 — now pinned)

Cloned read-only to the session scratchpad (NOT committed; GPLv?/MIT external):

| repo | commit | version |
|---|---|---|
| `bullet` (github.com/jw1912/bullet) | `cebc78a093d9` | `bullet_lib` 1.0.0, workspace edition 2024 |
| `bulletformat` (github.com/jw1912/bulletformat) | `cb6a3602d70b` | 1.8.0 (all `ChessBoard` fields public) |

`bullet_lib` pulls `bulletformat 1.8.0`, `montyformat 0.9.2`, `sfbinpack 0.6.4`,
`viriformat 2.0.1` from crates.io — so **the trainer build needs crates.io the
first time** (deps then cache). `bulletformat` itself is **zero-dependency** and
builds fully offline.

## 2. The text -> 32B bin converter

`bulletformat` is a *library*: its `convert_from_text::<ChessBoard>` is the
sanctioned text->bin path (`design/15 §3.2.2`; we do NOT hand-pack the 32-byte
record). We wrap it in a ~170-line bin. The reference source is committed at
`tools/nnue/bf_convert.rs`; it also has an `inspect` subcommand that decodes
records with bulletformat's own `DataLoader` and cross-checks each against the
source text (piece multiset + score-sign + result + king squares — the
`design/15 §3.2.4` canonicalisation).

Build it (drop the source into a throwaway crate with `bulletformat` as a path
dep):

```bash
SCRATCH=<your scratch>            # holds the bullet/ and bulletformat/ clones
mkdir -p "$SCRATCH/nnue_convert/src"
cp tools/nnue/bf_convert.rs "$SCRATCH/nnue_convert/src/main.rs"
cat > "$SCRATCH/nnue_convert/Cargo.toml" <<EOF
[package]
name = "nnue_convert"
version = "0.1.0"
edition = "2021"
[[bin]]
name = "nnue_convert"
path = "src/main.rs"
[dependencies]
bulletformat = { path = "../bulletformat" }
EOF
( cd "$SCRATCH/nnue_convert" && cargo build --release --offline )
CONV="$SCRATCH/nnue_convert/target/release/nnue_convert"
```

Invocations:

```bash
$CONV convert  <in.txt> <out.bin>      # bulletformat text->bin (ChessBoard)
$CONV inspect  <bin> <txt> [K]         # decode + cross-check all records; print K
```

### Wire into build_corpus.sh (the pipeline path)

`build_corpus.sh` already parametrises the converter via `BULLETFORMAT_CONVERT`
(and optional `BULLET_UTILS_SHUFFLE`/`_ILEAVE`). Run:

```bash
BULLETFORMAT_CONVERT="$CONV convert" \
  bash tools/nnue/build_corpus.sh --seed <seed> --require-convert
```

It converts every `data/nnue/raw/shard_*.txt` -> `data/nnue/bin/*.bin`, asserts
`bin_bytes % 32 == 0` and `records == text_lines` per shard, concatenates to
`data/nnue/train.bin` + carves `data/nnue/val.bin` (~1 %), and writes
`meta.json`. (For a real corpus, also set `BULLET_UTILS_SHUFFLE`/`_ILEAVE` to the
`bullet-utils` shuffle/interleave commands — `cargo b -r -p bullet-utils`, no
CUDA needed — instead of the plain `cp`/`cat` fallback.)

### Smoke result (shard_43, 5201 positions)

- 6 shards -> `train.bin` **5149** + `val.bin` **52** = **5201** records;
  every file `% 32 == 0`; `sum(bin records) == raw text lines == 5201`.
- `inspect` **fully decode-verified all 5201 records**: exactly one king per
  side, occ-popcount == nibble count, `|score| < 2500`, `result ∈ {0,1,2}`,
  king-square fields consistent, and — the `§3.2.4` sign trap — the on-disk
  STM-relative `score`/`result` match the White-relative text under
  bulletformat's canonicalisation (black-to-move => score negated, result
  flipped, board colour-swapped + vertically mirrored). **0 mismatches.**

## 3. Build the trainer (bullet, CUDA)

The `cuda` feature makes `build.rs` link `cuda/cudart/nvrtc/cublas`; kernels are
JIT-compiled at runtime via NVRTC (no `nvcc` at build time). `CUDA_PATH` must
point at a dir whose `lib64`/`include` exist — on this box `/usr` works (nvcc
root; the actual libs resolve via the default linker dirs + the WSL dir that
build.rs adds automatically).

The smoke net arch is a `bullet` example, kept as the committed reference
`tools/nnue/train_smoke.rs`; copy it into the bullet checkout and register it:

```bash
cp tools/nnue/train_smoke.rs "$SCRATCH/bullet/examples/nnue_smoke.rs"
# add to $SCRATCH/bullet/crates/bullet_lib/Cargo.toml:
#   [[example]]
#   name = "nnue_smoke"
#   path = "../../examples/nnue_smoke.rs"

cd "$SCRATCH/bullet"
CUDA_PATH=/usr cargo build --release -p bullet_lib --features cuda --example nnue_smoke
```

Result: builds in ~11 s (deps cached), 0 errors; the binary links
`libcuda.so.1 / libnvrtc.so.12 / libcublas.so.12` (verified with `ldd`) — a real
GPU build, not a CPU fallback.

## 4. Train (GPU)

Arch is pinned to `design/15 §3.2.4-5` / doc 11 §3.4.3: `Chess768` input,
`768 -> 2x256` dual-perspective SCReLU, `QA=255`, `QB=64`, `--scale 400`,
`--wdl 0.3`. Knobs are env vars (see `train_smoke.rs` header). Smoke run:

```bash
cd "$SCRATCH/bullet"
SMOKE_DATA=/root/projects/learnrust/chessengine/data/nnue/train.bin \
SMOKE_OUT="$SCRATCH/checkpoints" \
CUDA_PATH=/usr \
  ./target/release/examples/nnue_smoke
# defaults: HIDDEN 256, 10 superbatches x 50 batches x 16384 = 8.19M positions,
#           wdl 0.3, scale 400, StepLR 0.001.
```

### Smoke result

- Ran on the RTX 4060 Ti (`sm_89`); 10 superbatches in **~1.8 s** total.
- **Loss finite + monotonically decreasing:** running loss
  `0.0142 -> 1.9e-4 -> 4.4e-5 -> ... -> 6.6e-6` (per-superbatch `log.txt`). The
  collapse toward zero is expected/fine — a 256-wide net memorising a 5.2 k
  corpus. This is a **pipeline** proof, not a strength claim.
- **Net emitted + quantised** to `<out>/nnue_smoke-{5,10}/`:
  - `quantised.bin` = **394 816 B** = 197 377 `i16` params
    (l0w 768x256 + l0b 256 + l1w 512 + l1b 1) padded to a 64-B multiple —
    the exact `QA/QB` layout our future in-engine inference loads (doc 11
    §3.4.3). Its presence means quantisation did **not** overflow. Content is
    real: 194 578 / 197 408 non-zero, range [-105, 30050].
  - `raw.bin` = 789 508 B = 197 377 `f32` params; plus `optimiser_state/`.

### GPU throughput (for real-net extrapolation)

- Steady-state **~11-12.4 M positions/sec** for this `768->2x256` arch at batch
  16384 (first superbatch lower: ~0.6 -> 7.9 M/s NVRTC warm-up).
- **Caveat:** the tiny corpus lives entirely in page cache, so this is a
  compute+cache *upper bound*. A real 3.2 GB (100 M) corpus streams from disk
  through the host feature-mapper (4 threads here), so expect the loader, not
  the GPU, to bound throughput — realistically low-single-digit to ~10 M pos/s.
- Implication: a real training run (e.g. 40 superbatches x 100 M = 4 B positions)
  is **~6-15 min of GPU time**. Training is *not* the bottleneck — Tier-A
  **datagen** is (`design/15 §10.3`: ~1700-2100 core-hours for 100 M at
  `p_cap 10`). The GPU/box comfortably trains the planned net.
- **VRAM:** NNUE nets are tiny (768->2x256 ≈ 0.2 M params; activations at batch
  16384 are tens of MB). 8 GB is ample for this arch and far larger hidden
  sizes / batches; VRAM is not a limiter for any realistic NNUE here.

## 5. GO / NO-GO — training half

**GO.** On this machine (RTX 4060 Ti + CUDA 12.0 + rustc 1.92, WSL2):

1. `bulletformat` converter builds offline; produces valid 32-B `ChessBoard`
   records; all 5201 smoke records decode + cross-check with 0 mismatches.
2. `bullet` builds with the `cuda` feature in ~11 s and genuinely links CUDA.
3. Training runs on the GPU, loss is finite and decreasing, and a correctly
   quantised net (+ raw + optimiser state) is emitted.

Caveats / follow-ups (none blocking):
- crates.io must be reachable for the *first* bullet build (deps then cache);
  `bulletformat`/the converter are offline.
- The 11 M pos/s is a cached upper bound; measure the real rate on the first
  bulk corpus streamed from disk.
- For real training use `bullet-utils` shuffle+interleave (not the `cp`/`cat`
  fallback) so the loader doesn't see shard-ordered data.
- Wrapping the raw quantised `.bin` into the engine's `CENN` container +
  in-engine load/symmetry test is doc 11 §3.4.5/§3.4.7 (Phase C), out of scope
  here.
