# `archon-jax` — the NNUE trainer

ARCHON's NNUE training pipeline in JAX. It replaces the Rust [`bullet`] trainer, and it
exists as a **parity reimplementation**: every layer, loss, optimizer kernel and quantised
byte was proven equal to bullet's before any real run was launched. The engine's canonical
net (`engines/canonical/`) is trained here.

If you only read one thing, read [§2 Parity](#2-parity-is-the-point).

---

## 1. Quick start

```bash
cd jax
uv sync                                   # deps are PINNED — see §5
uv run pytest -q                          # 18 unit tests
uv run python tools/train_smoke.py        # CPU: overfit ~10k real rows, minutes
```

Real runs are driven by a config, not by flags:

```bash
uv run python tools/pump_train.py <run-config.json>          # start
uv run python tools/pump_train.py <run-config.json> --resume-sb 240   # resume
```

GPU is **opt-in** and needs the Pallas kernel (§4):

```bash
ARCHON_JAX_ALLOW_GPU=1 ARCHON_JAX_FT_IMPL=pallas uv run python tools/pump_train.py cfg.json
```

## 2. Parity is the point

The pipeline was not written to be "a JAX trainer". It was written to be **the same
trainer**, so that a net trained here is interchangeable with one trained by bullet and
the engine cannot tell the difference. Three proofs, in increasing scope — rerun them
after touching anything numeric:

| proof | tool | what it establishes |
|---|---|---|
| **1. quantised bytes** | `tools/quant_byte_parity.py` | re-quantise a real checkpoint through this code and get a **byte-identical** file to bullet's |
| **2. full-graph forward** | `tools/forward_parity.py` | the JAX forward on the **real** weights matches |
| **3. deploy level** | `tools/fen_probe.py`, `tools/engine_parity.py` | JAX float forward vs the **engine's int8 inference**, per-FEN and distributionally |

Everything numeric is transcribed from a specific line of the bullet source, and the
module docstrings cite it (`d40_features.rs`, `run.rs`). `optim.py` is bullet's AdamW as an
`optax` transformation — including the details that differ from textbook Adam: decay
applied as `p *= 1 - decay*lr` **before** the step, `eps` added to `sqrt(v)` and **not**
inside the sqrt, and a `±1.98` clip on every tensor **after** the step.

**If you change a constant in `spec.py`, you have probably broken parity.** Rerun proof 1.

## 3. Layout

```
archon_jax/
  spec.py        architecture + data contract. The single source of dims/quant constants.
  model.py       the graph, transcribed from bullet's ModelDefinition
  losses.py      loss functions matching bullet's graph exactly
  optim.py       bullet's AdamW as an optax GradientTransformation
  data.py        numpy structured-dtype reader over bulletformat .bin
  features.py    vectorized feature extraction for ChessBoard batches
  ft_kernel.py   GPU feature-transformer kernels — the sparse embedding-bag XLA can't do
  checkpoint.py  bullet raw-checkpoint IO (name-tagged f32 blobs)
  export.py      raw <-> quant, byte-compatible with bullet
  train.py       the training loop, with bullet run.rs semantics
tools/           drivers and proofs (see §2 and §6)
tests/           18 unit tests: features, ft kernel, freeze/ramp
```

**Architecture.** `HL = 1280` (accumulator per perspective), `L2 = 128`, `L3 = 32`,
8 output buckets, inputs = HalfKAv2_hm (22,528) + FullThreats (60,720). `ArchSpec` toggles
the optional pieces per run: `policy`, `halfka_only`, `factorize`, `psqt`, `out_skip`,
`pol_proj`, `dis_proj`. The width matches the engine's `arch-d48x`
(`src/eval/nnue/sota_head.rs`: `L1 = 1280`) — **but the engine's arch features and this
spec are two separate transcriptions of the same design, and nothing enforces agreement
automatically.** A net and a binary that disagree fail at load with a CENN section-size
error, which is the backstop, not the check. When adding an arch, change both and rerun
proof 3.

**Policy head.** `policy: bool = True` is the historical default, but **the policy program
was discontinued** (`pol_lambda 0`) — the head is dead weight for value training and
changes the quantised layout. Leave it off for new runs unless you are deliberately
reviving that line.

## 4. GPU: the kernel exists for a reason

The feature transformer is a **sparse embedding-bag** over ~192 active inputs per
position, and XLA has no good lowering for it: the obvious `jnp.take` reference path
**OOMs or crawls** on GPU. `ft_kernel.py` provides a Pallas kernel; select it with
`ARCHON_JAX_FT_IMPL=pallas`.

Default is CPU (`JAX_PLATFORMS=cpu`, no preallocation). GPU requires
`ARCHON_JAX_ALLOW_GPU=1` and `jax[cuda12]`, deliberately, so a stray import cannot grab
the card out from under a training run.

Throughput go/no-go: `tools/gpu_step_bench.py`.

## 5. Dependencies are pinned, on purpose

```
numpy==2.5.1   jax==0.11.0   optax==0.2.8
```

These are the exact versions the parity proofs ran on. Floor pins silently resolved to
them during the 2026-07-22 study, so they were frozen. **Bump deliberately, and rerun
`quant_byte_parity.py` and `forward_parity.py` afterwards** — a numerics change in any of
the three breaks the byte-identity claim, which is the whole basis for trusting a net
trained here.

The legal-move policy mask needs movegen; `chess>=1.10` (extra `policy`) is the prototype.
`archon_mask.so` is a build artifact of the `pymask/` PyO3 crate — untracked, rebuild it
rather than copy it.

## 6. Tools

| tool | use |
|---|---|
| `pump_train.py` | the run driver: config in, checkpoints out, resumable |
| `stage3_train.py` | a pinned historical recipe; read it as a worked example of a real config |
| `train_smoke.py` | CPU smoke — overfit ~10k real rows on a scaled-down graph |
| `bench_loader.py` | loader throughput |
| `gpu_step_bench.py` | jit'd train-step throughput on GPU |
| `export_warm_quant.py` | RAW checkpoint → bullet `quant.bin` under a run-config's arch |
| `quant_byte_parity.py`, `forward_parity.py`, `fen_probe.py`, `engine_parity.py` | the three proofs (§2) |
| `s5_sparsity_probe.py`, `policy_heldout_check.py` | probes from specific campaigns; historical |

## 7. Where the surrounding rules live

- **Data recipe is governed** — read `docs/research/data-recipe-contract.md` before
  building or modifying any corpus. Scale multipliers are derived and verified, filters
  run before clamps, and `tools/nnue/net_scale_check.sh` is a hard pre-game gate.
- **A trained net is not canonical until it is pointed at** — see
  `engines/canonical/MANIFEST.md` for how to promote one and what evidence to record.
- **The engine side of the contract** is `src/eval/nnue/` and the behavioural gate
  `CHECK_NNUE_GATE=1 bash tools/check.sh`, which is mandatory for anything under it.

[`bullet`]: https://github.com/jw1912/bullet
