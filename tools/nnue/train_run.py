#!/usr/bin/env python3
"""NNUE training-run orchestrator — the RUNBOOK.md pipeline as one command.

Training a net here is a *solved procedure* (tools/nnue/RUNBOOK.md): corpus →
train → verify → pack → sanity → SPRT → ledger. This script runs that whole
pipeline from a config file, so a run is "set the params, run one command" —
no subagent, no hand-orchestration.

    uv run python tools/nnue/train_run.py --config tools/nnue/runs/net-X.json
    uv run python tools/nnue/train_run.py --config … --dry-run     # print the plan, run nothing
    uv run python tools/nnue/train_run.py --config … --skip corpus,sprt
    uv run python tools/nnue/train_run.py --config … --only sprt,ledger   # resume a partial run
    uv run python tools/nnue/train_run.py --config … --set superbatches=45 --set sprt.nodes=25000

Stdlib-only (matches the zero-dep tooling philosophy); it shells out to the
existing tools (build_corpus.sh, bullet, netpack.py, chessengine nnue-eval,
fastchess, ledger.py). Every stage fails FAST with the failing command shown; a
`run.json` manifest (all resolved params + per-stage status/timing/results) is
written to tools/nnue/runs/<net_id>/ for reproducibility.

Design notes:
- The pipeline NEVER pushes and, by default, does not commit; `--commit` makes a
  LOCAL commit of the ledger rows (git-tracked) only (D38 provenance). Nets and
  corpora stay gitignored.
- The heavy stages (corpus, train, sprt) are the slow ones; use --skip/--only to
  resume after a failure without redoing them.
- Arch-specific quirks (the classic-net i16 out-weight clip vs the D40 hybrid
  save that hard-checks overflow) are config flags, not hardcoded — see the
  template in RUNBOOK.md §Automation.
"""

import argparse
import json
import os
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGES = ["preflight", "corpus", "train", "verify", "pack", "parity", "sanity", "sprt", "ledger"]

# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #

DEFAULTS = {
    "net_id": None,                       # REQUIRED — new net id, unused in nets.jsonl
    "arch": "768x2x256x1",                # ledger arch tag
    "provenance": "sf-distill",           # sf-distill | lc0-distill | self-play
    "parent": None,                       # parent net id (lineage)
    "notes": "",
    "engine_features": None,              # extra cargo features for the engine build (e.g. "sota" for D40)
    # --- corpus (stage 1) --- build a train.bin from text shards, or point at a prebuilt one
    "corpus": {
        "id": None,                       # ledger data.jsonl id (provenance)
        "raw_dir": None,                  # text shards FEN|cp|result → build_corpus.sh (None ⇒ skip build)
        "train_bin": "data/nnue/sf/train.bin",  # the bin the trainer reads (built here or pre-existing)
        "out_tag": None,                  # build_corpus.sh output tag (defaults to net_id)
    },
    # --- train (stage 2) --- bullet, GPU
    "train": {
        "bullet_dir": None,               # the bullet checkout (REQUIRED for a real train; ephemeral scratchpad)
        "example": "train_net0.rs",       # the tools/nnue/*.rs example to copy into the checkout
        "example_name": None,             # [[example]] name (defaults to file stem)
        "features": "cuda",
        "env": {                          # env knobs passed to the example (see its header)
            "NET0_SUPERBATCHES": "30",
            "NET0_BPS": "1520",
            "NET0_BATCH": "16384",
            "NET0_LR": "0.001",
            "NET0_WDL": "0.3",
        },
        "checkpoint_dir": None,           # where the example writes ckpts (defaults to tools/nnue/runs/<net_id>/ckpt)
        "quantised": "quantised.bin",     # the file inside the final-superbatch ckpt dir
    },
    # --- verify (stage 3) --- the out-weight clip (classic arch); D40 hybrid save hard-checks itself
    "verify_out_clip": True,
    # --- pack (stage 4) --- netpack.py → .cenn
    "pack": {"activation": "screlu", "format": "classic", "input": 768},
    # --- parity (stage 5) --- raw Bullet logits vs packed Rust logits
    "parity": {
        "enable": False,
        "fens": "tools/nnue/d40_probe.fens",
        "raw": None,                      # defaults to <checkpoint_dir>/<net_id>-final-raw.bin
        "max_abs_quantization_logit_error": 1.0,
        "mean_abs_quantization_logit_error": 0.25,
        # Runtime bars recalibrated 2026-07-17 (D45a diagnosis,
        # docs/research/d40-packing-tail-2026-07-17.md): the runtime error is
        # the engine's integer forward vs bullet's float forward on the SAME
        # weights — dominated by the (left*right) >> FT_SHIFT floor, NOT a
        # packing defect (zero tensor values overflow; bullet errors rather
        # than saturates on overflow). A thin OOD-endgame tail is arithmetic
        # noise, so the gate now bounds the DISTRIBUTION (mean + p99) and
        # keeps only a coarse absolute backstop against real transform bugs
        # (sf-100's true failure was mean 0.355 BROAD, caught by mean/p99
        # long before max). Strength remains the real gate: Elo testing.
        "max_abs_runtime_logit_error": 0.3,
        "p99_abs_runtime_logit_error": 0.1,
        "mean_abs_runtime_logit_error": 0.03,
    },
    # --- sanity (stage 6) --- nnue-eval load + semantic score constraints
    "sanity_fens": [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        "4k3/8/8/8/8/8/8/QQQQK3 w - - 0 1",   # white crushing
        "qqqqk3/8/8/8/8/8/8/4K3 w - - 0 1",   # black crushing
    ],
    # --- sprt (stage 6) --- fixed-node net-vs-net vs baseline (D39: contention-immune)
    "sprt": {
        "enable": True,
        "baseline_net": "data/nnue/nets/net-8.cenn",
        "baseline_id": "net-8",
        "nodes": 25000,
        "rounds": 1000,
        "concurrency": 8,
        "book": "tests/data/openings/openings-8ply.epd",
    },
}


def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def set_path(cfg, dotted, value):
    """Apply a `--set a.b=value` override (value JSON-parsed, else string)."""
    try:
        value = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path, sets):
    cfg = deep_merge(DEFAULTS, json.loads(Path(path).read_text()))
    for s in sets or []:
        k, _, v = s.partition("=")
        set_path(cfg, k.strip(), v.strip())
    if not cfg.get("net_id"):
        die("config: net_id is required")
    # derive defaults that depend on net_id
    run_dir = REPO / "tools/nnue/runs" / cfg["net_id"]
    cfg.setdefault("_run_dir", str(run_dir))
    if not cfg["train"].get("checkpoint_dir"):
        cfg["train"]["checkpoint_dir"] = str(run_dir / "ckpt")
    if not cfg["train"].get("example_name"):
        cfg["train"]["example_name"] = Path(cfg["train"]["example"]).stem
    if not cfg["corpus"].get("out_tag"):
        cfg["corpus"]["out_tag"] = cfg["net_id"]
    return cfg


# --------------------------------------------------------------------------- #
# shell helpers                                                                #
# --------------------------------------------------------------------------- #

class Ctx:
    def __init__(self, cfg, dry):
        self.cfg = cfg
        self.dry = dry
        self.results = {}  # per-stage result dict for the manifest

def run(ctx, cmd, *, cwd=None, env=None, stage="", capture=False):
    """Run a command, fail-fast on nonzero. In --dry-run, print and skip."""
    shown = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
    print(f"    $ {shown}")
    if ctx.dry:
        return ""
    full_env = {**os.environ, **(env or {})}
    if capture:
        p = subprocess.run(cmd, cwd=cwd or REPO, env=full_env, shell=isinstance(cmd, str),
                           text=True, capture_output=True)
        if p.returncode != 0:
            sys.stderr.write(p.stdout + p.stderr)
            die(f"stage '{stage}': command failed (exit {p.returncode})")
        return p.stdout
    p = subprocess.run(cmd, cwd=cwd or REPO, env=full_env, shell=isinstance(cmd, str))
    if p.returncode != 0:
        die(f"stage '{stage}': command failed (exit {p.returncode})")
    return ""


def die(msg):
    sys.stderr.write(f"\n✗ {msg}\n")
    sys.exit(1)


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------- #
# stages                                                                       #
# --------------------------------------------------------------------------- #

def stage_preflight(ctx):
    """Confirm the engine is in a good, reproducible state for the experiment."""
    cfg = ctx.cfg
    # fast release build (the strength/SPRT config, D16) — this is the binary the
    # sanity + SPRT stages use, so build it here and record the sha.
    features = cfg.get("engine_features")
    feat_flag = f" --features {features}" if features else ""
    run(ctx, f"RUSTFLAGS='-C target-cpu=x86-64-v3' cargo build --release{feat_flag}", stage="preflight")
    sha = run(ctx, ["git", "rev-parse", "--short", "HEAD"], stage="preflight", capture=True).strip()
    dirty = run(ctx, ["git", "status", "--porcelain", "src", "Cargo.toml"],
                stage="preflight", capture=True).strip()
    if dirty and not ctx.dry:
        print("    ⚠ uncommitted changes under src/ or Cargo.toml — the net's code provenance is not a clean sha:")
        print("      " + dirty.replace("\n", "\n      "))
    # net_id must be free in the ledger (append-only; catch collisions early)
    if not ctx.dry:
        nets = REPO / "ledger/nets.jsonl"
        if nets.exists() and any(json.loads(l).get("id") == cfg["net_id"]
                                 for l in nets.read_text().splitlines() if l.strip()):
            die(f"net_id '{cfg['net_id']}' already in ledger/nets.jsonl (append-only)")
    ctx.results["preflight"] = {"git_sha": sha or "dry", "dirty": bool(dirty)}
    print(f"    engine built @ {sha or 'dry-run'}  (net_id '{cfg['net_id']}' free)")


def stage_corpus(ctx):
    """Build train.bin from text shards, or accept a prebuilt bin."""
    c = ctx.cfg["corpus"]
    train_bin = REPO / c["train_bin"]
    if not c.get("raw_dir"):
        if not train_bin.exists() and not ctx.dry:
            die(f"corpus.raw_dir unset and corpus.train_bin missing: {train_bin}")
        print(f"    using prebuilt corpus: {rel(train_bin)}")
        ctx.results["corpus"] = {"train_bin": rel(train_bin), "built": False}
        return
    env = {"CORPUS_DIR": str(REPO / c["raw_dir"]), "OUT_TAG": c["out_tag"]}
    run(ctx, "bash tools/nnue/build_corpus.sh", env=env, stage="corpus")
    ctx.results["corpus"] = {"train_bin": rel(train_bin), "built": True, "raw_dir": c["raw_dir"]}


def stage_train(ctx):
    """Copy the bullet example into the checkout, build (CUDA), run it."""
    t = ctx.cfg["train"]
    if not t.get("bullet_dir"):
        die("train.bullet_dir is required for a real train (the ephemeral bullet checkout)")
    bdir = Path(t["bullet_dir"])
    example_src = REPO / "tools/nnue" / t["example"]
    if not example_src.exists() and (REPO / t["example"]).exists():
        example_src = REPO / t["example"]
    ex_name = t["example_name"]
    # 1. place the example + ensure a [[example]] entry exists
    dst = bdir / "examples" / f"{ex_name}.rs"
    run(ctx, ["cp", str(example_src), str(dst)], stage="train")
    cargo = bdir / "crates/bullet_lib/Cargo.toml"
    if not ctx.dry and cargo.exists() and f'name = "{ex_name}"' not in cargo.read_text():
        with cargo.open("a") as f:
            f.write(f'\n[[example]]\nname = "{ex_name}"\npath = "../../examples/{ex_name}.rs"\n')
        print(f"    registered [[example]] {ex_name} in {rel(cargo)}")
    # 2. build + run (from REPO root — the uv/env gotcha in RUNBOOK step 3)
    ckpt = Path(t["checkpoint_dir"])
    # Start with the config's explicit env, then add fallbacks for the classic
    # NET0_* convention. The explicit env wins — so a D40 config that sets
    # D40_DATA/D40_OUT/D40_ID skips these, while a classic config gets them free.
    env = {"NET0_DATA": str(REPO / ctx.cfg["corpus"]["train_bin"]),
           "NET0_OUT": str(ckpt),
           "NET0_ID": ctx.cfg["net_id"],
           "CUDA_PATH": os.environ.get("CUDA_PATH", "/usr")}
    # CLI --set values are JSON-decoded, so numeric env overrides arrive as
    # ints/floats even though subprocess requires string environment values.
    # Stringify at the boundary; config's explicit values still win.
    env.update({key: str(value) for key, value in t.get("env", {}).items()})
    manifest = f"{bdir}/Cargo.toml"
    run(ctx, ["cargo", "run", "--release", "--features", t["features"],
              "--manifest-path", manifest, "--example", ex_name], env=env, stage="train")
    ctx.results["train"] = {"example": t["example"], "checkpoint_dir": str(ckpt),
                            "env": t.get("env", {})}


def _quantised_path(ctx):
    t = ctx.cfg["train"]
    ckpt = Path(t["checkpoint_dir"])
    # Classic net: bullet writes <id>-<sb>/quantised.bin in subdirectories.
    # D40 net: bullet writes <id>-sb<N>-quant.bin directly in the output dir.
    if ctx.dry:
        return ckpt / "final" / t["quantised"]
    # 1. Check flat layout first (D40: file directly in ckpt dir).
    flat = ckpt / t["quantised"]
    if flat.exists():
        return flat
    # 2. Check subdirectory layout (classic: ckpt/<subdir>/quantised.bin).
    subs = sorted([d for d in ckpt.glob("*/") if (d / t["quantised"]).exists()],
                  key=lambda d: d.stat().st_mtime)
    if not subs:
        die(f"no {t['quantised']} under {ckpt} — did the train stage run?")
    return subs[-1] / t["quantised"]


def stage_verify(ctx):
    """Out-weight clip check (classic arch); the D40 hybrid save hard-checks itself."""
    if not ctx.cfg.get("verify_out_clip"):
        print("    verify_out_clip=false — bullet's save hard-checks quant overflow; skipping")
        ctx.results["verify"] = {"checked": False}
        return
    q = _quantised_path(ctx)
    print(f"    checking |out_weight| ≤ 127 in {rel(q)}")
    if ctx.dry:
        return
    import struct
    data = q.read_bytes()
    # Canonical classic layout is l0w[768*256], l0b[256], l1w[512], l1b[1],
    # all i16, followed by optional alignment padding.  Do not inspect a tail
    # window: that includes l0 weights and opaque padding and caused false
    # failures despite a valid clipped l1 block.
    width = int(ctx.cfg["pack"].get("l1", 256))
    input_width = int(ctx.cfg["pack"].get("input", 768))
    out_offset = 2 * (input_width * width + width)
    out_bytes = 2 * (2 * width)
    if len(data) < out_offset + out_bytes:
        die(f"verify: classic quantised file is truncated ({len(data)} bytes)")
    vals = struct.unpack(f"<{2 * width}h", data[out_offset:out_offset + out_bytes])
    mx = max(abs(v) for v in vals)
    if mx > 127:
        die(f"out-weight clip VIOLATED (max |w|={mx} > 127) — tighten the training clip and retrain "
            f"(the loader will reject this net; RUNBOOK step 3)")
    ctx.results["verify"] = {"checked": True, "max_out_weight": mx}
    print(f"    clip OK (max |out_weight| = {mx})")


def stage_pack(ctx):
    """netpack.py: quantised.bin → data/nnue/nets/<net_id>.cenn."""
    q = _quantised_path(ctx)
    out = REPO / "data/nnue/nets" / f"{ctx.cfg['net_id']}.cenn"
    out.parent.mkdir(parents=True, exist_ok=True)
    if ctx.cfg["pack"].get("format") == "halfka-classic":
        run(ctx, ["uv", "run", "--no-project", "python",
                  "tools/nnue/pack_halfka_classic.py", str(q), str(out)], stage="pack")
        ctx.results["pack"] = {"cenn": rel(out), "format": "CENN-v3"}
        ctx.cfg["_cenn"] = str(out)
        return
    strip_threats = bool(ctx.cfg["pack"].get("strip_d40_threats", False))
    packed = out.with_suffix(".with-threats.cenn") if strip_threats else out
    pack_env = {}
    if ctx.cfg["pack"].get("l1") is not None:
        pack_env["NET_L1"] = str(ctx.cfg["pack"]["l1"])
    run(ctx, ["uv", "run", "--no-project", "python", "tools/nnue/netpack.py",
              str(q), str(packed), ctx.cfg["pack"]["activation"]], env=pack_env, stage="pack")
    if strip_threats:
        run(ctx, ["uv", "run", "--no-project", "python", "tools/nnue/netpack.py",
                  "--strip-d40-threats", str(packed), str(out)], stage="pack")
        if not ctx.dry:
            packed.unlink()
    ctx.results["pack"] = {"cenn": rel(out)}
    ctx.cfg["_cenn"] = str(out)


def stage_sanity(ctx):
    """nnue-eval: require both loadability and configured score semantics."""
    cenn = ctx.cfg.get("_cenn") or str(REPO / "data/nnue/nets" / f"{ctx.cfg['net_id']}.cenn")
    binp = REPO / "target/release/chessengine"
    scores = []
    for entry in ctx.cfg["sanity_fens"]:
        spec = entry if isinstance(entry, dict) else {"fen": entry}
        fen = spec["fen"]
        out = run(ctx, [str(binp), "nnue-eval", cenn, fen], stage="sanity", capture=True)
        if not ctx.dry:
            last = out.strip().splitlines()[-1] if out.strip() else ""
            try:
                score = int(last)
            except ValueError:
                die(f"sanity: non-integer output for {fen}: {last!r}")
            scores.append(score)
            print(f"    {fen[:32]:32s} → {score}")
            if "min_cp" in spec and score < int(spec["min_cp"]):
                die(f"sanity: {score} < min_cp {spec['min_cp']} for {fen}")
            if "max_cp" in spec and score > int(spec["max_cp"]):
                die(f"sanity: {score} > max_cp {spec['max_cp']} for {fen}")
    ctx.results["sanity"] = {
        "cenn": rel(cenn), "fens": len(ctx.cfg["sanity_fens"]), "scores": scores,
    }


def stage_parity(ctx):
    """Reject a packed D40 net that does not reproduce its raw Bullet graph."""
    parity = ctx.cfg["parity"]
    if not parity.get("enable", False):
        print("    parity.enable=false — skipping trainer/runtime parity")
        ctx.results["parity"] = {"ran": False}
        return
    train = ctx.cfg["train"]
    bullet_dir = Path(train["bullet_dir"])
    example_src = REPO / "tools/nnue" / train["example"]
    if not example_src.exists() and (REPO / train["example"]).exists():
        example_src = REPO / train["example"]
    run(ctx, ["cp", str(example_src),
              str(bullet_dir / "examples" / f"{train['example_name']}.rs")],
        stage="parity")
    raw = Path(parity.get("raw") or
               Path(train["checkpoint_dir"]) / f"{ctx.cfg['net_id']}-final-raw.bin")
    fens_path = REPO / parity["fens"]
    cenn = Path(ctx.cfg.get("_cenn") or REPO / "data/nnue/nets" / f"{ctx.cfg['net_id']}.cenn")
    for required in (raw, fens_path, cenn):
        if not required.exists():
            die(f"parity: missing {required}")

    base_env = dict(train.get("env", {}))
    base_env.update({
        "D40_PROBE_FENS": str(fens_path.resolve()),
        "D40_DATA": str((REPO / ctx.cfg["corpus"]["train_bin"]).resolve()),
    })
    manifest = bullet_dir / "Cargo.toml"
    probe_cmd = ["cargo", "run", "--release", "--features", train["features"],
                 "--manifest-path", str(manifest), "--example", train["example_name"]]

    def trainer_probe(kind, path):
        env = {**base_env, f"D40_PROBE_{kind}": str(path.resolve())}
        output = run(ctx, probe_cmd, env=env, stage="parity", capture=True)
        prefix = f"D40_PROBE_{kind}\t"
        values, fens = [], []
        for line in output.splitlines():
            if line.startswith(prefix):
                _, value, fen = line.split("\t", 2)
                values.append(float(value))
                fens.append(fen)
        if not values:
            die(f"parity: trainer emitted no D40_PROBE_{kind} records")
        return values, fens

    raw_logits, probe_fens = trainer_probe("RAW", raw)
    quant_logits, quant_fens = trainer_probe("QUANT", _quantised_path(ctx))
    if quant_fens != probe_fens:
        die("parity: raw and quantized trainer probes emitted different FENs")

    print(f"    $ {REPO / 'target/release/chessengine'} nnue-eval {cenn} < {fens_path}")
    proc = subprocess.run(
        [str(REPO / "target/release/chessengine"), "nnue-eval", str(cenn)],
        input="\n".join(probe_fens) + "\n", text=True, capture_output=True,
        cwd=REPO, env=os.environ,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        die(f"parity: engine probe failed (exit {proc.returncode})")
    engine_cp = [int(line) for line in proc.stdout.splitlines() if line.strip()]
    if len(engine_cp) != len(raw_logits):
        die(f"parity: trainer emitted {len(raw_logits)} rows, engine emitted {len(engine_cp)}")
    network_scale = struct.unpack_from("<i", cenn.read_bytes(), 44)[0]
    quant_errors = [abs(quant - raw) for quant, raw in zip(quant_logits, raw_logits)]
    runtime_errors = [abs(cp / network_scale - quant)
                      for cp, quant in zip(engine_cp, quant_logits)]
    def percentile(values, fraction):
        ordered = sorted(values)
        at = fraction * (len(ordered) - 1)
        lo = int(at)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)

    quant_max, quant_mean = max(quant_errors), sum(quant_errors) / len(quant_errors)
    runtime_max = max(runtime_errors)
    runtime_mean = sum(runtime_errors) / len(runtime_errors)
    quant_p95, quant_p99 = percentile(quant_errors, 0.95), percentile(quant_errors, 0.99)
    runtime_p95, runtime_p99 = percentile(runtime_errors, 0.95), percentile(runtime_errors, 0.99)
    worst = max(range(len(runtime_errors)), key=runtime_errors.__getitem__)
    print(f"    parity rows={len(runtime_errors)} "
          f"quant mean/max={quant_mean:.6f}/{quant_max:.6f} "
          f"runtime mean/max={runtime_mean:.6f}/{runtime_max:.6f}")
    print(f"    parity quant p95/p99={quant_p95:.6f}/{quant_p99:.6f} "
          f"runtime p95/p99={runtime_p95:.6f}/{runtime_p99:.6f}")
    print(f"    worst quant={quant_logits[worst]:.6f} "
          f"runtime={engine_cp[worst] / network_scale:.6f} "
          f"fen={probe_fens[worst]}")
    gates = [
        (quant_max, "max_abs_quantization_logit_error"),
        (quant_mean, "mean_abs_quantization_logit_error"),
        (runtime_max, "max_abs_runtime_logit_error"),
        (runtime_mean, "mean_abs_runtime_logit_error"),
    ]
    if "p99_abs_quantization_logit_error" in parity:
        gates.append((quant_p99, "p99_abs_quantization_logit_error"))
    if "p99_abs_runtime_logit_error" in parity:
        gates.append((runtime_p99, "p99_abs_runtime_logit_error"))
    for value, key in gates:
        if value > float(parity[key]):
            die(f"parity: {key} observed {value:.6f} exceeds {parity[key]}")
    ctx.results["parity"] = {
        "ran": True, "rows": len(runtime_errors),
        "quantization_mean_abs_logit_error": quant_mean,
        "quantization_p95_abs_logit_error": quant_p95,
        "quantization_p99_abs_logit_error": quant_p99,
        "quantization_max_abs_logit_error": quant_max,
        "runtime_mean_abs_logit_error": runtime_mean,
        "runtime_p95_abs_logit_error": runtime_p95,
        "runtime_p99_abs_logit_error": runtime_p99,
        "runtime_max_abs_logit_error": runtime_max,
    }


def stage_sprt(ctx):
    """Fixed-node net-vs-net vs the baseline (D39: contention-immune isolation)."""
    s = ctx.cfg["sprt"]
    if not s.get("enable", True):
        print("    sprt.enable=false — skipping the strength gate")
        ctx.results["sprt"] = {"ran": False}
        return
    cenn = ctx.cfg.get("_cenn") or str(REPO / "data/nnue/nets" / f"{ctx.cfg['net_id']}.cenn")
    binp = REPO / "target/release/chessengine"
    fc = REPO / "engines/bin/fastchess"
    outdir = Path(ctx.cfg["_run_dir"]) / "sprt"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{ctx.cfg['net_id']}_vs_{s['baseline_id']}"
    cmd = [str(fc),
           "-engine", f"cmd={binp}", f"name={ctx.cfg['net_id']}", f"option.EvalFile={cenn}",
           "-engine", f"cmd={binp}", f"name={s['baseline_id']}", f"option.EvalFile={REPO/s['baseline_net']}",
           "-each", "proto=uci", "tc=inf", f"nodes={s['nodes']}", "option.Threads=1", "option.Hash=64",
           "-openings", f"file={REPO/s['book']}", "format=epd", "order=random",
           "-repeat", "-rounds", str(s["rounds"]), "-games", "2", "-concurrency", str(s["concurrency"]),
           "-draw", "movenumber=40", "movecount=8", "score=15", "-resign", "movecount=4", "score=700",
           "-ratinginterval", "50", "-pgnout", f"file={outdir/(tag+'.pgn')}", "notation=lan",
           "-srand", "20260713"]
    log = run(ctx, cmd, stage="sprt", capture=True)
    if not ctx.dry:
        (outdir / f"{tag}.log").write_text(log)
        summary = [l for l in log.splitlines() if any(k in l for k in ("Elo:", "Games:", "Ptnml", "LOS", "SPRT"))]
        print("    " + "\n    ".join(summary[-6:]) if summary else "    (no parseable SPRT summary — see log)")
        ctx.results["sprt"] = {"ran": True, "tag": tag, "log": rel(outdir / f"{tag}.log"),
                               "summary": summary[-6:]}
    ctx.cfg["_sprt_tag"] = tag


def stage_ledger(ctx):
    """Register the net (+ the SPRT exp) in the D38 ledger; optional local commit."""
    cfg = ctx.cfg
    cenn = cfg.get("_cenn") or f"data/nnue/nets/{cfg['net_id']}.cenn"
    hparams = json.dumps({**cfg["train"].get("env", {}), "example": cfg["train"]["example"]})
    add_net = ["uv", "run", "--no-project", "python", "tools/ledger/ledger.py", "add-net",
               "--id", cfg["net_id"], "--file", str(cenn), "--arch", cfg["arch"],
               "--provenance", cfg["provenance"], "--hparams", hparams,
               "--recipe", f"tools/nnue/train_run.py --config {cfg.get('_config_path','<config>')}"]
    if cfg["corpus"].get("id"):
        add_net += ["--train-data", cfg["corpus"]["id"]]
    if cfg.get("parent"):
        add_net += ["--parent", cfg["parent"]]
    if cfg.get("notes"):
        add_net += ["--notes", cfg["notes"]]
    run(ctx, add_net, stage="ledger")
    if cfg["sprt"].get("enable", True) and ctx.results.get("sprt", {}).get("ran"):
        run(ctx, ["uv", "run", "--no-project", "python", "tools/ledger/ledger.py", "add-exp",
                  "--kind", "sprt", "--purpose", f"{cfg['net_id']} vs {cfg['sprt']['baseline_id']}",
                  "--a", json.dumps({"net": cfg["net_id"]}),
                  "--b", json.dumps({"net": cfg["sprt"]["baseline_id"]}),
                  "--tc", f"fixed-node {cfg['sprt']['nodes']}",
                  "--notes", "auto: train_run.py fixed-node screen"], stage="ledger")
    run(ctx, ["uv", "run", "--no-project", "python", "tools/ledger/ledger.py", "verify"], stage="ledger")
    ctx.results["ledger"] = {"registered": cfg["net_id"]}
    if cfg.get("_commit"):
        run(ctx, ["git", "add", "ledger/"], stage="ledger")
        run(ctx, ["git", "commit", "-m",
                  f"ledger: {cfg['net_id']} ({cfg['arch']}, {cfg['provenance']}) + SPRT vs {cfg['sprt']['baseline_id']}"],
            stage="ledger")


STAGE_FNS = {
    "preflight": stage_preflight, "corpus": stage_corpus, "train": stage_train,
    "verify": stage_verify, "pack": stage_pack, "parity": stage_parity, "sanity": stage_sanity,
    "sprt": stage_sprt, "ledger": stage_ledger,
}


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="NNUE training-run orchestrator (RUNBOOK.md as one command)")
    ap.add_argument("--config", required=True, help="JSON run config (see RUNBOOK.md §Automation)")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved plan + commands, run nothing")
    ap.add_argument("--only", help="comma list: run ONLY these stages")
    ap.add_argument("--skip", help="comma list: skip these stages")
    ap.add_argument("--set", action="append", dest="sets", help="override a config key, e.g. --set sprt.nodes=25000")
    ap.add_argument("--commit", action="store_true", help="LOCAL commit of the ledger rows (never pushes)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.sets)
    cfg["_config_path"] = args.config
    cfg["_commit"] = args.commit
    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()
    plan = [s for s in STAGES if (only is None or s in only) and s not in skip]

    print(f"\n=== NNUE training run: {cfg['net_id']} ({cfg['arch']}, {cfg['provenance']}) ===")
    print(f"    stages: {' → '.join(plan)}" + ("   [DRY RUN]" if args.dry_run else ""))
    run_dir = Path(cfg["_run_dir"])
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    ctx = Ctx(cfg, args.dry_run)
    t0 = time.time()
    for i, s in enumerate(plan, 1):
        ts = time.time()
        print(f"\n[{i}/{len(plan)}] {s}")
        STAGE_FNS[s](ctx)
        ctx.results.setdefault(s, {})["seconds"] = round(time.time() - ts, 1)

    if not args.dry_run:
        manifest = {k: v for k, v in cfg.items() if not k.startswith("_")}
        manifest["_results"] = ctx.results
        manifest["_stages_run"] = plan
        manifest["_total_seconds"] = round(time.time() - t0, 1)
        (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
        print(f"\n✓ done in {manifest['_total_seconds']}s — manifest: {rel(run_dir / 'run.json')}")
    else:
        print("\n✓ dry run complete (nothing executed)")


if __name__ == "__main__":
    main()
