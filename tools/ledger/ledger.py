#!/usr/bin/env python3
"""Experimental ledger CLI (D38, design/18).

A content-addressed provenance DAG + results ledger for the chessengine
project. Three append-only JSONL registries under `ledger/`:

    nets.jsonl         one row per trained NNUE net
    data.jsonl         one row per training corpus
    experiments.jsonl  one row per match / SPRT / tuning run

Every artifact (net, corpus) is identified by its **sha256** (identity is the
hash, not the friendly `id`) and carries a **regeneration recipe** (git SHA +
inputs + command). The heavy bytes stay in gitignored stores (local now, GCS
mirror later, D36); these registries — small, diffable, git-tracked — are the
source of truth for metadata + hashes.

Stdlib only (no deps): runs as `python3 tools/ledger/ledger.py …` or
`uv run python tools/ledger/ledger.py …`.

Commands:
    add-net    register a trained net (hashes the .cenn file)
    add-data   register a corpus (manifest-hashes a file or shard dir)
    add-exp    record a match / SPRT / tuning result
    annotate   append a status/correction row for an existing id
    verify     recompute hashes vs the registry + check referential integrity
    show       print nets | data | exps | dag | milestone-stub
    reproduce  print the recipe to regenerate an artifact or rerun an experiment

Corrections (annotations.jsonl): registry rows are never edited — a fourth
append-only stream records status changes (`lost`/`superseded`/`corrected`/
`waived`) and metadata overlays (`--set key=value`) against existing rows.
verify/show/reproduce read rows with overlays applied; a `lost` status stops
verify counting that artifact's drift as a problem (the bytes are gone and
acknowledged, the recorded hash remains the historical identity).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "ledger"
NETS = LEDGER / "nets.jsonl"
DATA = LEDGER / "data.jsonl"
EXPS = LEDGER / "experiments.jsonl"
ANNS = LEDGER / "annotations.jsonl"


# ---- helpers ---------------------------------------------------------------

def today() -> str:
    return datetime.date.today().isoformat()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> tuple[str, int]:
    """(sha256 hex, byte size) of one file, chunked."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def sha256_tree(path: Path, glob: str, strict: bool = True) -> tuple[str | None, int]:
    """Manifest hash of a shard dir: sha256 over sorted `name:filehash` lines.

    Deterministic regardless of read chunking or filesystem order, so a corpus
    has one stable identity across machines. `strict` dies on an empty match
    (right for add-*: never register an empty dir); non-strict returns
    (None, 0) so verify can report the row instead of aborting the whole run.
    """
    files = sorted(p for p in path.glob(glob) if p.is_file())
    if not files:
        if strict:
            die(f"no files match {path}/{glob}")
        return None, 0
    manifest = hashlib.sha256()
    total = 0
    for p in files:
        fh, sz = sha256_file(p)
        total += sz
        manifest.update(f"{p.name}:{fh}\n".encode())
    return manifest.hexdigest(), total


def hash_path(path: Path, glob: str, strict: bool = True) -> tuple[str | None, int]:
    return sha256_file(path) if path.is_file() else sha256_tree(path, glob, strict)


def load(fp: Path) -> list[dict]:
    if not fp.exists():
        return []
    return [json.loads(ln) for ln in fp.read_text().splitlines() if ln.strip()]


def append(fp: Path, row: dict) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def ann_index() -> dict[tuple[str, str], dict]:
    """(target_kind, target_id) -> {'status': latest, 'set': merged overlay}.

    Annotations apply in file order, so a later row wins for `status` and
    later `--set` values override earlier ones per key.
    """
    idx: dict[tuple[str, str], dict] = {}
    for r in load(ANNS):
        e = idx.setdefault((r["target_kind"], r["target_id"]),
                           {"status": None, "set": {}})
        if r.get("status"):
            e["status"] = r["status"]
        e["set"].update(r.get("set") or {})
    return idx


def load_merged(fp: Path, kind: str) -> list[dict]:
    """Registry rows with annotation overlays applied (rows on disk stay
    append-only; corrections live in annotations.jsonl — design/18 §5)."""
    idx = ann_index()
    out = []
    for r in load(fp):
        ov = idx.get((kind, r["id"]))
        if ov and ov["set"]:
            r = {**r, **ov["set"]}
        out.append(r)
    return out


def die(msg: str) -> None:
    print(f"ledger: error: {msg}", file=sys.stderr)
    sys.exit(1)


def maybe_json(s: str | None):
    if s is None:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        die(f"expected JSON, got: {s}")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


# ---- commands --------------------------------------------------------------

def cmd_add_net(a) -> None:
    ids = {r["id"] for r in load(NETS)}
    if a.id in ids:
        die(f"net id already registered: {a.id} (ledger is append-only)")
    path = Path(a.file)
    if not path.exists():
        die(f"net file not found: {path}")
    digest, size = sha256_file(path)
    data_ids = {r["id"] for r in load(DATA)}
    if a.train_data and a.train_data not in data_ids:
        print(f"ledger: warning: train_data '{a.train_data}' not in data.jsonl "
              f"(register the corpus with add-data)", file=sys.stderr)
    if a.parent and a.parent not in ids:
        print(f"ledger: warning: parent net '{a.parent}' not in nets.jsonl",
              file=sys.stderr)
    row = {
        "id": a.id, "hash": f"sha256:{digest}", "size": size,
        "arch": a.arch, "act": a.act, "quant": a.quant,
        "train_data": a.train_data, "parent_net": a.parent,
        "provenance": a.provenance, "bullet_commit": a.bullet_commit,
        "hparams": maybe_json(a.hparams) or {}, "final_loss": a.final_loss,
        "elo_vs_parent": a.elo_vs_parent, "nps": a.nps,
        "code_sha": a.code_sha or git_head(),
        "recipe": a.recipe, "file": rel(path), "created": today(),
        "notes": a.notes,
    }
    append(NETS, row)
    print(f"registered net {a.id}  {row['hash'][:19]}…  {size} B")


def cmd_add_data(a) -> None:
    ids = {r["id"] for r in load(DATA)}
    if a.id in ids:
        die(f"data id already registered: {a.id} (ledger is append-only)")
    path = Path(a.path)
    if not path.exists():
        die(f"corpus path not found: {path}")
    digest, size = hash_path(path, a.glob)
    net_ids = {r["id"] for r in load(NETS)}
    if a.generator_net and a.generator_net not in net_ids:
        print(f"ledger: warning: generator_net '{a.generator_net}' not in "
              f"nets.jsonl", file=sys.stderr)
    row = {
        "id": a.id, "hash": f"sha256:{digest}", "size": size,
        "source": a.source, "positions": a.positions, "format": a.format,
        "gen_recipe": maybe_json(a.gen_recipe) or {},
        "generator_net": a.generator_net, "path": rel(path),
        "glob": a.glob if path.is_dir() else None, "created": today(),
        "notes": a.notes,
    }
    append(DATA, row)
    print(f"registered corpus {a.id}  sha256:{digest[:12]}…  "
          f"{size} B  {a.positions or '?'} positions")


def cmd_add_exp(a) -> None:
    rows = load(EXPS)
    n = len(rows)
    exp_id = a.id or f"exp-{n:04d}"
    row = {
        "id": exp_id, "date": today(), "kind": a.kind, "purpose": a.purpose,
        "A": maybe_json(a.a), "B": maybe_json(a.b),
        "tc": a.tc, "book": a.book, "games": a.games,
        "wdl": maybe_json(a.wdl), "elo": a.elo, "err": a.err, "llr": a.llr,
        "hw": a.hw, "cmd": a.cmd, "notes": a.notes,
    }
    append(EXPS, row)
    verdict = f"{a.elo:+g}" if a.elo is not None else "?"
    print(f"recorded {exp_id}: {a.purpose}  ({verdict} Elo)")


def cmd_annotate(a) -> None:
    fp = {"net": NETS, "data": DATA, "exp": EXPS}[a.kind]
    if a.target_id not in {r["id"] for r in load(fp)}:
        die(f"unknown {a.kind} id: {a.target_id}")
    fields: dict = {}
    for kv in a.set or []:
        if "=" not in kv:
            die(f"--set expects key=value, got: {kv}")
        k, v = kv.split("=", 1)
        try:
            fields[k] = json.loads(v)
        except json.JSONDecodeError:
            fields[k] = v
    if "hash" in fields or "id" in fields:
        die("hash and id are the row's identity — never overlay them")
    if not a.status and not fields:
        die("nothing to annotate: give --status and/or --set key=value")
    row = {
        "id": f"ann-{len(load(ANNS)):04d}", "target_kind": a.kind,
        "target_id": a.target_id, "status": a.status, "set": fields,
        "note": a.note, "date": today(), "code_sha": git_head(),
    }
    append(ANNS, row)
    print(f"annotated {a.kind} {a.target_id}: status={a.status or '-'} "
          f"set={fields or '-'}")


def cmd_verify(a) -> None:
    problems = 0
    net_ids = {r["id"] for r in load(NETS)}
    data_ids = {r["id"] for r in load(DATA)}
    anns = ann_index()
    # hash drift + missing files
    for kind, fp in (("net", NETS), ("data", DATA)):
        for r in load_merged(fp, kind):
            status = (anns.get((kind, r["id"])) or {}).get("status")
            p = REPO / r.get("file", r.get("path", ""))
            if not p.exists():
                tag = " (annotated lost)" if status == "lost" else ""
                print(f"  MISSING  {kind} {r['id']}: {rel(p)} not on disk "
                      f"(recorded {r['hash'][:19]}…){tag}")
                continue
            glob = r.get("glob") or "*"
            digest, size = hash_path(p, glob, strict=False)
            if digest is None:
                # Shard dir exists but every matching file is gone (deleted
                # sources / broken symlinks): bytes absent, same as MISSING.
                tag = " (annotated lost)" if status == "lost" else ""
                print(f"  MISSING  {kind} {r['id']}: {rel(p)} matches no files "
                      f"for glob {glob!r} (recorded {r['hash'][:19]}…){tag}")
                continue
            if f"sha256:{digest}" != r["hash"]:
                if status == "lost":
                    # Bytes changed AND the loss is acknowledged: identity
                    # stays the recorded hash; the leftovers don't count.
                    print(f"  LOST     {kind} {r['id']}: drifted remnants on "
                          f"disk, annotated lost — not counted")
                    continue
                problems += 1
                print(f"  DRIFT    {kind} {r['id']}: on-disk sha256:{digest[:12]}… "
                      f"!= recorded {r['hash'][:19]}…")
            elif r.get("size") is not None and r["size"] != size:
                # Content hash matches, so the bytes are right and the row's
                # size field itself was recorded wrong.
                problems += 1
                print(f"  BADSIZE  {kind} {r['id']}: row says {r['size']} B, "
                      f"artifact is {size} B (hash matches)")
            else:
                print(f"  ok       {kind} {r['id']}  {r['hash'][:19]}…")
    # referential integrity — on MERGED rows (design/18 §5: annotations can
    # correct a reference, e.g. a train_data backfill typo; the raw row stays
    # append-only history).
    for r in load_merged(NETS, "net"):
        if r.get("train_data") and r["train_data"] not in data_ids:
            problems += 1
            print(f"  DANGLING net {r['id']}.train_data -> {r['train_data']}")
        if r.get("parent_net") and r["parent_net"] not in net_ids:
            problems += 1
            print(f"  DANGLING net {r['id']}.parent_net -> {r['parent_net']}")
    for r in load_merged(EXPS, "exp"):
        for side in ("A", "B"):
            s = r.get(side) or {}
            if s.get("net") and s["net"] not in net_ids:
                problems += 1
                print(f"  DANGLING {r['id']}.{side}.net -> {s['net']}")
    print(f"\nverify: {problems} problem(s)")
    sys.exit(1 if problems else 0)


def cmd_show(a) -> None:
    what = a.what
    if what == "nets":
        for r in load_merged(NETS, "net"):
            print(f"{r['id']:<10} {r['arch']:<14} {r['provenance']:<12} "
                  f"loss={r.get('final_loss','?')}  vs_parent="
                  f"{r.get('elo_vs_parent','?')}  {r['hash'][:19]}…")
    elif what == "data":
        for r in load_merged(DATA, "data"):
            print(f"{r['id']:<12} {r['source']:<12} "
                  f"{r.get('positions','?')} pos  {r['hash'][:19]}…")
    elif what in ("exps", "experiments"):
        for r in load(EXPS):
            print(f"{r['id']:<10} {r.get('date',''):<11} {r['kind']:<6} "
                  f"{str(r.get('elo','?')):>6}±{r.get('err','?')}  {r['purpose']}")
    elif what == "dag":
        _print_dag()
    else:
        die(f"unknown show target: {what}")


def _print_dag() -> None:
    nets = {r["id"]: r for r in load(NETS)}
    data = {r["id"]: r for r in load(DATA)}
    print("provenance DAG (net <- train_data <- generator_net …):\n")
    for nid, n in nets.items():
        print(f"net {nid}  [{n['provenance']}]  code={n['code_sha']}")
        d = n.get("train_data")
        seen = set()
        while d and d in data and d not in seen:
            seen.add(d)
            drow = data[d]
            gen = drow.get("generator_net")
            print(f"   └─ data {d}  ({drow['source']}, "
                  f"{drow.get('positions','?')} pos)"
                  + (f"  <- gen by net {gen}" if gen else ""))
            d = nets.get(gen, {}).get("train_data") if gen else None
        if n.get("parent_net"):
            print(f"   └─ parent net {n['parent_net']}")
        print()


def cmd_reproduce(a) -> None:
    nets = {r["id"]: r for r in load_merged(NETS, "net")}
    data = {r["id"]: r for r in load_merged(DATA, "data")}
    exps = {r["id"]: r for r in load_merged(EXPS, "exp")}
    tid = a.id
    if tid in nets:
        r = nets[tid]
        print(f"# reproduce net {tid}  ({r['hash']})")
        print(f"git checkout {r['code_sha']}")
        print(f'RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release  '
              f"# fast config (D16)")
        print(f"# train corpus: {r['train_data']} "
              f"({data.get(r['train_data'],{}).get('hash','?')})")
        print(f"# recipe: {r.get('recipe') or 'see tools/nnue/TRAINING.md'}")
    elif tid in data:
        r = data[tid]
        print(f"# reproduce corpus {tid}  ({r['hash']})")
        print(f"# source: {r['source']}")
        print(f"# gen recipe: {json.dumps(r.get('gen_recipe', {}))}")
    elif tid in exps:
        r = exps[tid]
        print(f"# reproduce experiment {tid}: {r['purpose']}")
        print(f"# A: {json.dumps(r.get('A'))}")
        print(f"# B: {json.dumps(r.get('B'))}")
        print(f"# tc={r.get('tc')} book={r.get('book')} games={r.get('games')}")
        print(r.get("cmd") or "# (harness command not recorded)")
    else:
        die(f"unknown id: {tid}")


# ---- argparse --------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ledger", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("add-net", help="register a trained net")
    n.add_argument("--id", required=True)
    n.add_argument("--file", required=True, help="path to the .cenn net")
    n.add_argument("--arch", required=True)
    n.add_argument("--act", default="screlu")
    n.add_argument("--quant", default="QA255/QB64/scale400")
    n.add_argument("--train-data", dest="train_data", help="corpus id in data.jsonl")
    n.add_argument("--parent", help="parent net id (retrain/self-play lineage)")
    n.add_argument("--provenance", required=True,
                   help="sf-distill | lc0-distill | self-play | mixed")
    n.add_argument("--bullet-commit", dest="bullet_commit")
    n.add_argument("--hparams", help="JSON, e.g. '{\"sb\":30,\"lr\":\"0.001-cos\"}'")
    n.add_argument("--final-loss", dest="final_loss", type=float)
    n.add_argument("--elo-vs-parent", dest="elo_vs_parent", type=float)
    n.add_argument("--nps", type=int,
                   help="single-thread nps of this net's eval path (D39 speed axis)")
    n.add_argument("--code-sha", dest="code_sha", help="default: git HEAD")
    n.add_argument("--recipe", help="exact regen command / pointer")
    n.add_argument("--notes")
    n.set_defaults(func=cmd_add_net)

    d = sub.add_parser("add-data", help="register a training corpus")
    d.add_argument("--id", required=True)
    d.add_argument("--path", required=True, help="file or shard directory")
    d.add_argument("--glob", default="*.txt", help="shard glob for a dir")
    d.add_argument("--source", required=True,
                   help="sf-datagen | lc0 | self-play | mixed")
    d.add_argument("--positions", type=int)
    d.add_argument("--format", default="text FEN|cp|result")
    d.add_argument("--gen-recipe", dest="gen_recipe", help="JSON recipe")
    d.add_argument("--generator-net", dest="generator_net",
                   help="net id that generated it (self-play)")
    d.add_argument("--notes")
    d.set_defaults(func=cmd_add_data)

    e = sub.add_parser("add-exp", help="record a match / SPRT / tuning result")
    e.add_argument("--id", help="default: exp-NNNN")
    e.add_argument("--kind", required=True, help="sprt | gauntlet | tactics | spsa")
    e.add_argument("--purpose", required=True)
    e.add_argument("--a", help="JSON: {code_sha, net, opts}")
    e.add_argument("--b", help="JSON: {code_sha, net, opts}")
    e.add_argument("--tc")
    e.add_argument("--book")
    e.add_argument("--games", type=int)
    e.add_argument("--wdl", help="JSON [w,d,l]")
    e.add_argument("--elo", type=float)
    e.add_argument("--err", type=float)
    e.add_argument("--llr")
    e.add_argument("--hw")
    e.add_argument("--cmd", help="exact harness invocation")
    e.add_argument("--notes")
    e.set_defaults(func=cmd_add_exp)

    an = sub.add_parser("annotate",
                        help="append a status/correction row for an existing id")
    an.add_argument("kind", choices=["net", "data", "exp"])
    an.add_argument("target_id")
    an.add_argument("--status", choices=["lost", "superseded", "corrected",
                                         "waived"])
    an.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="metadata overlay applied by verify/show/reproduce "
                         "(value parsed as JSON when possible; hash/id "
                         "forbidden)")
    an.add_argument("--note", required=True,
                    help="why — one line, mandatory")
    an.set_defaults(func=cmd_annotate)

    v = sub.add_parser("verify", help="check hashes + referential integrity")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("show", help="nets | data | exps | dag")
    s.add_argument("what")
    s.set_defaults(func=cmd_show)

    r = sub.add_parser("reproduce", help="print the recipe for an id")
    r.add_argument("id")
    r.set_defaults(func=cmd_reproduce)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
