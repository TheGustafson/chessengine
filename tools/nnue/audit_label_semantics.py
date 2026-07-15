#!/usr/bin/env python3
"""Probe C: independent label-semantics audit of the bulk training corpora.

Decodes the EXACT bytes the trainer consumed from the two bulk value corpora
(the Lichess canonical bulletformat train.bin and the official-Stockfish
fishpack32.binpack), reconstructs each sampled position, and compares the
stored score against a fresh pinned Stockfish 18 search at fixed depth
(final completed iteration, bound-carrying finals rejected). Tests BOTH
POV hypotheses (stored-vs-SF-stm and stored-vs-SF-white), reports sign
agreement, Pearson/Spearman, OLS slope, result-byte distribution and
mate/extreme handling. Governing doc: docs/research/value-foundation-state.md
(Probe C).

Format contracts (verified against the writers, not guessed):
- bulletformat ChessBoard 32B (bulletformat-1.8.0 src/chess.rs): repr(C)
  {occ u64, pcs [u8;16] nibbles, score i16, result u8, ksq u8, opp_ksq u8,
  extra [u8;3]}. Black-to-move source positions are canonicalized at write
  time: bitboards vertically flipped (sq^56), colors swapped, score negated,
  result flipped (r -> 1-r). So every record decodes to a white-to-move
  position; nibble bit 3 = opponent piece; result byte = 2*wdl from the
  mover's POV; opp_ksq is stored pre-flipped (raw ^ 56 = board square).
  Castling/ep are NOT representable -- the trainer never saw them either.
- SF binpack (sfbinpack-0.6.4): TrainingDataEntry {pos, mv, score i16 (POV of
  side to move), ply, result in {-1,0,1} from the side to move, rule50}.
  Decoded via tools/nnue/sfbinpack-audit `sample` bin (full-file chunk-header
  scan, one entry per selected chunk => many distinct games).

Ops: max 2 concurrent SF processes, nice -n 10, 60 s per-position watchdog.

Usage:
  uv run tools/nnue/audit_label_semantics.py --smoke          # 20 rows/source
  uv run tools/nnue/audit_label_semantics.py                  # full 400/source
  uv run tools/nnue/audit_label_semantics.py --source lichess --count 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import struct
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import chess
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SF = REPO / "engines/stockfish-18-src/src/stockfish"
DEFAULT_LICHESS = (
    REPO / "data/corpora/lichess-chess-evaluations/kaggle-v14/bulk-value-v1/train.bin"
)
DEFAULT_FISHPACK = (
    REPO / "data/corpora/official-stockfish-master-binpacks/main/raw/fishpack32.binpack"
)
FISHPACK_PREFIX = (
    REPO
    / "data/corpora/official-stockfish-master-binpacks/main/samples/fishpack32.complete-prefix-128m.binpack"
)
SAMPLER = REPO / "tools/nnue/sfbinpack-audit/target/release/sample"
DEFAULT_OUT = REPO / "data/nnue/audits/label-semantics-v1"

RECORD_BYTES = 32
RECORD_STRUCT = struct.Struct("<Q16shBBB3s")
PIECE_CHARS = "PNBRQK"
EXTREME_CP = 2000  # |stored| >= this is excluded from slope stats (tail bucket)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Corpus decoding
# ---------------------------------------------------------------------------


def decode_bullet_record(raw: bytes) -> dict:
    """Decode one 32-byte bulletformat ChessBoard record into a row dict.

    Performs the structural checks that pin the canonicalization contract:
    valid nibble piece codes, exactly one king per side matching ksq /
    opp_ksq^56, no pawns on back ranks, zero padding bytes.
    """
    occ, pcs, score, result, ksq, opp_ksq_raw, extra = RECORD_STRUCT.unpack(raw)
    problems = []
    board = chess.Board.empty()
    white_kings, black_kings = [], []
    pawn_backrank = 0
    idx = 0
    bits = occ
    while bits:
        sq = (bits & -bits).bit_length() - 1
        bits &= bits - 1
        nibble = (pcs[idx >> 1] >> (4 * (idx & 1))) & 0xF
        idx += 1
        pt = nibble & 7
        theirs = bool(nibble & 8)
        if pt > 5:
            problems.append(f"invalid piece nibble {nibble} at sq {sq}")
            continue
        color = chess.BLACK if theirs else chess.WHITE
        board.set_piece_at(sq, chess.Piece(pt + 1, color))
        if pt == 5:
            (black_kings if theirs else white_kings).append(sq)
        if pt == 0 and (sq < 8 or sq >= 56):
            pawn_backrank += 1
    if white_kings != [ksq]:
        problems.append(f"mover king mismatch: field {ksq}, board {white_kings}")
    if black_kings != [opp_ksq_raw ^ 56]:
        problems.append(
            f"opp king mismatch: field {opp_ksq_raw}^56={opp_ksq_raw ^ 56}, "
            f"board {black_kings}"
        )
    if pawn_backrank:
        problems.append(f"{pawn_backrank} pawns on rank 1/8")
    if extra != b"\x00\x00\x00":
        problems.append(f"nonzero padding bytes {extra.hex()}")
    if result not in (0, 1, 2):
        problems.append(f"result byte {result} not in 0/1/2")
    # Canonical frame: mover plays as White; castling/ep unrepresentable.
    board.turn = chess.WHITE
    status = board.status()
    if status != chess.STATUS_VALID:
        problems.append(f"python-chess status {status!r}")
    return {
        "fen": board.fen(),
        "stored_score": score,
        "stored_result": result,
        "stm": "w",  # canonical frame is always mover-as-white
        "ksq": ksq,
        "opp_ksq_raw": opp_ksq_raw,
        "structural_problems": problems,
    }


def sample_lichess(path: Path, count: int, seed: int) -> tuple[list[dict], dict]:
    size = path.stat().st_size
    if size % RECORD_BYTES:
        raise SystemExit(f"{path}: size {size} not divisible by {RECORD_BYTES}")
    n_records = size // RECORD_BYTES
    stride = n_records // count
    offset0 = seed % stride
    indices = [offset0 + i * stride for i in range(count)]
    rows = []
    with path.open("rb") as stream:
        for i, index in enumerate(indices):
            stream.seek(index * RECORD_BYTES)
            row = decode_bullet_record(stream.read(RECORD_BYTES))
            row["sample_index"] = i
            row["record_index"] = index
            rows.append(row)
    meta = {
        "file": str(path),
        "bytes": size,
        "records": n_records,
        "sampling": {
            "method": "evenly spaced record indices: offset0 + i*stride",
            "count": count,
            "seed": seed,
            "stride": stride,
            "offset0": offset0,
            "first_index": indices[0],
            "last_index": indices[-1],
        },
    }
    return rows, meta


def sample_fishpack(path: Path, count: int, seed: int) -> tuple[list[dict], dict]:
    if not SAMPLER.exists():
        raise SystemExit(
            f"sampler binary missing: {SAMPLER}\n"
            "build it: cd tools/nnue/sfbinpack-audit && cargo build --release --bin sample"
        )
    proc = subprocess.run(
        ["nice", "-n", "10", str(SAMPLER), str(path), str(count), str(seed)],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    meta_line = next(line for line in lines if line["type"] == "meta")
    rows = []
    for line in lines:
        if line["type"] != "row":
            continue
        board = chess.Board(line["fen"])
        problems = []
        status = board.status()
        if status != chess.STATUS_VALID:
            problems.append(f"python-chess status {status!r}")
        if line["result_stm"] not in (-1, 0, 1):
            problems.append(f"result {line['result_stm']} not in -1/0/1")
        rows.append(
            {
                "sample_index": line["sample_index"],
                "chunk_index": line["chunk_index"],
                "chunk_offset": line["chunk_offset"],
                "entries_in_chunk": line["entries_in_chunk"],
                "entry_index": line["entry_index"],
                "fen": line["fen"],
                "move_uci": line["move_uci"],
                "stored_score": line["score_stm"],
                "stored_result": line["result_stm"],
                "ply": line["ply"],
                "rule50": line["rule50"],
                "stm": line["stm"],
                "in_check": line["in_check"],
                "structural_problems": problems,
            }
        )
    meta = {
        "file": str(path),
        "bytes": meta_line["bytes"],
        "complete_chunks": meta_line["complete_chunks"],
        "tail_note": meta_line["tail_note"],
        "sampling": {
            "method": meta_line["selection"],
            "count": count,
            "seed": seed,
            "sampler": str(SAMPLER),
        },
    }
    return rows, meta


# ---------------------------------------------------------------------------
# Stockfish oracle (hand-rolled UCI driver: exact control over final-info
# parsing and the per-position watchdog; python-chess is used only for board
# reconstruction/validation above)
# ---------------------------------------------------------------------------


class EngineTimeout(Exception):
    pass


class Engine:
    def __init__(self, path: Path, hash_mb: int = 16):
        self.path = path
        self.proc = subprocess.Popen(
            ["nice", "-n", "10", str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.id_name = None
        self._send("uci")
        for line in self._read_until(lambda l: l == "uciok", 30):
            if line.startswith("id name "):
                self.id_name = line[len("id name ") :]
        self._send("setoption name Threads value 1")
        self._send(f"setoption name Hash value {hash_mb}")
        self._sync(30)

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def _send(self, command: str):
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _read_until(self, predicate, timeout: float) -> list[str]:
        deadline = time.monotonic() + timeout
        collected = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineTimeout(f"timeout waiting; got {len(collected)} lines")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                raise EngineTimeout(f"timeout waiting; got {len(collected)} lines")
            if line is None:
                raise EngineTimeout("engine stdout closed")
            collected.append(line)
            if predicate(line):
                return collected

    def _sync(self, timeout: float):
        self._send("isready")
        self._read_until(lambda l: l == "readyok", timeout)

    def analyse(self, fen: str, depth: int, timeout: float) -> dict:
        """`ucinewgame` (clears hash) + `position fen` + `go depth N`.

        Returns the FINAL completed info record at the target depth
        (bound-carrying finals flagged for rejection), plus node counts.
        """
        self._send("ucinewgame")
        self._sync(timeout)
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        start = time.monotonic()
        lines = self._read_until(lambda l: l.startswith("bestmove"), timeout)
        elapsed = time.monotonic() - start
        infos = [parse_info(l) for l in lines if l.startswith("info ")]
        scored = [i for i in infos if i and i["score_kind"]]
        finals = [i for i in scored if i["depth"] == depth]
        best = lines[-1].split()
        result = {
            "bestmove": best[1] if len(best) > 1 else None,
            "elapsed_s": round(elapsed, 3),
            "reached_depth": max((i["depth"] for i in scored), default=None),
        }
        if finals:
            final = finals[-1]
            result.update(
                sf_depth=final["depth"],
                sf_score_kind=final["score_kind"],
                sf_score=final["score_value"],
                sf_nodes=final["nodes"],
                sf_bound=final["bound"],
            )
        elif scored:
            # e.g. terminal positions: "info depth 0 score mate 0"
            final = scored[-1]
            result.update(
                sf_depth=final["depth"],
                sf_score_kind=final["score_kind"],
                sf_score=final["score_value"],
                sf_nodes=final["nodes"],
                sf_bound=final["bound"],
                incomplete_depth=True,
            )
        else:
            result.update(sf_depth=None, sf_score_kind=None, sf_score=None,
                          sf_nodes=None, sf_bound=None)
        return result

    def kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass


def parse_info(line: str) -> dict | None:
    tokens = line.split()
    info = {
        "depth": None,
        "score_kind": None,
        "score_value": None,
        "nodes": None,
        "bound": None,
        "multipv": 1,
    }
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "depth":
            info["depth"] = int(tokens[i + 1]); i += 2
        elif token == "multipv":
            info["multipv"] = int(tokens[i + 1]); i += 2
        elif token == "nodes":
            info["nodes"] = int(tokens[i + 1]); i += 2
        elif token == "score":
            info["score_kind"] = tokens[i + 1]
            info["score_value"] = int(tokens[i + 2])
            i += 3
            if i < len(tokens) and tokens[i] in ("lowerbound", "upperbound"):
                info["bound"] = tokens[i]; i += 1
        elif token == "pv":
            break
        else:
            i += 1
    if info["depth"] is None:
        return None
    return info if info["multipv"] == 1 else None


def run_oracle(rows: list[dict], sf_path: Path, depth: int, workers: int,
               timeout: float, label: str):
    """Fill each row in place with fresh-SF fields. Max `workers` engines."""
    tasks: queue.Queue[int] = queue.Queue()
    for i in range(len(rows)):
        tasks.put(i)
    done = [0]
    lock = threading.Lock()
    id_names = []

    def worker():
        engine = Engine(sf_path)
        with lock:
            id_names.append(engine.id_name)
        while True:
            try:
                i = tasks.get_nowait()
            except queue.Empty:
                break
            row = rows[i]
            skip = bool(row["structural_problems"])
            if skip:
                row["sf"] = {"skipped": "structural problems"}
            else:
                try:
                    row["sf"] = engine.analyse(row["fen"], depth, timeout)
                except EngineTimeout as error:
                    row["sf"] = {"timeout": True, "detail": str(error)}
                    engine.kill()
                    engine = Engine(sf_path)
            with lock:
                done[0] += 1
                if done[0] % 25 == 0 or done[0] == len(rows):
                    print(f"[{label}] {done[0]}/{len(rows)} positions", flush=True)
        engine.kill()

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return id_names


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS of y on x: y ~ slope*x + intercept."""
    if len(x) < 3 or x.std() == 0:
        return {"slope": None, "intercept": None, "r2": None, "n": int(len(x))}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else None,
        "n": int(len(x)),
    }


def sign_agreement(stored: np.ndarray, sf: np.ndarray) -> dict:
    both = (stored != 0) & (sf != 0)
    agree = int(np.sum(np.sign(stored[both]) == np.sign(sf[both])))
    return {
        "n_both_nonzero": int(both.sum()),
        "n_agree": agree,
        "agreement": agree / both.sum() if both.sum() else None,
        "stored_zero": int(np.sum(stored == 0)),
        "sf_zero": int(np.sum(sf == 0)),
    }


def compute_stats(rows: list[dict]) -> dict:
    n = len(rows)
    structural_bad = [r["sample_index"] for r in rows if r["structural_problems"]]
    timeouts = [r["sample_index"] for r in rows if r.get("sf", {}).get("timeout")]
    bound_rejected = [
        r["sample_index"]
        for r in rows
        if r.get("sf", {}).get("sf_bound") and r["sf"].get("sf_depth") is not None
    ]
    incomplete = [
        r["sample_index"] for r in rows if r.get("sf", {}).get("incomplete_depth")
    ]
    mate_rows = [
        r for r in rows
        if r.get("sf", {}).get("sf_score_kind") == "mate"
    ]
    usable = [
        r for r in rows
        if r.get("sf", {}).get("sf_score_kind") == "cp"
        and not r["sf"].get("sf_bound")
        and not r["sf"].get("incomplete_depth")
        and not r["structural_problems"]
    ]

    stored = np.array([r["stored_score"] for r in usable], dtype=np.float64)
    sf_stm = np.array([r["sf"]["sf_score"] for r in usable], dtype=np.float64)
    white_sign = np.array([1.0 if r["stm"] == "w" else -1.0 for r in usable])
    sf_white = sf_stm * white_sign
    nodes = np.array(
        [r["sf"]["sf_nodes"] or 0 for r in usable], dtype=np.float64
    )

    def pov_block(sf: np.ndarray) -> dict:
        core = np.abs(stored) < EXTREME_CP
        return {
            "pearson_all": pearson(stored, sf),
            "spearman_all": spearman(stored, sf),
            "sign": sign_agreement(stored, sf),
            "core_abs_stored_lt_2000": {
                "pearson": pearson(stored[core], sf[core]),
                "spearman": spearman(stored[core], sf[core]),
                "ols_stored_on_sf": ols(sf[core], stored[core]),
                "ols_sf_on_stored": ols(stored[core], sf[core]),
                "sign": sign_agreement(stored[core], sf[core]),
            },
            "tail_abs_stored_ge_2000": {
                "n": int((~core).sum()),
                "sign": sign_agreement(stored[~core], sf[~core])
                if (~core).sum()
                else None,
                "rows": [
                    {
                        "sample_index": r["sample_index"],
                        "stored": r["stored_score"],
                        "sf_stm": r["sf"]["sf_score"],
                    }
                    for r, is_tail in zip(usable, ~core)
                    if is_tail
                ],
            },
        }

    results, counts = np.unique(
        np.array([r["stored_result"] for r in rows]), return_counts=True
    )
    result_dist = {int(k): int(v) for k, v in zip(results, counts)}
    result_means = {}
    for bucket in result_dist:
        sel = [
            r for r in usable if r["stored_result"] == bucket
        ]
        if sel:
            result_means[bucket] = {
                "n": len(sel),
                "mean_stored": float(np.mean([r["stored_score"] for r in sel])),
                "mean_sf_stm": float(np.mean([r["sf"]["sf_score"] for r in sel])),
            }

    stm_counts = {}
    for r in rows:
        stm_counts[r["stm"]] = stm_counts.get(r["stm"], 0) + 1

    return {
        "n_sampled": n,
        "n_usable_cp": len(usable),
        "structural_problem_rows": structural_bad,
        "timeouts": timeouts,
        "bound_rejected_finals": bound_rejected,
        "incomplete_depth_rows": incomplete,
        "sf_mate_rows": [
            {
                "sample_index": r["sample_index"],
                "fen": r["fen"],
                "stored": r["stored_score"],
                "sf_mate": r["sf"]["sf_score"],
            }
            for r in mate_rows
        ],
        "stored_score": {
            "min": int(min((r["stored_score"] for r in rows), default=0)),
            "max": int(max((r["stored_score"] for r in rows), default=0)),
            "mean": float(np.mean([r["stored_score"] for r in rows])) if rows else None,
            "std": float(np.std([r["stored_score"] for r in rows])) if rows else None,
            "n_abs_ge_2000": int(sum(abs(r["stored_score"]) >= EXTREME_CP for r in rows)),
        },
        "stm_counts": stm_counts,
        "result_byte_distribution": result_dist,
        "result_bucket_means_usable": result_means,
        "sf_nodes": {
            "min": int(nodes.min()) if len(nodes) else None,
            "median": float(np.median(nodes)) if len(nodes) else None,
            "max": int(nodes.max()) if len(nodes) else None,
        },
        "pov_stm": pov_block(sf_stm),
        "pov_white": pov_block(sf_white),
    }


def verdict_for(stats: dict, expected_pov: str) -> tuple[str, list[str]]:
    notes = []
    match_block = stats[f"pov_{expected_pov}"]
    pearson_core = match_block["core_abs_stored_lt_2000"]["pearson"]
    sign = match_block["core_abs_stored_lt_2000"]["sign"]["agreement"]
    problems = []
    if stats["structural_problem_rows"]:
        problems.append(
            f"{len(stats['structural_problem_rows'])} rows failed structural decode checks"
        )
    if pearson_core is None or math.isnan(pearson_core) or pearson_core < 0.8:
        problems.append(
            f"core Pearson vs SF ({expected_pov}-POV) = {pearson_core} < 0.8"
        )
    if sign is None or sign < 0.85:
        problems.append(f"core sign agreement = {sign} < 0.85")
    other = "white" if expected_pov == "stm" else "stm"
    other_pearson = stats[f"pov_{other}"]["core_abs_stored_lt_2000"]["pearson"]
    if (
        other_pearson is not None
        and not math.isnan(other_pearson)
        and not math.isnan(pearson_core)
        and other_pearson > pearson_core + 0.05
    ):
        problems.append(
            f"opposite POV hypothesis fits BETTER ({other}: {other_pearson:.3f} "
            f"vs {expected_pov}: {pearson_core:.3f}) -- perspective defect"
        )
    if problems:
        return "DEFECT", problems
    notes.append(
        f"core Pearson {pearson_core:.4f}, sign agreement {sign:.4f} under the "
        f"expected {expected_pov}-POV hypothesis"
    )
    return "PASS", notes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_outputs(out_dir: Path, name: str, payload: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    stats = payload["stats"]
    verdict, notes = payload["verdict"], payload["verdict_notes"]
    pov = payload["expected_pov"]
    block = stats[f"pov_{pov}"]
    core = block["core_abs_stored_lt_2000"]
    md = [
        f"# Label-semantics audit: {payload['source']}",
        "",
        f"- Date: {payload['generated_utc']}",
        f"- Corpus: `{payload['corpus']['file']}` ({payload['corpus']['bytes']} bytes)",
        f"- Corpus sha256: {payload['corpus'].get('sha256', '(not recomputed)')}",
        f"- Oracle: `{payload['oracle']['binary']}` (sha256 {payload['oracle']['sha256'][:16]}...), "
        f"`go depth {payload['oracle']['depth']}`, Threads=1, hash cleared per position",
        f"- Sampled: {stats['n_sampled']} rows, usable cp rows {stats['n_usable_cp']}",
        "",
        f"## VERDICT: {verdict}",
        "",
        *[f"- {n}" for n in notes],
        "",
        "## Headline stats (expected POV hypothesis: "
        f"stored is {pov}-relative)",
        "",
        f"- Pearson (all usable): {block['pearson_all']:.4f}; "
        f"Spearman: {block['spearman_all']:.4f}",
        f"- Core (|stored| < {EXTREME_CP}): Pearson {core['pearson']:.4f}, "
        f"Spearman {core['spearman']:.4f}",
        f"- OLS stored = a*SF + b: slope {core['ols_stored_on_sf']['slope']:.4f}, "
        f"intercept {core['ols_stored_on_sf']['intercept']:.1f}, "
        f"R^2 {core['ols_stored_on_sf']['r2']:.4f} (n={core['ols_stored_on_sf']['n']})",
        f"- Sign agreement (both nonzero): {core['sign']['agreement']:.4f} "
        f"({core['sign']['n_agree']}/{core['sign']['n_both_nonzero']})",
        f"- Opposite-POV Pearson (core): "
        f"{stats['pov_' + ('white' if pov == 'stm' else 'stm')]['core_abs_stored_lt_2000']['pearson']:.4f}",
        "",
        "## Distributions / edge cases",
        "",
        f"- stm counts: {stats['stm_counts']}",
        f"- result byte distribution: {stats['result_byte_distribution']}",
        f"- result-bucket means (usable rows): {json.dumps(stats['result_bucket_means_usable'])}",
        f"- stored score: min {stats['stored_score']['min']}, max {stats['stored_score']['max']}, "
        f"mean {stats['stored_score']['mean']:.1f}, std {stats['stored_score']['std']:.1f}, "
        f"|s|>={EXTREME_CP}: {stats['stored_score']['n_abs_ge_2000']}",
        f"- SF mate rows: {len(stats['sf_mate_rows'])}; bound-rejected finals: "
        f"{len(stats['bound_rejected_finals'])}; timeouts: {len(stats['timeouts'])}; "
        f"structural problems: {len(stats['structural_problem_rows'])}",
        f"- SF nodes at depth {payload['oracle']['depth']}: min {stats['sf_nodes']['min']}, "
        f"median {stats['sf_nodes']['median']}, max {stats['sf_nodes']['max']}",
        "",
        f"Full row-level data: `{json_path.name}`.",
        "",
    ]
    (out_dir / f"{name}-summary.md").write_text("\n".join(md))
    return json_path


def audit_source(source: str, args, sf_sha: str) -> dict:
    seed = args.seed
    if source == "lichess":
        rows, corpus_meta = sample_lichess(args.lichess, args.count, seed)
        expected_pov = "stm"  # canonical frame: stm == white by construction
        pov_note = (
            "bulletformat canonicalizes to mover-as-white; every reconstructed "
            "position is white-to-move, so stm-POV and white-POV coincide on "
            "this corpus by construction. The POV test therefore verifies that "
            "black-to-move source rows were flipped+negated correctly (a "
            "missed negation would destroy the correlation)."
        )
    else:
        rows, corpus_meta = sample_fishpack(args.fishpack, args.count, seed)
        expected_pov = "stm"
        pov_note = (
            "fishpack has both colors to move (~50/50), so stored-vs-SF-stm and "
            "stored-vs-SF-white are genuinely distinguishable here."
        )
    if args.hash_corpora:
        corpus_meta["sha256"] = sha256_file(Path(corpus_meta["file"]))
    print(f"[{source}] decoded {len(rows)} rows; launching SF18 oracle "
          f"({args.workers} workers, depth {args.depth})", flush=True)
    id_names = run_oracle(rows, args.sf, args.depth, args.workers,
                          args.timeout, source)
    stats = compute_stats(rows)
    verdict, notes = verdict_for(stats, expected_pov)
    payload = {
        "schema": "archon.label-semantics-audit.v1",
        "source": source,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "governing_doc": "docs/research/value-foundation-state.md (Probe C)",
        "corpus": corpus_meta,
        "oracle": {
            "binary": str(args.sf),
            "sha256": sf_sha,
            "uci_id_names": sorted(set(filter(None, id_names))),
            "depth": args.depth,
            "threads": 1,
            "hash_mb": 16,
            "hash_cleared_per_position": "ucinewgame before every search",
            "final_info_rule": (
                "last multipv-1 info record at exactly the target depth; rows "
                "whose final record carries lowerbound/upperbound are rejected "
                "from stats and counted"
            ),
            "timeout_s": args.timeout,
            "nice": 10,
            "max_workers": args.workers,
        },
        "expected_pov": expected_pov,
        "pov_note": pov_note,
        "stats": stats,
        "verdict": verdict,
        "verdict_notes": notes,
        "rows": rows,
    }
    name = f"{args.prefix}{source}"
    json_path = write_outputs(args.out_dir, name, payload)
    print(f"[{source}] VERDICT: {verdict} -> {json_path}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["lichess", "fishpack", "both"],
                        default="both")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2,
                        help="max concurrent SF processes (ops cap: 2)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sf", type=Path, default=DEFAULT_SF)
    parser.add_argument("--lichess", type=Path, default=DEFAULT_LICHESS)
    parser.add_argument("--fishpack", type=Path, default=DEFAULT_FISHPACK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true",
                        help="20 rows/source, smoke- output prefix, no corpus hash")
    parser.add_argument("--hash-corpora", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()
    if args.workers > 2:
        parser.error("ops cap: max 2 concurrent SF processes")
    if args.smoke:
        args.count = min(args.count, 20)
        args.prefix = "smoke-"
        args.hash_corpora = False
    else:
        args.prefix = ""
    if not args.sf.exists():
        parser.error(f"SF binary not found: {args.sf}")
    sf_sha = sha256_file(args.sf)
    sources = ["lichess", "fishpack"] if args.source == "both" else [args.source]
    for source in sources:
        audit_source(source, args, sf_sha)


if __name__ == "__main__":
    main()
