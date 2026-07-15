#!/usr/bin/env python3
"""Resumable 4x3 D40 policy campaign plus Bullet reference and pair gauntlets."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def fail(message):
    raise SystemExit(f"campaign: {message}")


def load(path):
    cfg = json.loads(Path(path).read_text())
    cfg["_path"] = str(Path(path).resolve())
    return cfg


class Campaign:
    def __init__(self, cfg, *, smoke=False, dry_run=False):
        self.cfg = cfg
        self.smoke = smoke
        self.dry_run = dry_run
        suffix = "-smoke" if smoke else ""
        self.root = REPO / "data/nnue/campaigns" / f"{cfg['campaign_id']}{suffix}"
        self.logs = self.root / "logs"
        self.artifacts = self.root / "artifacts"
        self.matches = self.root / "gauntlet"
        self.bin = self.root / "chessengine-policy"
        self.state_path = self.root / "state.json"
        self.state = {"campaign_id": cfg["campaign_id"], "smoke": smoke, "jobs": {}, "matches": {}}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        for directory in (self.root, self.logs, self.artifacts, self.matches):
            directory.mkdir(parents=True, exist_ok=True)

    def save(self):
        self.state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n")

    def run(self, command, *, env=None, log=None, capture=False):
        print("$", " ".join(shlex.quote(str(value)) for value in command), flush=True)
        if self.dry_run:
            return ""
        merged = {**os.environ, **{key: str(value) for key, value in (env or {}).items()}}
        if capture:
            result = subprocess.run(command, cwd=REPO, env=merged, text=True, capture_output=True)
            if result.returncode:
                sys.stderr.write(result.stdout + result.stderr)
                fail(f"command failed ({result.returncode})")
            return result.stdout
        stream = open(log, "a", buffering=1) if log else None
        try:
            result = subprocess.run(command, cwd=REPO, env=merged, stdout=stream, stderr=subprocess.STDOUT)
        finally:
            if stream:
                stream.close()
        if result.returncode:
            fail(f"command failed ({result.returncode}); see {log}")
        return ""

    @property
    def bullet(self):
        return Path(self.cfg["bullet_dir"])

    @property
    def trainer_bin(self):
        return self.bullet / "target/release/examples/d40_policy"

    @property
    def ref_trainer_bin(self):
        return self.bullet / "target/release/examples/d40_features"

    def preflight(self):
        required = [
            REPO / self.cfg["dual_corpus"] / "white/feats.npy",
            REPO / self.cfg["dual_corpus"] / "black/feats.npy",
            REPO / self.cfg["bullet_corpus"],
            self.bullet / "crates/bullet_lib/Cargo.toml",
            REPO / self.cfg["gauntlet"]["book"],
            REPO / "engines/bin/fastchess",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            fail("missing prerequisites:\n  " + "\n  ".join(missing))
        free = shutil.disk_usage(self.root).free
        if free < 4_000_000_000:
            fail(f"only {free / 1e9:.1f} GB free; require at least 4 GB")
        for ref in self.cfg["gauntlet"]["references"]:
            engine = self.bin if ref["engine"] == "campaign" else REPO / ref["engine"]
            if ref["engine"] != "campaign" and not os.access(engine, os.X_OK):
                fail(f"reference engine is missing/not executable: {engine}")
            if ref.get("net") and not (REPO / ref["net"]).exists():
                fail(f"reference net is missing: {ref['net']}")
        print(f"preflight: inputs present, {free / 1e9:.1f} GB free")

    def build(self):
        source = REPO / "tools/nnue/bullet/d40_policy.rs"
        ref_source = REPO / "tools/nnue/bullet/d40_features.rs"
        examples = self.bullet / "examples"
        if not self.dry_run:
            shutil.copy2(source, examples / "d40_policy.rs")
            shutil.copy2(ref_source, examples / "d40_features.rs")
        manifest = self.bullet / "crates/bullet_lib/Cargo.toml"
        self.run(["cargo", "build", "--release", "--features", "cuda", "--manifest-path", manifest,
                  "--example", "d40_policy", "--example", "d40_features"], env={"CUDA_PATH": "/usr"},
                 log=self.logs / "build-trainers.log")
        self.run(["cargo", "build", "--release", "--features", "policy"],
                 env={"RUSTFLAGS": "-C target-cpu=x86-64-v3"}, log=self.logs / "build-engine.log")
        if not self.dry_run:
            shutil.copy2(REPO / "target/release/chessengine", self.bin)

    def jobs(self):
        for replicate in self.cfg["replicates"]:
            for klass in self.cfg["classes"]:
                yield {
                    "id": f"{self.cfg['campaign_id']}-{replicate['id']}-{klass['id']}",
                    "seed": replicate["seed"],
                    "class": klass["id"],
                    "head": klass["head"],
                    "lambda": klass["policy_lambda"],
                    "reference": False,
                }
        ref = self.cfg["bullet_reference"]
        yield {"id": f"{self.cfg['campaign_id']}-{ref['id']}", "seed": ref["seed"],
               "class": "eval", "head": "eval", "lambda": 0.0, "reference": True}

    def paths(self, job):
        stem = self.artifacts / job["id"]
        return stem.with_suffix(".quant.bin"), stem.with_suffix(".cenn"), stem.with_suffix(".cpol")

    def train_job(self, job):
        quant, cenn, cpol = self.paths(job)
        record = self.state["jobs"].setdefault(job["id"], {})
        if record.get("status") == "complete" and cenn.exists() and (job["head"] == "eval" or cpol.exists()):
            print(f"skip complete job {job['id']}")
            return
        record.update({"status": "running", "started": time.time(), **job})
        self.save()
        if job["reference"]:
            spec = self.cfg["bullet_reference"]
            sb, bps, batch = spec["superbatches"], spec["batches_per_superbatch"], spec["batch"]
            if self.smoke: sb, bps, batch = 1, 1, 256
            env = {"D40_DATA": REPO / self.cfg["bullet_corpus"], "D40_OUT": self.artifacts,
                   "D40_ID": job["id"], "D40_SEED": job["seed"], "D40_SUPERBATCHES": sb,
                   "D40_BPS": bps, "D40_BATCH": batch, "D40_SAVE_RATE": sb}
            self.run([self.ref_trainer_bin], env=env, log=self.logs / f"{job['id']}.log")
            produced = self.artifacts / f"{job['id']}-sb{sb}-quant.bin"
        else:
            spec = self.cfg["train"]
            sb, bps, batch = spec["superbatches"], spec["batches_per_superbatch"], spec["batch"]
            if self.smoke: sb, bps, batch = 1, 1, 256
            env = {"D40_POLICY_DATA": REPO / self.cfg["dual_corpus"], "D40_POLICY_HEAD": job["head"],
                   "D40_POLICY_LAMBDA": job["lambda"], "D40_BATCH": batch, "D40_BPS": bps,
                   "D40_SUPERBATCHES": sb, "D40_THREADS": spec["threads"], "D40_LR": spec["learning_rate"],
                   "D40_OUT": self.artifacts, "D40_ID": job["id"], "D40_SEED": job["seed"]}
            self.run([self.trainer_bin], env=env, log=self.logs / f"{job['id']}.log")
            produced = self.artifacts / f"{job['id']}-{job['head']}-quant.bin"
        if not self.dry_run:
            produced.replace(quant)
        self.run(["uv", "run", "--no-project", "python", "tools/nnue/netpack.py", quant, cenn, "d40"],
                 log=self.logs / f"{job['id']}-pack.log")
        if job["head"] != "eval":
            self.run(["uv", "run", "--no-project", "python", "tools/nnue/policy_pack.py",
                      quant, cpol, job["head"]], log=self.logs / f"{job['id']}-policy-pack.log")
        self.sanity(job, cenn, cpol)
        record.update({"status": "complete", "completed": time.time(), "quant": str(quant),
                       "cenn": str(cenn), "cpol": str(cpol) if cpol.exists() else None})
        self.save()

    def sanity(self, job, cenn, cpol):
        fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "4k3/8/8/8/8/8/8/QQQQK3 w - - 0 1",
            "qqqqk3/8/8/8/8/8/8/4K3 w - - 0 1",
        ]
        print("$", self.bin, "nnue-eval", cenn, "< 3 sanity FENs", flush=True)
        if self.dry_run:
            out = ""
        else:
            result = subprocess.run([self.bin, "nnue-eval", cenn], input="\n".join(fens) + "\n",
                                    text=True, capture_output=True, cwd=REPO)
            if result.returncode:
                sys.stderr.write(result.stdout + result.stderr)
                fail(f"{job['id']}: nnue-eval sanity command failed")
            out = result.stdout
        if not self.dry_run:
            scores = [int(value) for value in out.split()]
            # Smoke models are random after one update: only bound them. Full
            # semantic sign gates apply to production artifacts.
            if any(abs(value) > 100_000 for value in scores):
                fail(f"{job['id']}: non-finite/saturated sanity scores {scores}")
            if not self.smoke and not (abs(scores[0]) <= 250 and scores[1] >= 500 and scores[2] <= -500):
                fail(f"{job['id']}: semantic sanity failed {scores}")
        if job["head"] != "eval":
            probe = ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                     "|e2e4 d2d4 g1f3\n")
            oracle = subprocess.run([self.bin, "policy-eval", cenn, cpol], input=probe, text=True,
                                    capture_output=True, cwd=REPO)
            try:
                values = [float(value) for value in oracle.stdout.split()]
            except ValueError:
                values = []
            if oracle.returncode or len(values) != 3 or not all(abs(value) < 1e6 for value in values):
                fail(f"{job['id']}: explicit policy oracle failed: {oracle.stdout}{oracle.stderr}")
            env = {"POLICY_NET": cpol, "POLICY_W": self.cfg["gauntlet"]["policy_weight"]}
            commands = (f"uci\nsetoption name EvalFile value {cenn}\nisready\nposition startpos\n"
                        "go depth 1\nquit\n")
            result = subprocess.run([self.bin], input=commands, text=True, capture_output=True,
                                    cwd=REPO, env={**os.environ, **{k: str(v) for k, v in env.items()}})
            if result.returncode or "bestmove " not in result.stdout:
                fail(f"{job['id']}: policy load/play smoke failed: {result.stderr}")

    def train_all(self):
        for job in self.jobs():
            self.train_job(job)

    def engine(self, entry):
        if "job" in entry:
            job = entry["job"]
            _, cenn, cpol = self.paths(job)
            command = self.bin
            if job["head"] != "eval":
                wrapper = self.root / "wrappers" / job["id"]
                wrapper.parent.mkdir(exist_ok=True)
                if not self.dry_run:
                    wrapper.write_text("#!/usr/bin/env bash\n" +
                        f"export POLICY_NET={shlex.quote(str(cpol))}\n" +
                        f"export POLICY_W={int(self.cfg['gauntlet']['policy_weight'])}\n" +
                        f"exec {shlex.quote(str(self.bin))} \"$@\"\n")
                    wrapper.chmod(0o755)
                command = wrapper
            return {"id": job["id"], "command": command, "net": cenn}
        ref = entry["ref"]
        command = self.bin if ref["engine"] == "campaign" else REPO / ref["engine"]
        return {"id": ref["id"], "command": command,
                "net": REPO / ref["net"] if ref.get("net") else None}

    def play_pair(self, left, right, phase):
        tag = f"{phase}--{left['id']}--vs--{right['id']}"
        record = self.state["matches"].setdefault(tag, {})
        pgn = self.matches / f"{tag}.pgn"
        if record.get("status") == "complete" and pgn.exists():
            print(f"skip complete match {tag}")
            return
        # Fastchess appends to an existing PGN. A killed/interrupted match left
        # as `running` must restart from a clean file or the aggregate silently
        # double-counts its completed prefix while state claims exactly N games.
        if pgn.exists() and record.get("status") != "complete":
            print(f"restart incomplete match {tag}: deleting partial PGN", flush=True)
            if not self.dry_run:
                pgn.unlink()
        log_path = self.logs / f"{tag}.log"
        if log_path.exists() and record.get("status") != "complete" and not self.dry_run:
            log_path.unlink()
        g = self.cfg["gauntlet"]
        rounds = 1 if self.smoke else g["rounds_per_pair"]
        nodes = 64 if self.smoke else g["nodes"]
        command = [REPO / "engines/bin/fastchess"]
        for side in (left, right):
            command += ["-engine", f"cmd={side['command']}", f"name={side['id']}"]
            if side.get("net"):
                command += [f"option.EvalFile={side['net']}"]
        command += ["-each", "proto=uci", "tc=inf", f"nodes={nodes}", "option.Threads=1", "option.Hash=16",
                    "-openings", f"file={REPO / g['book']}", "format=epd", "order=random", "-repeat",
                    "-games", "2", "-rounds", str(rounds), "-concurrency", str(g["concurrency"]), "-recover",
                    "-draw", "movenumber=40", "movecount=8", "score=15", "-resign", "movecount=4", "score=700",
                    "-maxmoves", "250", "-srand", "20260714", "-pgnout", f"file={pgn}", "notation=lan"]
        record.update({"status": "running", "started": time.time(), "phase": phase})
        self.save()
        self.run(command, log=log_path)
        record.update({"status": "complete", "completed": time.time(), "pgn": str(pgn),
                       "games": 2 * rounds})
        self.save()

    def gauntlet(self):
        candidates = [self.engine({"job": job}) for job in self.jobs()]
        references = [self.engine({"ref": ref}) for ref in self.cfg["gauntlet"]["references"]]
        if self.cfg["gauntlet"].get("candidate_round_robin", True):
            for left, right in combinations(candidates, 2):
                self.play_pair(left, right, "candidates")
        for candidate in candidates:
            for reference in references:
                self.play_pair(candidate, reference, "references")
        pgns = sorted(self.matches.glob("*.pgn"))
        combined = self.matches / "all-games.pgn"
        if not self.dry_run:
            with combined.open("wb") as output:
                for pgn in pgns:
                    output.write(pgn.read_bytes())
        ordo = REPO / "engines/bin/ordo"
        if pgns and ordo.exists():
            # No arbitrary anchor: the table is relative, with d40-c2 fixed at 0.
            self.run([ordo, "-q", "-a", "0", "-A", "d40-c2", "-W", "-G", "-s100",
                      "-p", combined, "-o", self.matches / "ordo.txt"], log=self.logs / "ordo.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tools/nnue/configs/d40-policy-campaign.json")
    parser.add_argument("--stage", choices=("all", "train", "gauntlet"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    campaign = Campaign(load(args.config), smoke=args.smoke, dry_run=args.dry_run)
    campaign.preflight()
    campaign.build()
    if args.stage in ("all", "train"):
        campaign.train_all()
    if args.stage in ("all", "gauntlet"):
        campaign.gauntlet()
    campaign.save()
    print(f"campaign complete: {campaign.root}")


if __name__ == "__main__":
    main()
