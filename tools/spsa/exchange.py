"""Exchange layer for distributed SPSA (D36-lite): a dumb shared prefix.

Two backends behind four verbs (put/get/list/exists):
  file:///abs/dir   — local filesystem (dry runs, local-worker mode)
  gs://bucket/pfx   — GCS via the `gcloud storage` CLI (spot workers)

No locking, no daemon. Claims are advisory (duplicate unit play is
harmless — the coordinator keeps the first result per unit id). file://
writes are atomic (tmp + rename); gs:// uploads are already atomic.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def _is_gs(uri: str) -> bool:
    return uri.startswith("gs://")


def _local(uri: str) -> Path:
    assert uri.startswith("file://"), uri
    return Path(uri[len("file://"):])


def put_bytes(uri: str, name: str, data: bytes) -> None:
    if _is_gs(uri):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(data)
        try:
            subprocess.run(["gcloud", "storage", "cp", "-q", tf.name,
                            f"{uri.rstrip('/')}/{name}"], check=True)
        finally:
            os.unlink(tf.name)
    else:
        dest = _local(uri) / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        # The temp name must be unique PER WRITER. Workers race to claim the
        # same unit, so with a shared `<dest>.tmp` two of them write the same
        # path, the first os.replace consumes it, and the rest die with
        # FileNotFoundError. That is not theoretical: it killed 3 of 4 local
        # workers at startup on 2026-07-27 and quartered SPSA throughput
        # while the run looked healthy from the coordinator's side.
        tmp = dest.with_suffix(f"{dest.suffix}.{os.getpid()}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


def put_file(uri: str, name: str, src: Path) -> None:
    if _is_gs(uri):
        subprocess.run(["gcloud", "storage", "cp", "-q", str(src),
                        f"{uri.rstrip('/')}/{name}"], check=True)
    else:
        put_bytes(uri, name, Path(src).read_bytes())


def get_bytes(uri: str, name: str) -> bytes | None:
    if _is_gs(uri):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            pass
        try:
            r = subprocess.run(["gcloud", "storage", "cp", "-q",
                                f"{uri.rstrip('/')}/{name}", tf.name],
                               capture_output=True)
            if r.returncode:
                return None
            return Path(tf.name).read_bytes()
        finally:
            os.unlink(tf.name)
    p = _local(uri) / name
    return p.read_bytes() if p.exists() else None


def get_file(uri: str, name: str, dest: Path) -> bool:
    data = get_bytes(uri, name)
    if data is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return True


def list_names(uri: str, prefix: str) -> list[str]:
    """Names (relative to the exchange root) under `prefix`."""
    if _is_gs(uri):
        r = subprocess.run(["gcloud", "storage", "ls",
                            f"{uri.rstrip('/')}/{prefix}*"],
                           capture_output=True, text=True)
        if r.returncode:
            return []
        root = uri.rstrip("/") + "/"
        return [line[len(root):] for line in r.stdout.split()
                if line.startswith(root)]
    base = _local(uri)
    dirpart, _, namepart = prefix.rpartition("/")
    pdir = base / dirpart if dirpart else base
    if not pdir.is_dir():
        return []
    out = []
    for p in pdir.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith(namepart) and not p.name.endswith(".tmp"):
            out.append(f"{dirpart}/{p.name}" if dirpart else p.name)
    return sorted(out)


def exists(uri: str, name: str) -> bool:
    if _is_gs(uri):
        return get_bytes(uri, name) is not None
    return (_local(uri) / name).exists()


def mtime_age(uri: str, name: str) -> float | None:
    """Seconds since last modification; None if missing. gs:// claims use
    re-upload as heartbeat, and age falls back to 0 when stat is
    unavailable (conservative: treat as fresh)."""
    if _is_gs(uri):
        return 0.0 if exists(uri, name) else None
    p = _local(uri) / name
    if not p.exists():
        return None
    import time
    return time.time() - p.stat().st_mtime
