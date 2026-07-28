#!/usr/bin/env python3
"""Concatenate the rendered scene clips in order and mux the narration track.

  venv/bin/python pipeline/assemble.py 02 <quality_dir> <out.mp4>
  e.g. ... 02 480p15  media/pilot/archon-02-480p.mp4
       ... 02 1080p30 media/pilot/archon-02-board-representation.mp4

Threads are capped at 2 (tournament box rule). Video is stream-copied when the
scene clips share codec params (they do — same manim settings); falls back to a
re-encode if copy fails.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# manifest scene-key -> manim Scene class -> mp4 basename
ORDER = ["S1Hook", "S2Squares", "S3Bitboard", "S4SetOps", "S5Knight",
         "S6Tricks", "S7Mailbox", "S8Layout", "S9Recap"]


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main(vid: str, quality_dir: str, out_rel: str) -> None:
    vids_dir = ROOT / "media" / "videos" / "pilot02" / quality_dir
    clips = [vids_dir / f"{name}.mp4" for name in ORDER]
    missing = [c for c in clips if not c.exists()]
    if missing:
        raise SystemExit(f"missing scene clips: {[str(m) for m in missing]}")

    audio = ROOT / "audio" / vid / "full_track.wav"
    if not audio.exists():
        raise SystemExit("run build_audio.py first (audio/<vid>/full_track.wav missing)")

    out = ROOT / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)

    listfile = ROOT / "media" / "pilot" / f"_concat_{quality_dir}.txt"
    listfile.parent.mkdir(parents=True, exist_ok=True)
    listfile.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips))

    silent = out.with_suffix(".silent.mp4")
    # 1) concat video only (stream copy)
    try:
        run(["ffmpeg", "-y", "-v", "error", "-threads", "2",
             "-f", "concat", "-safe", "0", "-i", str(listfile),
             "-c", "copy", str(silent)])
    except subprocess.CalledProcessError:
        run(["ffmpeg", "-y", "-v", "error", "-threads", "2",
             "-f", "concat", "-safe", "0", "-i", str(listfile),
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", str(silent)])
    # 2) mux narration
    run(["ffmpeg", "-y", "-v", "error", "-threads", "2",
         "-i", str(silent), "-i", str(audio),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(out)])
    silent.unlink(missing_ok=True)
    print(f"[assemble] -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
