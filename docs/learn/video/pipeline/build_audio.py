#!/usr/bin/env python3
"""Assemble one narration track from the per-beat clips, placed at exact offsets.

Reads manifests/<VID>-timing.json and overlays each beat's wav onto a silence bed.
Beat offsets *within* a scene come from the manifest (frame-exact). Cumulative
*scene start* offsets optionally come from the ACTUAL rendered clip durations
(probe) — this absorbs manim's end-of-scene frame-count quirk (it emits one fewer
frame than the mathematical duration), so the track tracks the real video and A/V
sync is exact, not just approximate.

  venv/bin/python pipeline/build_audio.py 02 [quality_dir]
     quality_dir given -> scene starts from real media/videos/pilot02/<dir> clips
     omitted           -> scene starts from the manifest (mathematical) durations
  ->  audio/<VID>/full_track.wav
"""
import json
import subprocess
import sys
from pathlib import Path

from pydub import AudioSegment

ROOT = Path(__file__).resolve().parent.parent
SR = 22050
ORDER = ["S1Hook", "S2Squares", "S3Bitboard", "S4SetOps", "S5Knight",
         "S6Tricks", "S7Mailbox", "S8Layout", "S9Recap"]


def clip_dur(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def main(vid: str, quality_dir: str | None = None) -> None:
    man = json.loads((ROOT / "manifests" / f"{vid}-timing.json").read_text())

    # real scene start offsets (cumulative) if a rendered tier is given
    real_start = {}
    if quality_dir:
        vids = ROOT / "media" / "videos" / f"pilot{vid}" / quality_dir
        acc = 0.0
        for name, scene in zip(ORDER, man["scenes"]):
            real_start[scene["key"]] = acc
            acc += clip_dur(vids / f"{name}.mp4")
        total = acc
        print(f"[build_audio] using REAL {quality_dir} scene durations; total={total:.3f}s")
    else:
        for scene in man["scenes"]:
            real_start[scene["key"]] = scene["global_start"]
        total = man["total_duration"]

    total_ms = round(total * 1000)
    track = AudioSegment.silent(duration=total_ms, frame_rate=SR).set_channels(1)
    placed = 0
    for scene in man["scenes"]:
        base = real_start[scene["key"]]
        for beat in scene["beats"]:
            clip = (AudioSegment.from_wav(ROOT / beat["wav"])
                    .set_channels(1).set_frame_rate(SR))
            pos = round((base + beat["scene_beat_offset"]) * 1000)
            track = track.overlay(clip, position=pos)
            placed += 1

    track = track.fade_in(120).fade_out(400)
    out = ROOT / "audio" / vid / "full_track.wav"
    track.export(out, format="wav")
    print(f"[build_audio] placed {placed} clips  ->  {out.relative_to(ROOT)} "
          f"({len(track)/1000:.3f}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "02",
         sys.argv[2] if len(sys.argv) > 2 else None)
