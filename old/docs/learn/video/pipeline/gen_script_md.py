#!/usr/bin/env python3
"""Generate the human-readable narration script (Markdown) from the JSON source.

Keeps scripts/<VID>-narration.md in lockstep with the machine source of truth
(scripts/<VID>-narration.json) so they can never drift. If the timing manifest
exists, per-beat measured durations and cumulative timestamps are shown too.

  venv/bin/python pipeline/gen_script_md.py 02
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ts(sec: float) -> str:
    m = int(sec // 60)
    return f"{m}:{sec - 60*m:05.2f}"


def main(vid: str) -> None:
    spec = json.loads((ROOT / "scripts" / f"{vid}-narration.json").read_text())
    man_path = ROOT / "manifests" / f"{vid}-timing.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else None
    dmap = {}
    if man:
        for s in man["scenes"]:
            for b in s["beats"]:
                dmap[b["id"]] = b

    L = []
    L.append(f"# Narration — Video {vid}: {spec['title']}\n")
    L.append("> Generated from `scripts/{0}-narration.json` by "
             "`pipeline/gen_script_md.py` — edit the JSON, not this file.\n".format(vid))
    L.append(f"- **Voice:** `{spec['voice']}` (Piper), length_scale "
             f"{spec.get('length_scale', 1.0)}")
    L.append(f"- **Per-beat pad:** {spec['pad_seconds']}s   "
             f"**inter-scene gap:** {spec.get('scene_gap_seconds', 0)}s   "
             f"**fps:** {spec['fps']}")
    if man:
        L.append(f"- **Total runtime:** {ts(man['total_duration'])} "
                 f"({man['total_duration']:.1f}s), {man['total_words']} words")
    L.append("\n---\n")

    for si, scene in enumerate(spec["scenes"], 1):
        smob = None
        if man:
            smob = next(s for s in man["scenes"] if s["key"] == scene["key"])
            L.append(f"## Scene {si} · {scene.get('title','')}  "
                     f"`[{scene['key']}]`  —  start {ts(smob['global_start'])}, "
                     f"{smob['duration']:.1f}s\n")
        else:
            L.append(f"## Scene {si} · {scene.get('title','')}  `[{scene['key']}]`\n")
        for beat in scene["beats"]:
            b = dmap.get(beat["id"])
            if b:
                L.append(f"**`{beat['id']}`**  ·  @{ts(b['global_offset'])}  ·  "
                         f"audio {b['audio_dur']:.1f}s / budget {b['beat_dur']:.1f}s\n")
            else:
                L.append(f"**`{beat['id']}`**\n")
            L.append(f"> {beat['text']}\n")
        L.append("")

    out = ROOT / "scripts" / f"{vid}-narration.md"
    out.write_text("\n".join(L))
    print(f"[gen_script_md] -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "02")
