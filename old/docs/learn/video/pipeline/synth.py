#!/usr/bin/env python3
"""Synthesize per-beat narration with Piper and emit a frame-quantized timing manifest.

Source of truth: scripts/<VID>-narration.json
Outputs:
  audio/<VID>/<beat_id>.wav       one clip per beat
  manifests/<VID>-timing.json     per-beat + per-scene durations and global offsets

The timing manifest is the contract shared by three consumers:
  * the manim scenes  -> size each beat so beat_dur >= audio_dur (never overrun)
  * build_audio.py    -> place each clip at its global offset
  * verify.py         -> assert audio_dur <= beat_dur and total drift ~ 0

Beat budgets are quantized UP to the 1/fps grid so the final render is frame-exact.
Run inside the venv:  venv/bin/python pipeline/synth.py 02
"""
import json
import math
import os
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("OMP_NUM_THREADS", "2")  # respect the box: cap TTS threads


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main(vid: str) -> None:
    from piper import PiperVoice, SynthesisConfig  # imported late so --help stays cheap

    spec = json.loads((ROOT / "scripts" / f"{vid}-narration.json").read_text())
    fps = int(spec["fps"])
    pad = float(spec["pad_seconds"])
    scene_gap = float(spec.get("scene_gap_seconds", 0.0))
    voice_name = spec["voice"]
    syn_cfg = SynthesisConfig(length_scale=float(spec.get("length_scale", 1.0)))

    voice = PiperVoice.load(str(ROOT / "voices" / f"{voice_name}.onnx"))
    out_audio = ROOT / "audio" / vid
    out_audio.mkdir(parents=True, exist_ok=True)

    def quant_up(sec: float) -> float:
        return math.ceil(sec * fps - 1e-9) / fps

    scenes_out = []
    global_offset = 0.0
    total_words = 0
    for si, scene in enumerate(spec["scenes"]):
        beats_out = []
        scene_start = global_offset
        beat_offset = 0.0
        for beat in scene["beats"]:
            bid = beat["id"]
            text = beat["text"]
            total_words += len(text.split())
            wav_path = out_audio / f"{bid}.wav"
            with wave.open(str(wav_path), "wb") as wf:
                voice.synthesize_wav(text, wf, syn_config=syn_cfg)
            adur = wav_duration(wav_path)
            beat_dur = quant_up(adur + pad)
            beats_out.append({
                "id": bid,
                "text": text,
                "wav": str(wav_path.relative_to(ROOT)),
                "audio_dur": round(adur, 4),
                "beat_dur": round(beat_dur, 4),
                "scene_beat_offset": round(beat_offset, 4),
                "global_offset": round(scene_start + beat_offset, 4),
            })
            beat_offset += beat_dur
        # trailing breath at the end of each scene except the last
        tail = quant_up(scene_gap) if si < len(spec["scenes"]) - 1 else 0.0
        scene_dur = beat_offset + tail
        scenes_out.append({
            "key": scene["key"],
            "title": scene.get("title", ""),
            "duration": round(scene_dur, 4),
            "tail": round(tail, 4),
            "global_start": round(scene_start, 4),
            "beats": beats_out,
        })
        global_offset += scene_dur

    manifest = {
        "video": vid,
        "title": spec["title"],
        "voice": voice_name,
        "fps": fps,
        "pad_seconds": pad,
        "total_duration": round(global_offset, 4),
        "total_words": total_words,
        "scenes": scenes_out,
    }
    man_path = ROOT / "manifests" / f"{vid}-timing.json"
    man_path.write_text(json.dumps(manifest, indent=2))

    mins = int(global_offset // 60)
    secs = global_offset - mins * 60
    print(f"[synth] {vid}: {len(scenes_out)} scenes, "
          f"{sum(len(s['beats']) for s in scenes_out)} beats, "
          f"{total_words} words")
    print(f"[synth] total video duration = {global_offset:.2f}s ({mins}m{secs:04.1f}s)")
    print(f"[synth] manifest -> {man_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "02")
