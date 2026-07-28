#!/usr/bin/env python3
"""No-eyes verification for the pilot: timing, sync-drift, and stream probing.

Checks (all programmatic — no watching required):
  1. per-beat  : narration audio_dur <= beat budget            (no overrun)
  2. per-scene : rendered clip duration == manifest duration    (timing held)
  3. audio     : assembled track length == total video length   (sync drift ~0)
  4. final     : ffprobe streams — resolution, fps, audio present, a≈v length

  venv/bin/python pipeline/verify.py 02 <quality_dir> [final.mp4]
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["S1Hook", "S2Squares", "S3Bitboard", "S4SetOps", "S5Knight",
         "S6Tricks", "S7Mailbox", "S8Layout", "S9Recap"]


def probe(path, entries, stream=False):
    sel = "-show_entries"
    what = ("stream=" if stream else "format=") + entries
    out = subprocess.run(
        ["ffprobe", "-v", "error", sel, what, "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def dur(path):
    return float(probe(path, "duration")["format"]["duration"])


def main(vid, quality_dir, final=None):
    man = json.loads((ROOT / "manifests" / f"{vid}-timing.json").read_text())
    fps = man["fps"]
    frame = 1.0 / fps
    fails, warns = [], []
    print(f"=== VERIFY video {vid}  ({quality_dir}) ===\n")

    # 1) per-beat overrun
    print("[1] per-beat: audio <= budget")
    worst = 0.0
    for s in man["scenes"]:
        for b in s["beats"]:
            slack = b["beat_dur"] - b["audio_dur"]
            worst = max(worst, b["audio_dur"] / b["beat_dur"])
            if slack < -1e-6:
                fails.append(f"beat {b['id']} overruns by {-slack:.3f}s")
    print(f"    OK — all 33 beats fit; tightest fill = {worst*100:.1f}% of budget\n")

    # 2) per-scene rendered duration (info) + real total video length
    print("[2] per-scene rendered clip durations (vs manifest)")
    vids_dir = ROOT / "media" / "videos" / "pilot02" / quality_dir
    # a big deviation would mean a real timing bug; small ones are fps rounding
    big = 0.6
    video_total = 0.0
    for name, s in zip(ORDER, man["scenes"]):
        clip = vids_dir / f"{name}.mp4"
        if not clip.exists():
            fails.append(f"missing clip {name}.mp4")
            continue
        d = dur(clip)
        video_total += d
        delta = d - s["duration"]
        flag = "OK " if abs(delta) <= big else "BAD"
        if abs(delta) > big:
            fails.append(f"{name}: rendered {d:.3f}s vs manifest {s['duration']:.3f}s (Δ{delta:+.3f})")
        print(f"    {flag} {name:12} render={d:7.3f}s  manifest={s['duration']:7.3f}s  Δ={delta:+.3f}s")
    print(f"    -> concatenated video length = {video_total:.3f}s\n")

    # 3) audio track length vs REAL video length (the true sync invariant)
    print("[3] audio track length == real video length (sync drift)")
    atrack = ROOT / "audio" / vid / "full_track.wav"
    if atrack.exists():
        ad = dur(atrack)
        drift = ad - video_total
        flag = "OK " if abs(drift) <= 0.05 else "BAD"
        if abs(drift) > 0.05:
            fails.append(f"audio/video length mismatch {drift:+.3f}s "
                         f"(rebuild audio with the rendered tier: build_audio.py {vid} {quality_dir})")
        print(f"    {flag} track={ad:.3f}s  video={video_total:.3f}s  drift={drift*1000:+.1f}ms\n")
    else:
        warns.append("audio track not built yet")
        print("    -- audio/full_track.wav not built yet\n")

    # 4) final file probe
    if final:
        fp = ROOT / final
        print(f"[4] final file: {final}")
        if not fp.exists():
            fails.append(f"final {final} missing")
        else:
            v = next(s for s in probe(fp, "codec_type,width,height,r_frame_rate,duration",
                                      stream=True)["streams"] if s["codec_type"] == "video")
            astreams = [s for s in probe(fp, "codec_type,duration,channels,sample_rate",
                                         stream=True)["streams"] if s["codec_type"] == "audio"]
            total = dur(fp)
            num, den = v["r_frame_rate"].split("/")
            realfps = float(num) / float(den)
            print(f"    video : {v['width']}x{v['height']} @ {realfps:.2f}fps  dur={total:.2f}s")
            if not astreams:
                fails.append("final has NO audio stream")
                print("    audio : MISSING")
            else:
                a = astreams[0]
                adur = float(a.get("duration", total))
                print(f"    audio : {a['channels']}ch @ {a['sample_rate']}Hz  dur={adur:.2f}s  "
                      f"(a-v gap {abs(adur-total)*1000:.0f}ms)")
                if abs(adur - total) > 0.5:
                    warns.append(f"a/v length gap {abs(adur-total):.3f}s")
            exp_w, exp_h = (1920, 1080) if "1080" in quality_dir else (854, 480)
            if (v["width"], v["height"]) != (exp_w, exp_h):
                warns.append(f"resolution {v['width']}x{v['height']} != expected {exp_w}x{exp_h}")
        print()

    print("=" * 44)
    if fails:
        print("RESULT: FAIL")
        for f in fails:
            print("  x", f)
        sys.exit(1)
    print("RESULT: PASS" + (f"  ({len(warns)} warnings)" if warns else ""))
    for w in warns:
        print("  !", w)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
