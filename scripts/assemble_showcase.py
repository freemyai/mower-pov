"""Assemble the four independent garden shots with an ambience-forward soundtrack."""

import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from mower_pov.audio import mix_audio
from mower_pov.cache import compute_run_id, load_bundle
from mower_pov.quality import verify_sequence
from mower_pov.rendering.preview import encode_preview, mux_audio
from mower_pov.rendering.showcase_motion import SHOTS


def garden_audio(bundle, output):
    mix = mix_audio(bundle)
    rate = mix.rate
    weights = {"hedgehog": .05, "dog": .07, "trampoline": .04, "cut-strip": .70}
    pieces = []
    for shot in SHOTS:
        lo, hi = (round(frame * rate / 30) for frame in (shot.first_frame, shot.end_frame))
        machinery = mix.drive[lo:hi] + mix.blade[lo:hi] + mix.grass[lo:hi]
        audio = 2.55 * mix.yard[lo:hi] + weights[shot.name] * machinery
        if shot.name == "trampoline":
            t = np.arange(round(.13 * rate)) / rate
            thump = .043 * np.sin(2 * np.pi * (79 - 21 * t) * t) * np.exp(-38 * t)
            thump *= np.minimum(1., t / .004)
            for hop in range(5):
                landing = round((26.35 + (hop + .68) * .84) * rate) - lo
                audio[landing:landing + len(t)] += thump[:, None]
        fade = round(.040 * rate)
        ramp = np.linspace(0, 1, fade)[:, None]
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
        pieces.append(audio)
    samples = np.concatenate(pieces)
    data = np.rint(np.clip(samples, -.86, .86) * 32767).astype("<i2")
    with wave.open(str(output), "wb") as sound:
        sound.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        sound.writeframes(data.tobytes())


def assemble(bundle_path, folder):
    out = folder.resolve()
    work = load_bundle(bundle_path)
    final = out / "garden-extras-18s.mp4"
    inputs = sorted(out.glob("batch-*"))
    for shard in inputs:
        request = json.loads((shard / "request.json").read_text())
        identity = request["identity"]
        if (request["fingerprint"] != compute_run_id(identity) or identity.get("run_id") != work.run_id
                or identity.get("view") != "showcase" or identity.get("preset") != "beauty"):
            raise ValueError(f"mismatched garden footage: {shard}")
    selected = tuple(frame for shot in SHOTS for frame in range(shot.first_frame, shot.end_frame))
    verify_sequence(inputs, selected, (1280, 720), linear=False)
    total_frames = len(selected)
    with tempfile.TemporaryDirectory(prefix=".showcase-", dir=out) as temp:
        temp = Path(temp)
        lines = []
        for i, shot in enumerate(SHOTS):
            name = f"shot-{i}.mp4"
            encode_preview(inputs, tuple(range(shot.first_frame, shot.end_frame)), 30, temp / name)
            lines.append(f"file {name}\n")
        index = temp / "segments.txt"
        index.write_text("".join(lines))
        joined = temp / "joined.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-f", "concat", "-safe", "1",
            "-i", str(index), "-map", "0:v", "-c", "copy", str(joined),
        ], check=True)
        wav = temp / "garden.wav"
        garden_audio(work, wav)
        report = mux_audio(joined, wav, 0, total_frames, 30, final)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(final), "-f", "null", "-"],
                   check=True)
    report.update(frames=total_frames, fps=30, source_ranges=[shot.__dict__ for shot in SHOTS])
    final.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_bundle", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    assemble(args.work_bundle, args.output_dir)
