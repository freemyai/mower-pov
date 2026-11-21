"""Produce and verify local onboard films and reviews from cached work timelines."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from mower_pov.cache import _code_identity, _git_commit, load_bundle
from mower_pov.quality import frame_candidates, verify_sequence
from mower_pov.rendering.blender_scene import PRESETS
from mower_pov.rendering.driver import _stats
from mower_pov.rendering.preview import encode_preview, frame_sources, mux_audio
from mower_pov.rendering.shutter import Exposure

FPS = 30
FEATURE_FRAMES = 1200
FEATURE_SIZE = (1920, 1080)
REVIEW_DURATIONS = {"yard": (8, 10), "dog": (8, 12), "hedgehog": (6, 10)}
COURSE_REVIEW_DURATIONS = {"work-area": (4, 12), "avoidance": (4, 12), "turning": (4, 12)}
REVIEW_PRESETS = ("preview", "beauty")
COURSE_SCENARIO = "mowing-course"


def absolute_range(value):
    match = re.fullmatch(r"(\d+):(\d+)", value)
    if match:
        first, stop = map(int, match.groups())
        if first < stop:
            return first, stop
    raise argparse.ArgumentTypeError("expected an ascending half-open absolute range FIRST:STOP")


def frame_shards(first, stop, count):
    n = min(count, stop - first)
    boundaries = [first + i * (stop - first) // n for i in range(n + 1)]
    return list(zip(boundaries, boundaries[1:]))


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    with TemporaryDirectory(prefix=".report-", dir=path.parent) as directory:
        temp = Path(directory) / path.name
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf8")
        temp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def gpu_settings():
    device = os.environ.get("MOWER_DEVICE", "OPTIX").upper()
    if device == "CPU":
        return device, (0,)
    indices = tuple(map(int, os.environ.get("MOWER_GPU_INDICES", "0,1,2,3").split(",")))
    if min(indices) < 0 or len(indices) != len(set(indices)):
        raise ValueError("MOWER_GPU_INDICES must contain distinct nonnegative indices")
    return device, indices


def shutter_settings():
    milliseconds = float(os.environ.get("MOWER_SHUTTER_MS", "0"))
    samples = int(os.environ.get("MOWER_SHUTTER_SAMPLES", "3" if milliseconds else "1"))
    exposure = Exposure(milliseconds, samples)
    return exposure.milliseconds, exposure.samples if exposure.milliseconds else 1


def render_batches(bundle, blender, output, frame_range, preset):
    device, indices = gpu_settings()
    milliseconds, samples = shutter_settings()
    ranges = frame_shards(*frame_range, len(indices))
    batches = [output / f"batch-{i}" for i in range(len(ranges))]
    if set(output.glob("batch-*")) - set(batches):
        raise ValueError("existing output has a different shard layout; use a new output directory")

    def render(job):
        batch, gpu, (first, stop) = job
        log_path = output / f"render-{batch.name}.log"
        with log_path.open("ab") as log:
            result = subprocess.run([
                sys.executable, "-m", "mower_pov", "render", str(bundle.run_dir),
                "--blender", str(blender), "--view", "onboard", "--yard", "house",
                "--device", device, "--gpu-index", str(gpu), "--preset", preset,
                "--frames", f"{first}:{stop}", "--output", str(batch),
                "--png-only", "--shutter-ms", str(milliseconds), "--shutter-samples", str(samples), "--deep-focus",
            ], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError(f"{batch.name} exited with {result.returncode}; see {log_path} and {batch / 'blender.log'}")

    # The renderer enforces exact request fingerprints and validates resumed PNGs.
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        list(pool.map(render, zip(batches, indices, ranges)))
    return batches


def contact_sheet(batches, frame_range, output, fps):
    first, stop = frame_range
    sample_count = min(12, stop - first)
    indices = np.linspace(first, stop - 1, sample_count, dtype=int)
    width, height, gap, label = 320, 180, 8, 18
    cols = min(4, sample_count)
    rows = (sample_count + cols - 1) // cols
    dx, dy = width + gap, height + gap + label
    sheet = Image.new("RGB", (cols * dx + gap, rows * dy + gap), "#161b1e")
    draw = ImageDraw.Draw(sheet)
    for i, (frame, path) in enumerate(zip(indices, frame_sources(batches, indices))):
        x, y = gap + i % cols * dx, gap + i // cols * dy
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 3, y + height + 4), f"{frame / fps:05.2f}s  |  frame {frame:04}",
                  fill="#eeeae3")
    sheet.save(output / "contact.jpg", quality=94)


def build_media(output, batches, frame_range, dimensions, filename, pace, fps):
    first, stop = frame_range
    duration = (stop - first) / fps
    video = output / f"{filename}.mp4"
    with TemporaryDirectory(prefix=".encode-", dir=output) as directory:
        silent = Path(directory) / "silent.mp4"
        report = encode_preview(batches, tuple(range(first, stop)), fps, silent)
        if (report["width"], report["height"]) != dimensions:
            raise ValueError("encoded dimensions differ from the requested render settings")
        if abs(report["duration_s"] - duration) > .001:
            raise ValueError("encoded video duration differs from the absolute frame window")
        report.update(mux_audio(silent, output / "timeline.wav", first, stop, fps, video))

    # Pace decoded frames by presentation timestamp; input pacing can finish ahead of buffered frames.
    video_filters = "blackdetect=d=0.15:pix_th=0.05,freezedetect=n=0.0005:d=0.45"
    audio_filters = "astats=metadata=0:reset=0"
    if pace:
        video_filters += ",realtime"
        audio_filters += ",arealtime"
    started = time.monotonic()
    with (output / "decode.log").open("w", encoding="utf8") as log:
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-v", "info", "-xerror", "-i", str(video),
            "-map", "0:v:0", "-map", "0:a:0", "-vf", video_filters, "-af", audio_filters, "-f", "null", "-",
        ], check=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    if pace and elapsed < duration - .35:
        raise ValueError("paced decode completed faster than the source timeline")
    decode_log = (output / "decode.log").read_text(encoding="utf8")
    peaks = [float(value) for value in re.findall(r"Peak level dB:\s*(-inf|[-+.\d]+)", decode_log)]
    if not peaks or max(peaks) >= -.05:
        raise ValueError("AAC full decode did not verify audio headroom")

    contact_sheet(batches, frame_range, output, fps)
    report.update(
        filename=video.name, sha256=file_digest(video), absolute_end_frame_exclusive=stop,
        machine_checks={"full_av_decode": True, "dimensions_and_timing": True, "audio_headroom": True},
        playback={"paced_decode": pace, "wall_s": elapsed, "expected_s": duration},
        audio_decoded_peak_dbfs=None if max(peaks) == float("-inf") else max(peaks),
        review_candidates=frame_candidates(decode_log),
    )
    return report


def audit_audio_alignment(output, report):
    first, stop = report["audio_start_sample"], report["audio_end_sample"]
    length = stop - first
    with wave.open(str(output / "timeline.wav"), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("audio alignment requires a signed 16-bit timeline WAV")
        wav.setpos(first)
        source = np.frombuffer(wav.readframes(length), dtype="<i2").reshape(-1, 2)
    pcm = subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(output / report["filename"]),
        "-map", "0:a:0", "-f", "s16le", "-c:a", "pcm_s16le", "-",
    ])
    decoded = np.frombuffer(pcm, dtype="<i2").reshape(-1, 2)
    max_lag, edge = 64, 4096
    if len(decoded) < length or len(decoded) - length > 1024 or length < 2 * edge + 1024:
        raise ValueError("decoded PCM length differs from the absolute WAV sample window")
    stride = max(8, (length + 119_999) // 120_000)
    indices = np.arange(edge, length - edge, stride)
    actual = decoded[indices].astype(np.float64)
    actual -= actual.mean(axis=0)
    actual_energy = np.einsum("ij,ij->", actual, actual)
    if not actual_energy:
        raise ValueError("decoded AAC contains no signal for the audio alignment check")
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        expected = source[indices + lag].astype(np.float64)
        expected -= expected.mean(axis=0)
        energy = np.einsum("ij,ij->", expected, expected)
        scores.append(np.einsum("ij,ij->", expected, actual) / np.sqrt(energy * actual_energy) if energy else 0.)
    best = int(np.argmax(scores))
    lag, correlation = best - max_lag, float(scores[best])
    if abs(lag) > 2 or correlation < .90:
        raise ValueError(f"decoded AAC alignment differs from the source WAV (lag={lag}, r={correlation:.4f})")
    return {
        "video_sha256": report["sha256"], "source_wav_first_sample": first,
        "source_wav_end_sample_exclusive": stop, "decoded_pcm_samples_including_codec_padding": len(decoded),
        "best_source_lag_samples": lag, "normalized_pcm_correlation": correlation,
        "cross_correlation_lag_range_samples": [-max_lag, max_lag], "comparison_stride_samples": stride,
        "excluded_edge_samples": edge, "subjective_audition_status": "pending",
    }


def player_clock_stats(log, duration, elapsed):
    status = re.findall(r"^ *([+-]?\d+\.\d+) A-V:\s*([+-]?\d+\.\d+) fd=\s*(\d+)",
                        log.replace("\r", "\n"), re.MULTILINE)
    samples = [(float(clock), float(skew), int(drops)) for clock, skew, drops in status]
    steady = [abs(skew) for clock, skew, _ in samples if clock >= .25]
    if not steady or samples[-1][0] < duration - .25:
        raise ValueError("ffplay did not reach the end of the media clock")
    # The SDL dummy audio clock can gain time over a long run. Still reject
    # playback far from real time, and apply the A/V and dropped-frame checks.
    early = max(.35, .015 * duration)
    late = max(1., .015 * duration)
    if not duration - early <= elapsed <= duration + late:
        raise ValueError(f"ffplay wall clock {elapsed:.3f}s differs from {duration:.3f}s of media")
    if max(steady) > .125:
        raise ValueError("ffplay audio and video clocks differ by more than 125 ms")
    if any(drops for _, _, drops in samples):
        raise ValueError("ffplay reported dropped frames")
    return {"expected_duration_s": duration, "wall_duration_s": elapsed,
            "final_player_clock_s": samples[-1][0], "sampled_status_lines": len(samples),
            "final_av_skew_s": samples[-1][1],
            "median_abs_av_skew_s_after_250ms": statistics.median(steady),
            "max_abs_av_skew_s_after_250ms": max(steady), "dropped_frames": 0}


def audit_player(output, report):
    width, height, duration = report["width"], report["height"], report["duration_s"]
    command = [
        "xvfb-run", "-a", "-s", f"-screen 0 {width}x{height}x24", "ffplay", "-autoexit", "-stats",
        "-sync", "audio", "-noframedrop", "-x", str(width), "-y", str(height), str(output / report["filename"]),
    ]
    environment = {**os.environ, "SDL_VIDEODRIVER": "x11", "SDL_RENDER_DRIVER": "software", "SDL_AUDIODRIVER": "dummy"}
    log_path = output / "ffplay.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf8") as log:
        subprocess.run(command, env=environment, timeout=2 * duration + 15, check=True,
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    try:
        stats = player_clock_stats(log_path.read_text(encoding="utf8"), duration, elapsed)
    except ValueError as error:
        raise ValueError(f"{error}; see {log_path}") from error
    return {
        "video": report["filename"], "video_sha256": report["sha256"], "exit_code": 0,
        **stats, "audio_output": "SDL dummy device", "video_output": "SDL software display via Xvfb",
        "display_playback_pass": True, "command": command,
    }


def write_state(output, bundle, frame_range):
    from mower_pov.actors.behavior import ACTOR_NAMES, actor_pose

    trajectory, story = bundle.trajectory, bundle.story
    cfg = bundle.resolved_config
    fields = ("mode", "blocked", "blade_command", "rotor_fraction", "clearance_m", "obstacle_name")
    with (output / "state.jsonl").open("w", encoding="utf8") as file:
        for frame in range(*frame_range):
            absolute_s = frame / cfg.render.fps
            i = np.searchsorted(trajectory.time_s, absolute_s + 1e-10, side="right") - 1
            controller = {name: getattr(story, name)[i].item() for name in fields}
            if not np.isfinite(controller["clearance_m"]):
                controller["clearance_m"] = None
            actors = {}
            for name in ACTOR_NAMES:
                pose = actor_pose(bundle, name, absolute_s)
                actors[name] = {"xy_m": list(map(float, pose.xy)), "heading_rad": float(pose.heading),
                                "distance_m": float(pose.distance), "movement_weight": float(pose.moving)}
            row = {
                "frame": frame, "absolute_s": absolute_s,
                "mower": {"position_m": trajectory.position_m[i].tolist(), "speed_mps": float(trajectory.v_mps[i]),
                          "wheel_speeds_mps": trajectory.wheel_speeds[i].tolist()},
                "controller": controller, "actors": actors,
            }
            if cfg.scenario == COURSE_SCENARIO:
                row["mower"].update(yaw_rad=float(trajectory.unwrapped_yaw[i]),
                                    yaw_rate_radps=float(trajectory.omega_radps[i]),
                                    blade_enabled=bool(trajectory.blade_enabled[i]))
            file.write(json.dumps(row, allow_nan=False) + "\n")
    event_path = bundle.run_dir / "events.json"
    shutil.copyfile(event_path, output / "timeline-events.json")


def approved_reviews(directory, run_id, review_mode="review"):
    source = _code_identity()
    reports = {}
    names = COURSE_REVIEW_DURATIONS if review_mode == "course-review" else REVIEW_DURATIONS
    for name in names:
        folder = directory / name
        proof = read_json(folder / "acceptance.json")
        current = read_json(folder / "manifest.json")
        if (proof["name"] != name or proof["visual_review"]["status"] != "accepted"
                or current["mode"] != review_mode or current["preset"] not in REVIEW_PRESETS or current["name"] != name
                or proof["media_sha256"] != current["media"]["sha256"]
                or proof["manifest_sha256"] != file_digest(folder / "manifest.json")
                or proof["media_sha256"] != file_digest(folder / f"{name}.mp4")
                or current["bundle"]["run_id"] != run_id
                or current["render"]["source_sha256"] != source):
            raise ValueError(f"{name} acceptance does not match the exact clip, bundle or renderer")
        reports[name] = {**proof, "evidence_directory": str(folder.resolve())}
    return reports


def accept_review(args):
    output = args.output / args.name
    manifest_path = output / "manifest.json"
    mode = "course-review" if args.name in COURSE_REVIEW_DURATIONS else "review"
    with (output / ".production.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = read_json(manifest_path)
        media = manifest["media"]
        if (manifest["mode"] != mode or manifest["name"] != args.name or manifest["preset"] not in REVIEW_PRESETS
                or not all(media["machine_checks"].values())
                or file_digest(output / media["filename"]) != media["sha256"]):
            raise ValueError(f"{args.name} has no verified review clip to accept")
        if not args.visual_notes.strip():
            raise ValueError("a visual assessment is required")
        write_json(output / "acceptance.json", {
            "name": args.name, "media_sha256": media["sha256"],
            "manifest_sha256": file_digest(manifest_path), "visual_review": {
                "status": "accepted", "method": args.visual_method, "notes": args.visual_notes},
            "audio_audition": {"status": "auditioned" if args.audio_notes else "pending", "notes": args.audio_notes},
        })
    print(output / "acceptance.json")


def run_production(args):
    bundle = load_bundle(args.bundle.resolve())
    cfg = bundle.resolved_config
    is_course = args.mode in {"course", "course-review"}
    fps = cfg.render.fps
    if is_course:
        if bundle.story is None or cfg.scenario != COURSE_SCENARIO or cfg.frame_count <= 60 * fps:
            raise ValueError("expected a mowing-course work bundle with more than 60 seconds of frames")
    elif bundle.story is None or (fps, cfg.frame_count) != (FPS, FEATURE_FRAMES):
        raise ValueError("expected a 40-second, 30 fps dynamic backyard work bundle")

    reviews = None
    if args.mode in {"feature", "course"}:
        preset = "final"
        name = "mower-pov-mowing-course" if is_course else "mower-pov-dynamic-40s"
        frame_range, dimensions = (0, cfg.frame_count), (cfg.render.width, cfg.render.height)
        if not is_course and dimensions != FEATURE_SIZE:
            raise ValueError("the full feature requires a 1920x1080 render bundle")
        review_mode = "course-review" if is_course else "review"
        reviews = approved_reviews(args.reviews, bundle.run_id, review_mode)
        output = args.output
    else:
        preset, name, frame_range = args.preset, args.name, args.frames
        dimensions = ((cfg.render.width, cfg.render.height) if preset == "final" else PRESETS[preset][:2])
        durations = COURSE_REVIEW_DURATIONS if is_course else REVIEW_DURATIONS
        low, high = durations[name]
        frame_count = frame_range[1] - frame_range[0]
        if frame_range[1] > cfg.frame_count:
            raise ValueError(f"{name} range exceeds the {cfg.frame_count}-frame timeline")
        if preset in REVIEW_PRESETS and not low * fps <= frame_count <= high * fps:
            raise ValueError(f"{name} requires a {low}–{high}-second window within the full timeline")
        # Short preflight runs can live beside the corresponding review.
        suffix = {"smoke": "-smoke", "final": "-preflight"}.get(preset, "")
        output = args.output / (name + suffix)

    playback = preset in REVIEW_PRESETS or args.mode in {"feature", "course"}
    if is_course and playback:
        missing = [name for name in ("ffplay", "xvfb-run", "Xvfb", "xauth") if not shutil.which(name)]
        if missing:
            raise ValueError(f"course playback requires tools missing from PATH: {', '.join(missing)}")

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".production.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.monotonic()
        batches = render_batches(bundle, args.blender, output, frame_range, preset)
        render_wall = time.monotonic() - started
        frames = tuple(range(*frame_range))
        verify_sequence(batches, frames, dimensions, linear=False)
        subprocess.run([sys.executable, "-m", "mower_pov", "audio", str(bundle.run_dir),
                        "--output", str(output / "timeline.wav")], check=True)
        report = build_media(output, batches, frame_range, dimensions, name, playback and not is_course, fps)
        write_state(output, bundle, frame_range)

        requests = [read_json(batch / "request.json") for batch in batches]
        timings = {}
        for batch in batches:
            for row in _stats(batch / "frames.jsonl"):
                if "frame" in row:
                    timings[row["frame"]] = row["render_and_write_s"]
        identity = requests[0]["identity"]
        evidence = {
            "contact_sheet": "contact.jpg", "state_log": "state.jsonl", "decode_log": "decode.log",
            "events": "timeline-events.json", "audio": "timeline.wav",
        }
        render_metrics = {}
        if is_course:
            render_metrics = {
                "render_stage_wall_s": render_wall,
                "render_workers": len(batches),
                "timed_frames": len(timings),
                "p95_seconds_per_frame": float(np.percentile(list(timings.values()), 95)),
            }
            if playback:
                write_json(output / "ffplay-audit.json", audit_player(output, report))
                write_json(output / "audio-alignment.json", audit_audio_alignment(output, report))
                evidence.update(player="ffplay-audit.json", player_log="ffplay.log", audio_alignment="audio-alignment.json")
                report["machine_checks"].update(display_playback=True, pcm_alignment=True)
        write_json(output / "manifest.json", {
            "mode": args.mode, "name": name, "preset": preset, "media": report,
            "review_acceptance": reviews,
            "bundle": {"run_id": bundle.run_id, "path": str(bundle.run_dir),
                       "manifest_sha256": file_digest(bundle.run_dir / "manifest.json")},
            "render": {"source_sha256": identity["source_sha256"], "git_commit": _git_commit(),
                       "blender": identity["blender"], "blender_build": identity["blender_build"],
                       "view": identity["view"], "yard": identity["yard"], "max_samples": (
                           cfg.render.samples if preset == "final" else PRESETS[preset][2]),
                       "shutter_ms": identity["shutter_ms"], "shutter_samples": identity["shutter_samples"],
                       "verified_pngs": len(frames), "median_seconds_per_frame": statistics.median(timings.values()),
                       **render_metrics,
                       "requests": {batch.name: request for batch, request in zip(batches, requests)}},
            "evidence": {key: {"filename": name, "sha256": file_digest(output / name)}
                         for key, name in evidence.items()},
        })
    print(output / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode, roles, presets in (
        ("review", REVIEW_DURATIONS, (*REVIEW_PRESETS, "smoke")),
        ("course-review", COURSE_REVIEW_DURATIONS, (*REVIEW_PRESETS, "smoke", "final")),
    ):
        review = commands.add_parser(mode, help="render one absolute onboard review window")
        review.add_argument("name", choices=roles)
        review.add_argument("bundle", type=Path)
        review.add_argument("blender", type=Path)
        review.add_argument("output", type=Path)
        review.add_argument("frames", type=absolute_range)
        review.add_argument("--preset", choices=presets, default="preview")

    accept = commands.add_parser("accept", help="record an assessment tied to the verified review video")
    accept.add_argument("name", choices=(*REVIEW_DURATIONS, *COURSE_REVIEW_DURATIONS))
    accept.add_argument("output", type=Path)
    accept.add_argument("--visual-method", choices=("contact-sheet", "playback"), required=True)
    accept.add_argument("--visual-notes", required=True)
    accept.add_argument("--audio-notes")

    for mode, help_text in (
        ("feature", "render the full 40-second feature after review acceptance"),
        ("course", "render the full mowing course after review acceptance"),
    ):
        feature = commands.add_parser(mode, help=help_text)
        feature.add_argument("bundle", type=Path)
        feature.add_argument("blender", type=Path)
        feature.add_argument("output", type=Path)
        feature.add_argument("reviews", type=Path)

    args = parser.parse_args()
    try:
        if args.mode == "accept":
            accept_review(args)
        else:
            run_production(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
