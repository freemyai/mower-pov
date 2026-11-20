"""Inspect frame evidence and the encoded film without overclaiming artistic review."""

import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

from .cache import load_bundle
from .rendering.driver import _stats, validate_frame
from .rendering.preview import frame_sources


def verify_sequence(directories, frames, dimensions, *, linear=True):
    directories = tuple(map(Path, directories))
    paths = frame_sources(directories, frames)
    ledgers = {}
    for directory in map(Path, directories):
        ledgers[directory] = {record["frame"]: record for record in _stats(directory / "frames.jsonl")
                              if "frame" in record}
    for number, path in zip(frames, paths):
        stats = ledgers[path.parent].get(number, {})
        if not validate_frame(path, stats) or (linear and not validate_frame(path.with_suffix(".exr"), stats)):
            raise ValueError(f"corrupt or missing PNG/EXR pair for frame {number}")
        with path.open("rb") as image:
            header = image.read(24)
        if len(header) != 24 or struct.unpack(">II", header[16:24]) != dimensions:
            raise ValueError(f"invalid PNG dimensions for frame {number}")
        if linear:
            with path.with_suffix(".exr").open("rb") as raw:
                header = raw.read(8192)
            if not all(key in header for key in (b"RawBeauty.R", b"Albedo.R", b"Normal.X")):
                raise ValueError(f"missing linear beauty or guide channels for frame {number}")
    return paths


def frame_candidates(log):
    keys = ("black_start", "black_end", "freeze_start", "freeze_end")
    return {key: [float(number) for number in re.findall(
        rf"(?<!\w){key}:\s*([0-9]+(?:\.[0-9]+)?)", log)] for key in keys}


def inspect_run(run_dir: Path, video_path: Path, *, visual_notes=None, audio_notes=None) -> dict:
    """Full decode and format checks plus explicit visual and sound review status."""
    state = load_bundle(run_dir)
    cfg = state.resolved_config
    report = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(video_path),
    ], text=True)
    streams = json.loads(report)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next((stream for stream in streams if stream["codec_type"] == "audio"), {})
    audio_filter = ["-af", "astats=metadata=0:reset=0"] if audio else []
    inspected = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "info", "-xerror", "-i", str(video_path),
        "-vf", "blackdetect=d=0.15:pix_th=0.05,freezedetect=n=0.0005:d=0.45",
        *audio_filter, "-f", "null", "-",
    ], check=True, capture_output=True, text=True)
    format_ok = (
        (video["width"], video["height"]) == (cfg.render.width, cfg.render.height)
        and video.get("avg_frame_rate") == f"{cfg.render.fps}/1"
        and int(video["nb_read_frames"]) == cfg.frame_count
        and video.get("codec_name") == "h264" and video.get("pix_fmt") == "yuv420p"
        and all(video.get(key) == "bt709" for key in ("color_primaries", "color_transfer", "color_space"))
    )
    audio_ok = (
        audio.get("codec_name") == "aac"
        and int(audio.get("sample_rate", 0)) == cfg.audio.sample_rate_hz
        and audio.get("channels") == cfg.audio.channels
        and abs(float(audio.get("duration", 0)) - cfg.world.duration_s) <= 1 / cfg.render.fps
        and abs(float(audio.get("start_time", 0)) - float(video.get("start_time", 0))) <= 1 / cfg.render.fps
    )
    peaks = [float(value) for value in re.findall(
        r"Peak level dB:\s*(-inf|[+-]?\d+(?:\.\d+)?)", inspected.stderr
    )]
    peak_dbfs = max(peaks) if peaks else None
    clipping_ok = peak_dbfs is not None and peak_dbfs < -.05
    with Path(video_path).open("rb") as handle:
        sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "work_run_id": state.run_id, "sha256": sha256,
        "machine_checks": {"format": format_ok, "audio_timeline": audio_ok, "full_decode": True,
                           "audio_peak_has_headroom": clipping_ok},
        "visual_review": {"status": "sampled" if visual_notes else "pending", "notes": visual_notes},
        "audio_review": {"status": "sampled" if audio_notes else "pending", "notes": audio_notes},
        "artistic_approval": "pending",
        "review_candidates": frame_candidates(inspected.stderr),
        "audio_decoded_peak_dbfs": peak_dbfs if peak_dbfs != float("-inf") else None,
        "streams": {"video": video, "audio": audio},
    }
