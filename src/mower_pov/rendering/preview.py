"""Assemble and verify a contiguous H.264 review clip from absolute frame indices."""

import json
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path


def frame_sources(directories, frame_indices):
    available = {}
    for directory in map(Path, directories):
        for path in directory.glob("frame-*.png"):
            if path.name in available:
                raise ValueError(f"ambiguous source image {path.name}")
            available[path.name] = path
    sources = []
    for frame in frame_indices:
        name = f"frame-{frame:06d}.png"
        if name not in available:
            raise ValueError(f"missing absolute frame {frame}")
        sources.append(available[name])
    return sources


def _validate_shards(directories, fps):
    expected = None
    for directory in map(Path, directories):
        manifest = directory / "request.json"
        identity = json.loads(manifest.read_text())["identity"] if manifest.is_file() else {}
        if "fps" in identity and identity["fps"] != fps:
            raise ValueError(f"encode fps differs from the source timeline fps ({identity['fps']})")
        stats = directory / "frames.jsonl"
        if identity and "fps" not in identity and stats.is_file():
            # Earlier preview requests recorded the timeline in per-frame settings.
            records = [json.loads(line) for line in stats.read_text().splitlines()[1:]]
            if any(abs(record["frame"] / fps - record["absolute_s"]) > 1e-9 for record in records):
                raise ValueError("encode fps differs from the recorded source timeline fps")
        key = {name: value for name, value in identity.items() if name not in {"frames", "gpu_index", "fps"}}
        if expected is not None and key != expected:
            raise ValueError("frame directories have different render settings or bundle identities")
        expected = key


def _probe(path):
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_streams", "-of", "json", str(path),
    ], text=True)
    return json.loads(output)["streams"][0]


def encode_preview(directories, frame_indices, fps, output: Path):
    """Encode the selected timeline without gaps, preserving absolute indices in the report."""
    if not frame_indices or tuple(frame_indices) != tuple(range(frame_indices[0], frame_indices[-1] + 1)):
        raise ValueError("a preview requires a contiguous ascending absolute frame range")
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    directories = tuple(directories)
    output = Path(output).resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("preview output must be an MP4 file")
    _validate_shards(directories, fps)
    sources = frame_sources(directories, frame_indices)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preview-", dir=output.parent) as temp_dir:
        temp = Path(temp_dir)
        for i, source in enumerate(sources):
            (temp / f"{i:06d}.png").symlink_to(source.resolve())
        video = temp / "preview.mp4"
        # Blender AgX display PNG is sRGB. Convert its signal and range to BT.709.
        transform = ("format=gbrp,zscale=matrixin=gbr:transferin=iec61966-2-1:"
                     "primariesin=709:rangein=full:matrix=709:transfer=709:"
                     "primaries=709:range=limited,format=yuv420p")
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
            "-framerate", str(fps), "-i", str(temp / "%06d.png"),
            "-frames:v", str(len(sources)), "-an", "-vf", transform,
            "-c:v", "libx264", "-crf", "17", "-preset", "slow",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-color_range", "tv", "-movflags", "+faststart", str(video),
        ], check=True)
        data = _probe(video)
        if (int(data["nb_read_frames"]) != len(sources)
                or Fraction(data["avg_frame_rate"]) != fps
                or any(data.get(key) != "bt709" for key in
                       ("color_primaries", "color_transfer", "color_space"))
                or data["pix_fmt"] != "yuv420p"):
            raise ValueError("encoded preview failed media verification")
        os.replace(video, output)
    return {
        "source_first_frame": frame_indices[0], "source_last_frame": frame_indices[-1],
        "source_start_s": frame_indices[0] / fps, "fps": data["avg_frame_rate"],
        "duration_s": float(data["duration"]), "read_frames": int(data["nb_read_frames"]),
        "width": data["width"], "height": data["height"], "codec": data["codec_name"],
        "pixel_format": data["pix_fmt"], "color_transfer": data["color_transfer"],
    }
