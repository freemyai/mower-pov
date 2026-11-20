"""Build the review contact sheets and measurements for the backyard 8-second clip."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mower_pov.rendering.preview import frame_sources

# Reference crop coordinates at 960 × 540, scaled to the rendered dimensions.
DETAILS = (
    ("near-grass", 315, (290, 314, 671, 486)),
    ("tree-shade", 108, (20, 241, 401, 413)),
    ("distant-grass", 438, (0, 91, 381, 263)),
    ("turn-boundary", 582, (575, 144, 956, 316)),
    ("old-strip", 738, (91, 306, 472, 478)),
)


def _tile(canvas, src, origin, size, label):
    x, y = origin
    canvas.paste(src.resize(size, Image.Resampling.LANCZOS), (x, y))
    ImageDraw.Draw(canvas).text((x + 5, y + size[1] + 4), label, fill="#ebedf1")


def _read_stats(directories):
    data, devices = [], []
    for folder in directories:
        records = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
        devices.append(records[0])
        data.extend(records[1:])
    return data, devices


def _stats(records):
    render = [record["render_and_write_s"] for record in records]
    output = [record["bytes"] for record in records]
    return {
        "frames": len(records),
        "render_seconds_median": float(np.median(render)),
        "render_seconds_p95": float(np.percentile(render, 95)),
        "render_seconds_max": max(render),
        "bytes_median": int(np.median(output)),
        "bytes_max": max(output),
        "process_peak_rss_mib": max(record["process_peak_rss_kib"] for record in records) / 1024,
    }


def create_evidence(root):
    rendered = sorted(root.glob("clip-*"))
    frames = tuple(range(525, 765))
    paths = frame_sources(rendered, frames)
    output = root / "review"
    output.mkdir(exist_ok=True)

    size = (256, 144)
    step = (size[0] + 7, size[1] + 22)
    sampled = [*range(0, 240, 12), 239]
    sheet = Image.new("RGB", (step[0] * 4 + 7, step[1] * 6 + 7), "#14181c")
    thumbnails = []
    for i, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
            thumbnails.append(np.asarray(image.resize((192, 108), Image.Resampling.BOX), dtype=np.float32))
            if i in sampled:
                sequence = sampled.index(i)
                _tile(sheet, image, (7 + sequence % 4 * step[0], 7 + sequence // 4 * step[1]),
                      size, f"Frame {frames[i]} | {frames[i] / 30:.2f} s")
    sheet.save(output / "timeline-contact.jpg", quality=91)

    crops = Image.new("RGB", (781, 196 * len(DETAILS) + 8), "#14181c")
    for i, (label, frame, bounds) in enumerate(DETAILS):
        images = []
        for j, (folder, setting) in enumerate((("raw-png", "Raw"), ("keyframes", "OIDN"))):
            with Image.open(root / folder / f"frame-{frame:06d}.png") as source:
                scale = (source.width / 960, source.height / 540)
                region = tuple(round(value * scale[axis % 2]) for axis, value in enumerate(bounds))
                cut = source.convert("RGB").crop(region).resize((381, 172), Image.Resampling.LANCZOS)
            _tile(crops, cut, (6 + j * 387, 8 + i * 196), cut.size,
                  f"{setting} | {label} | {frame / 30:.2f} s")
            images.append(cut)
        pair = Image.new("RGB", (781, 196), "#14181c")
        _tile(pair, images[0], (6, 4), images[0].size, f"Raw | {label}")
        _tile(pair, images[1], (393, 4), images[1].size, f"OIDN | {label}")
        pair.save(output / f"{label}.png")
    crops.save(output / "raw-vs-oidn.jpg", quality=95)

    preview_stats, devices = _read_stats(rendered)
    boundary_frames = sorted(
        json.loads((folder / "frames.jsonl").read_text().splitlines()[1])["frame"]
        for folder in rendered
    )[1:]
    if boundary_frames:
        seam = Image.new("RGB", (step[0] * 4 + 7, step[1] * len(boundary_frames) + 7), "#14181c")
        for row, first in enumerate(boundary_frames):
            for col, frame in enumerate(range(first - 2, first + 2)):
                with Image.open(paths[frame - frames[0]]) as source:
                    _tile(seam, source.convert("RGB"), (7 + col * step[0], 7 + row * step[1]), size,
                          f"{frame} | {frame / 30:.2f} s")
        seam.save(output / "shard-boundaries.jpg", quality=93)

    benchmark_stats, _ = _read_stats([root / "bench"])
    temporal = np.mean(np.abs(np.diff(thumbnails, axis=0)), axis=(1, 2, 3))
    cuts = len(np.flatnonzero(temporal == 0))
    details = {label: frame for label, frame, _ in DETAILS}
    metrics = {
        "benchmark": _stats(benchmark_stats),
        "benchmark_frame_indices": [record["frame"] for record in benchmark_stats],
        "preview": _stats(preview_stats),
        "devices": [{key: dev[key] for key in ("device", "backend", "blender", "blender_numpy", "scene_setup_s")}
                    for dev in devices],
        "temporal_screen": {
            "mean_abs_rgb_diff_p95_8bit": float(np.percentile(temporal, 95)),
            "max_mean_abs_rgb_diff_8bit": float(np.max(temporal)),
            "exact_duplicate_pairs_at_192px": cuts,
            "at_shard_boundaries": {
                str(i): float(temporal[i - frames[0] - 1]) for i in boundary_frames},
        },
        "comparison_crops": details,
    }
    (output / "measurements.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    create_evidence(parser.parse_args().run)
