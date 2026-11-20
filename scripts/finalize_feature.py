"""Assemble and verify the feature film, then write a cover and review contact sheet."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mower_pov.quality import inspect_run
from mower_pov.rendering.driver import _stats
from mower_pov.rendering.preview import assemble_video

REVIEW_FRAMES = (
    (24, "Hedgehog in the shade"), (96, "Mower starts"), (300, "Outbound pass"),
    (470, "Flower bed approach"), (535, "Begin turning"), (597, "Dog and rolling ball"),
    (645, "End of the turn"), (738, "Previous cut strip"), (1002, "Return pass"),
    (1170, "Decreasing wheel speed"),
)


def _extract(video, frame, target):
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-ss", f"{frame/30:.9f}",
        "-i", str(video), "-frames:v", "1", "-update", "1", "-y", str(target),
    ], check=True)


def features(root):
    video = assemble_video(root, root / "backyard-40s.wav")
    review = root / "review"
    review.mkdir(exist_ok=True)
    rows, cols, width, height = 2, 5, 320, 180
    tile_w, tile_h = width + 8, height + 23
    sheet = Image.new("RGB", (cols * tile_w + 8, rows * tile_h + 8), "#14191c")
    label = ImageDraw.Draw(sheet)
    for i, (frame, note) in enumerate(REVIEW_FRAMES):
        path = review / f"frame-{frame:06d}.png"
        _extract(video, frame, path)
        with Image.open(path) as image:
            x = 8 + i % cols * tile_w
            y = 8 + i // cols * tile_h
            sheet.paste(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), (x, y))
            label.text((x + 3, y + height + 4), f"{frame/30:.2f}s  {note}", fill="#f0ece5")
    sheet.save(root / "feature-contact.jpg", quality=93)
    with Image.open(review / "frame-000738.png") as cover:
        cover.convert("RGB").save(root / "mower-pov-cover.jpg", quality=95, subsampling=0)
    work = Path((root / "work-bundle.txt").read_text().strip())
    report = inspect_run(work, video)
    rows = [entry for shard in sorted(root.glob("batch-*")) for entry in _stats(shard / "frames.jsonl")]
    frames = {row["frame"]: row for row in rows if "frame" in row}
    seconds = np.asarray([value["render_and_write_s"] for value in frames.values()])
    report["render"] = {
        "source_commit": (root / "source-commit.txt").read_text().strip(),
        "verified_frame_pairs": len(frames),
        "median_s_per_frame": float(np.median(seconds)),
        "p95_s_per_frame": float(np.percentile(seconds, 95)),
        "peak_worker_rss_mib": max(item["process_peak_rss_kib"] for item in frames.values()) / 1024,
    }
    (root / "feature-qa.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not all(report["machine_checks"].values()):
        raise ValueError("encoded film failed machine QA")
    print(video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production_dir", type=Path)
    features(parser.parse_args().production_dir)
