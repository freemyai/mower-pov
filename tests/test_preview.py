import pytest
from PIL import Image

from mower_pov.rendering.preview import encode_preview, frame_sources


def test_frame_sources_use_exact_absolute_range(tmp_path):
    img = tmp_path / "one"
    img.mkdir()
    for frame in [525, 526, 528]:
        (img / f"frame-{frame:06d}.png").touch()
    with pytest.raises(ValueError, match="527"):
        frame_sources([img], tuple(range(525, 529)))
    assert [x.name for x in frame_sources([img], (525, 526))] == [
        "frame-000525.png", "frame-000526.png"
    ]


def test_encoded_preview_matches_source_range(tmp_path):
    import shutil

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe needed for the preview encode")
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    for i in range(525, 529):
        Image.new("RGB", (64, 36), (40 + i - 525, 140, 91)).save(
            image_dir / f"frame-{i:06d}.png")
    path = tmp_path / "segment.mp4"
    report = encode_preview([image_dir], tuple(range(525, 529)), 30, path)
    assert path.is_file()
    assert (report["read_frames"], report["fps"], report["width"]) == (4, "30/1", 64)
    assert report["source_first_frame"] == 525
    assert report["color_transfer"] == "bt709"


def test_encoding_rejects_wrong_duration_and_file_type(tmp_path):
    import json

    folder = tmp_path / "render"
    folder.mkdir()
    (folder / "request.json").write_text(json.dumps({"identity": {"fps": 30}}))
    with pytest.raises(ValueError, match="fps"):
        encode_preview([folder], (525,), 24, tmp_path / "preview.mp4")
    with pytest.raises(ValueError, match="MP4"):
        encode_preview([folder], (525,), 30, tmp_path / "preview.json")
