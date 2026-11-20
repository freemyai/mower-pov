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


def test_mux_audio_uses_absolute_sample_window(tmp_path):
    import shutil
    import wave

    import numpy as np

    from mower_pov.rendering.preview import mux_audio

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe needed")
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(30, 36):
        Image.new("RGB", (64, 36), (120, 49, 30)).save(frames / f"frame-{i:06d}.png")
    video = tmp_path / "silent.mp4"
    encode_preview([frames], tuple(range(30, 36)), 30, video)
    wav = tmp_path / "long.wav"
    with wave.open(str(wav), "wb") as sound:
        sound.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        samples = np.zeros((2 * 48000, 2), dtype="<i2")
        samples[48000:58000] = 7000
        sound.writeframes(samples.tobytes())
    result = mux_audio(video, wav, 30, 36, 30, tmp_path / "sound.mp4")
    assert result["audio_start_sample"] == 48000
    assert result["audio_channels"] == 2
    assert abs(result["audio_duration_s"] - .2) <= 1 / 30
    with pytest.raises(ValueError, match="fps"):
        mux_audio(video, wav, 30, 36, 24, tmp_path / "wrong.mp4")


def test_color_reference_patches(tmp_path):
    import shutil
    import subprocess

    import numpy as np

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe needed")
    rgb = (120, 49, 30)
    frame = tmp_path / "frame-000000.png"
    Image.new("RGB", (64, 36), rgb).save(frame)
    video = tmp_path / "patch.mp4"
    encode_preview([tmp_path], (0,), 30, video)
    inverse = ("zscale=matrixin=709:transferin=709:primariesin=709:rangein=limited:"
               "matrix=gbr:transfer=iec61966-2-1:primaries=709:range=full,format=gbrp,format=rgb24")
    values = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(video), "-vf", inverse, "-frames:v", "1",
        "-f", "rawvideo", "-",
    ])
    np.testing.assert_allclose(np.frombuffer(values, np.uint8).reshape(-1, 3).mean(axis=0), rgb, atol=3)
