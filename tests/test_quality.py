from pathlib import Path

import pytest

from mower_pov.quality import frame_candidates, verify_sequence


def test_missing_frame_is_detected(tmp_path: Path):
    with pytest.raises(ValueError, match="missing"):
        verify_sequence([tmp_path], (0, 1), (640, 360))


def test_normal_stop_is_a_review_candidate():
    candidates = frame_candidates("black_start:0 black_end:0.25\nfreeze_start:38.9\nfreeze_end:39.9")
    assert set(candidates) == {"black_start", "black_end", "freeze_start", "freeze_end"}
    assert candidates["freeze_start"] == [38.9]


def test_unreviewed_is_not_passed(tmp_path):
    import shutil
    import wave

    import numpy as np
    from PIL import Image

    from mower_pov.cache import prepare_run, save_bundle
    from mower_pov.config import load_config
    from mower_pov.quality import inspect_run
    from mower_pov.rendering.preview import encode_preview, mux_audio

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe needed")
    config = load_config("configs/smoke.toml", {
        "world.duration_s": .2, "render.width": 64, "render.height": 36, "grass.roots_per_m2": 10,
    })
    bundle = save_bundle(prepare_run(config, tmp_path / "cache"))
    for index in range(6):
        Image.new("RGB", (64, 36), (40 + index * 7, 142, 91)).save(tmp_path / f"frame-{index:06d}.png")
    silent = tmp_path / "silent.mp4"
    encode_preview([tmp_path], tuple(range(6)), 30, silent)
    wav = tmp_path / "sound.wav"
    with wave.open(str(wav), "wb") as sound:
        samples = (.12 * np.sin(np.arange(9600) / 48000 * 2 * np.pi * 470) * 32767).astype("<i2")
        sound.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        sound.writeframes(np.repeat(samples, 2).tobytes())
    video = tmp_path / "tested.mp4"
    mux_audio(silent, wav, 0, 6, 30, video)
    report = inspect_run(bundle, video)
    assert all(report["machine_checks"].values())
    assert report["visual_review"]["status"] == "pending"
    assert report["audio_review"]["status"] == "pending"
    assert report["artistic_approval"] == "pending"
