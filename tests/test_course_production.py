import importlib.util
from pathlib import Path
import shutil
import wave

import numpy as np
import pytest
from PIL import Image


spec = importlib.util.spec_from_file_location(
    "render_dynamic_feature", Path(__file__).resolve().parents[1] / "scripts/render_dynamic_feature.py")
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires FFmpeg")
def test_course_media_uses_bundle_fps_and_absolute_audio_samples(tmp_path):
    batch = tmp_path / "batch-0"
    batch.mkdir()
    for frame in range(12, 24):
        Image.new("RGB", (64, 36), (54 + frame, 100, 130)).save(batch / f"frame-{frame:06d}.png")

    samples = np.arange(48_000) / 48_000
    phase = 2 * np.pi * (106 * samples + 164 * samples**2)
    pcm = np.column_stack((np.sin(phase), np.cos(phase))) * 8_000
    with wave.open(str(tmp_path / "timeline.wav"), "wb") as wav:
        wav.setparams((2, 2, 48_000, 48_000, "NONE", "not compressed"))
        wav.writeframes(pcm.astype("<i2").tobytes())

    report = production.build_media(tmp_path, [batch], (12, 24), (64, 36), "course", pace=False, fps=24)
    assert report["duration_s"] == 0.5
    assert report["audio_start_sample"] == 24_000
    assert report["audio_end_sample"] == 48_000
    alignment = production.audit_audio_alignment(tmp_path, report)
    assert alignment["best_source_lag_samples"] == 0
    assert alignment["normalized_pcm_correlation"] >= 0.98
