import wave
from dataclasses import replace

import numpy as np
import pytest

from mower_pov.audio import event_envelope, mix_audio, synthesize_audio
from mower_pov.cache import prepare_run
from mower_pov.config import load_config
from mower_pov.contracts import Event, EventLog


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    config = load_config("configs/smoke.toml", {"grass.roots_per_m2": 25})
    return prepare_run(config, tmp_path_factory.mktemp("audio"))


def test_audio_length_and_channels(demo, tmp_path):
    output = synthesize_audio(demo, tmp_path / "yard.wav")
    with wave.open(str(output), "rb") as f:
        assert f.getnframes() == 1_920_000
        assert f.getframerate() == 48000
        assert f.getnchannels() == 2


def test_blade_off_removes_blade_component(demo):
    trajectory = replace(demo.trajectory, blade_enabled=np.zeros_like(demo.trajectory.blade_enabled))
    silent_blade = mix_audio(replace(demo, trajectory=trajectory))
    assert not np.any(silent_blade.blade)
    assert np.any(silent_blade.drive)


def test_events_are_sample_aligned():
    event = Event(5.347, "cut_load", (0., 0., 0.), .02)
    rate = 48_000
    envelope = event_envelope(EventLog((event,)), rate, 6 * rate, "cut_load")
    first_nonzero = int(np.flatnonzero(envelope)[0])
    assert abs(first_nonzero - event.time_s * rate) <= 1


def test_mix_has_headroom_and_no_nan(demo):
    mix = mix_audio(demo).master
    assert np.isfinite(mix).all()
    assert 0.01 < np.abs(mix).max() < .92
    assert np.max(np.abs(mix[:, 0] - mix[:, 1])) > .002


def test_sampled_cut_load_preserves_event_intensity():
    log = EventLog((Event(.12, "cut_load", (0., 0., 0.), .031),
                    Event(.8, "cut_load", (0., 0., 0.), .003)))
    signal = event_envelope(log, 1000, 1100, "cut_load")
    assert signal[120:400].max() > 5 * signal[800:].max()
