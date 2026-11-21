"""Synthesize stereo sound from the cached mower trajectory and its event history."""

import wave
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .contracts import EventLog, RunBundle


@dataclass
class AudioMix:
    rate: int
    master: np.ndarray
    drive: np.ndarray
    blade: np.ndarray
    grass: np.ndarray
    yard: np.ndarray


def _sample_index(time, rate):
    return int(np.floor(time * rate + .5))


def event_envelope(events: EventLog, rate: int, frames: int, kind: str) -> np.ndarray:
    """Place causal decays at the nearest audio sample on the shared timeline."""
    t = np.arange(round(.18 * rate), dtype=np.float32) / rate
    kernel = (.05 + .95 * (1 - np.exp(-t / .012))) * np.exp(-t / .069)
    envelope = np.zeros(frames, dtype=np.float32)
    for event in events.events:
        if event.kind != kind or event.intensity <= 0:
            continue
        first = _sample_index(event.time_s, rate)
        overlap = min(len(kernel), frames - first)
        if first >= 0 and overlap > 0:
            envelope[first:first + overlap] += event.intensity * kernel[:overlap]
    return envelope


def _average(signal, width):
    span = np.pad(signal, (width - 1, 0))
    total = np.r_[0., np.cumsum(span, dtype=np.float64)]
    return ((total[width:] - total[:-width]) / width).astype("f")


def _phase(rate, frequency):
    return (np.cumsum(frequency, dtype=np.float64)
            - frequency[0]) * (2 * np.pi / rate)


def _yard_sound(time, noise_l, noise_r, rate):
    stereo = []
    gusts = .65 + .22 * np.sin(2.05 * time + .5) + .13 * np.sin(.48 * time)
    for signal in (noise_l, noise_r):
        canopy = 9 * _average(signal, round(.008 * rate)) - 7 * _average(signal, round(.062 * rate))
        leaves = 2.1 * (_average(signal, round(.0015 * rate)) - _average(signal, round(.012 * rate)))
        stereo.append((.016 * canopy * gusts + .006 * leaves).astype("f"))
    ambient = np.column_stack(stereo)
    for start, base, pan in ((.70, 2630, -.51), (1.11, 3010, -.46), (8.98, 2420, .50),
                             (9.21, 2780, .58), (15.32, 3350, -.65), (27.42, 2290, .33),
                             (33.16, 2765, -.36), (33.47, 3110, -.41)):
        i = _sample_index(start, rate)
        width = min(round(.17 * rate), len(time) - i)
        if width <= 0:
            continue
        t = np.arange(width, dtype=np.float32) / rate
        phi = 2 * np.pi * (base * t - 2600 * t ** 2 + 4600 * t ** 3)
        chirp = .023 * np.sin(np.pi * t / .17) ** 2 * (np.sin(phi) + .14 * np.sin(2 * phi))
        ambient[i:i + width] += chirp[:, None] * np.sqrt(np.array([1 - pan, 1 + pan]) / 2)
    return ambient


def mix_audio(bundle: RunBundle) -> AudioMix:
    cfg, trajectory = bundle.resolved_config, bundle.trajectory
    rate = cfg.audio.sample_rate_hz
    count = cfg.audio_frames
    time = np.arange(count) / rate
    rng = np.random.default_rng([cfg.seed, 1071])
    noise = rng.standard_normal((2, count), dtype=np.float32)
    cut = np.clip(event_envelope(bundle.events, rate, count, "cut_load") * 112, 0, 1.6)
    shock = np.clip(event_envelope(bundle.events, rate, count, "terrain_impact") * .11, 0, 1.)
    wheel = np.column_stack([
        np.interp(time, trajectory.time_s, trace)
        for trace in trajectory.wheel_speeds.T
    ])
    i = np.searchsorted(trajectory.time_s, time, side="right") - 1
    blade_on = trajectory.blade_enabled[i]
    if bundle.story is None:
        rotor = np.clip(_average(blade_on.astype("f"), round(.65 * rate)), 0., 1.)
        rotor_gate = rotor * blade_on
    else:
        rotor = np.interp(time, trajectory.time_s, bundle.story.rotor_fraction)
        rotor_gate = rotor
    rotor_phase = _phase(rate, 29 + 34 * rotor - 2.1 * cut)
    motor = np.empty((count, 2), dtype=np.float32)
    blade = np.empty_like(motor)
    grass = np.empty_like(motor)
    for side in range(2):
        speed = np.abs(wheel[:, side])
        motor_level = np.clip(speed * 3.4, 0, 1.)
        drive_phase = _phase(rate, 83 + 590 * speed)
        transmission = (.58 * np.sin(drive_phase + .21 * np.sin(.5 * drive_phase))
                        + .26 * np.sin(2.01 * drive_phase + .31 * side)
                        + .10 * np.sin(4.0 * drive_phase))
        motor[:, side] = (.081 * motor_level ** .79 * transmission
                          + .048 * shock * _average(noise[side], round(.0017 * rate)))
        rotation = (.58 * np.sin(2 * rotor_phase + side * .06)
                    + .37 * np.sin(rotor_phase + .19 * np.sin(3 * rotor_phase))
                    + .14 * np.sin(5 * rotor_phase + side * .27)
                    + .10 * np.sin(7 * rotor_phase))
        whir = 3.2 * _average(noise[side], round(.003 * rate))
        blade[:, side] = rotor_gate * (.15 * rotation + .044 * whir)
        rasp = .55 * noise[side] + 1.8 * _average(noise[side], round(.0011 * rate))
        grass[:, side] = .044 * rotor_gate * cut * np.tanh(rasp)
    ambient = _yard_sound(time, noise[0], noise[1], rate)
    # Each near-mounted microphone also hears a little of the opposite traction wheel.
    drive = .81 * motor + .19 * motor[:, ::-1]
    master = drive + blade + grass + ambient
    length = min(len(master), round(.075 * rate))
    if length:
        master[-length:] *= np.linspace(1, 0, length)[:, None]
    master *= min(2.25, .86 / max(float(np.max(np.abs(master))), 1e-8))
    return AudioMix(rate, master, drive, blade, grass, ambient)


def synthesize_audio(bundle: RunBundle, output_wav: Path) -> Path:
    """Write a PCM WAV atomically after rendering the complete deterministic mix."""
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    mix = mix_audio(bundle)
    pcm = np.rint(mix.master * 32767).astype("<i2")
    with TemporaryDirectory(prefix=".audio-", dir=output_wav.parent) as directory:
        staged = Path(directory) / "mix.wav"
        with wave.open(str(staged), "wb") as sound:
            sound.setnchannels(2)
            sound.setsampwidth(2)
            sound.setframerate(mix.rate)
            sound.writeframes(pcm.tobytes())
        staged.replace(output_wav)
    return output_wav
