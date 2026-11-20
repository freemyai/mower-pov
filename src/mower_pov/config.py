"""Load TOML or JSON and normalize all configuration to frozen SI records."""

import json
import math
import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Mapping, get_args, get_origin

from .contracts import (
    AudioConfig, CameraConfig, GrassConfig, MowerConfig, RenderConfig, RunConfig, WorldConfig,
)

SECTIONS = dict(world=WorldConfig, mower=MowerConfig, camera=CameraConfig,
                grass=GrassConfig, render=RenderConfig, audio=AudioConfig)


def _read(path, visited):
    path = Path(path).resolve()
    if path in visited:
        raise ValueError(f"cyclic config inheritance: {path}")
    raw = json.loads(path.read_text()) if path.suffix == ".json" else tomllib.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    base = _read(path.parent / parent, visited | {path})
    for key, value in raw.items():
        base[key] = {**base.get(key, {}), **value} if key in SECTIONS else value
    return base


def _typed(value, kind, name):
    if get_origin(kind) is tuple:
        args = get_args(kind)
        if isinstance(value, (list, tuple)) and len(value) == len(args):
            return tuple(_typed(item, item_type, name) for item, item_type in zip(value, args))
    elif kind is float:
        if type(value) in (int, float) and math.isfinite(value):
            return float(value)
    elif type(value) is kind:
        return value
    raise ValueError(f"{name}: invalid value {value!r}")


def config_from_dict(raw):
    if set(raw) != {"seed", *SECTIONS}:
        raise ValueError(f"configuration fields differ: {set(raw) ^ {'seed', *SECTIONS}}")
    sections = {}
    for name, kind in SECTIONS.items():
        expected = {field.name: field.type for field in fields(kind)}
        values = raw[name]
        if not isinstance(values, dict) or values.keys() != expected.keys():
            raise ValueError(f"{name}: expected fields {', '.join(expected)}")
        sections[name] = kind(**{
            key: _typed(values[key], field_type, f"{name}.{key}")
            for key, field_type in expected.items()
        })
    config = RunConfig(_typed(raw["seed"], int, "seed"), **sections)
    validate_config(config)
    return config


def load_config(path: Path, overrides: Mapping[str, object] | None = None) -> RunConfig:
    raw = _read(path, set())
    for key, value in (overrides or {}).items():
        if key == "seed":
            raw[key] = value
            continue
        section, _, field = key.partition(".")
        if section not in SECTIONS or field not in raw.get(section, {}):
            raise ValueError(f"unknown override: {key}")
        raw[section][field] = value
    return config_from_dict(raw)


def validate_config(config: RunConfig) -> None:
    for section, kind in SECTIONS.items():
        data = getattr(config, section)
        for field in fields(kind):
            name = f"{section}.{field.name}"
            value = _typed(getattr(data, field.name), field.type, name)
            if field.type in (int, float) and field.name != "pitch_deg" and value <= 0:
                raise ValueError(f"{name}: must be positive")
    w, m, c, g, r, a = (
        config.world, config.mower, config.camera, config.grass, config.render, config.audio
    )
    x0, y0, x1, y1 = w.flower_bed_xyxy_m
    conditions = {
        "seed": type(config.seed) is int and config.seed >= 0,
        "world.flower_bed_xyxy_m": x0 < x1 and y0 < y1,
        "world.cell_m": w.cell_m <= min(w.width_m, w.depth_m),
        "world.duration_s": all(math.isclose(w.duration_s * rate, round(w.duration_s * rate),
                                               rel_tol=0, abs_tol=1e-8) and
                                round(w.duration_s * rate) > 0
                                for rate in (w.step_hz, r.fps, a.sample_rate_hz)),
        "mower.track_m": m.track_m <= m.width_m,
        "camera.hfov_deg": c.hfov_deg < 180,
        "camera.pitch_deg": abs(c.pitch_deg) < 90,
        "grass.max_height_m": g.max_height_m >= g.min_height_m,
        "render.pixel_format": r.pixel_format == "yuv420p",
        "render.width": r.width % 2 == 0,
        "render.height": r.height % 2 == 0,
        "render.engine_version": bool(r.engine_version.strip()),
        "audio.channels": a.channels == 2,
    }
    for name, valid in conditions.items():
        if not valid:
            raise ValueError(f"{name}: outside supported range")
