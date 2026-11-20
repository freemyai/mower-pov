"""Content-addressed, independently loadable work state."""

import errno
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from zipfile import BadZipFile

import numpy as np

from .config import config_from_dict, validate_config
from .contracts import BladeRoots, CutState, RunBundle, RunConfig, Terrain, Trajectory
from .cutting import compute_cut_state
from .events import build_events, events_from_json, events_to_json
from .motion import integrate_motion, make_demo_schedule, validate_swept_body
from .terrain import build_terrain, generate_blade_roots

SCHEMA_VERSION = 1
ARRAY_FILES = {"terrain.npz", "trajectory.npz", "grass_state.npz"}
BUNDLE_FILES = ARRAY_FILES | {"config.json", "events.json"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_bytes(content):
    return hashlib.sha256(content).hexdigest()


def _hash_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compute_run_id(inputs: Mapping[str, object]) -> str:
    normalized = {key: value for key, value in inputs.items() if key != "report_timestamp"}
    return _hash_bytes(_json(normalized).encode())


def _record_arrays(record, prefix=""):
    return {prefix + field.name: np.asarray(getattr(record, field.name)) for field in fields(record)}


def _datasets(terrain, trajectory, roots, state):
    return {
        "terrain.npz": _record_arrays(terrain),
        "trajectory.npz": _record_arrays(trajectory),
        "grass_state.npz": {**_record_arrays(roots, "roots_"), **_record_arrays(state, "state_")},
    }


def _array_identity(data):
    return {name: {"sha256": _hash_bytes(np.ascontiguousarray(value).tobytes()),
                   "dtype": value.dtype.str, "shape": list(value.shape)}
            for name, value in data.items()}


def _code_identity():
    src = Path(__file__).resolve().parent
    files = {str(path.relative_to(src)): _hash_file(path) for path in src.rglob("*.py")}
    return compute_run_id(files)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).resolve().parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(map(_freeze, value))
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return list(map(_thaw, value))
    return value


def _record(record_type, arrays, prefix=""):
    data = {field.name: arrays[prefix + field.name] for field in fields(record_type)}
    for name, value in data.items():
        if value.ndim == 0:
            data[name] = value.item()
        elif name == "obstacle_xyxy_m":
            data[name] = tuple(value.tolist())
        else:
            value.flags.writeable = False
    return record_type(**data)


def prepare_run(config: RunConfig, base_dir: Path = Path("runs"),
                environment: Mapping | None = None) -> RunBundle:
    validate_config(config)
    terrain = build_terrain(config)
    trajectory = integrate_motion(config, terrain, make_demo_schedule(config, terrain))
    violations = validate_swept_body(config, terrain, trajectory)
    if violations:
        first = min(violations, key=lambda v: v.time_s)
        raise ValueError(f"invalid demo at {first.time_s:.3f}s: {first.reason}")
    roots = generate_blade_roots(config, terrain)
    state = compute_cut_state(config, terrain, trajectory, roots)
    events = build_events(config, trajectory, state, roots)
    data = _datasets(terrain, trajectory, roots, state)
    build = dict(environment or {})
    toolchain = {name: build.get(name) for name in (
        "blender", "ffmpeg", "ffprobe", "verified_device", "cycles_devices", "nvidia_smi")}
    # Install paths and free disk space are provenance, independent of state identity.
    for name in ("blender", "ffmpeg", "ffprobe"):
        if toolchain[name]:
            toolchain[name] = {k: v for k, v in toolchain[name].items() if k != "path"}
    identity = {
        "schema_version": SCHEMA_VERSION, "source_sha256": _code_identity(),
        "config": asdict(config), "arrays": {name: _array_identity(value) for name, value in data.items()},
        "events_sha256": _hash_bytes(events_to_json(events).encode()),
        "python": platform.python_version(), "numpy": np.__version__, "toolchain": toolchain,
    }
    run_id = compute_run_id(identity)
    manifest = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "identity": identity,
                "git_commit": _git_commit(), "environment": build, "assets": []}
    return RunBundle(
        run_id, Path(base_dir) / run_id, config, _record(Terrain, data["terrain.npz"]),
        _record(Trajectory, data["trajectory.npz"]), _record(BladeRoots, data["grass_state.npz"], "roots_"),
        _record(CutState, data["grass_state.npz"], "state_"), events, _freeze(manifest),
    )


def save_bundle(bundle: RunBundle) -> Path:
    target = bundle.run_dir
    if target.exists():
        if load_bundle(target).run_id != bundle.run_id:
            raise ValueError(f"existing bundle has a different identity: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete directory in one rename, including on concurrent prepares.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=target.parent) as staging:
        path = Path(staging)
        for name, arrays in _datasets(bundle.terrain, bundle.trajectory, bundle.roots, bundle.cut_state).items():
            np.savez_compressed(path / name, **arrays)
        (path / "config.json").write_text(_json(asdict(bundle.resolved_config)), encoding="utf8")
        (path / "events.json").write_text(events_to_json(bundle.events), encoding="utf8")
        manifest = _thaw(bundle.manifest)
        manifest["files"] = {name: _hash_file(path / name) for name in sorted(BUNDLE_FILES)}
        (path / "manifest.json").write_text(_json(manifest), encoding="utf8")
        if load_bundle(path).run_id != bundle.run_id:
            raise ValueError("bundle identity changed during preparation")
        try:
            os.rename(path, target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or load_bundle(target).run_id != bundle.run_id:
                raise
    return target


def _read_text(path):
    try:
        return path.read_text(encoding="utf8")
    except OSError as exc:
        raise ValueError(f"missing bundle file: {path.name}") from exc


def load_bundle(run_dir: Path) -> RunBundle:
    path = Path(run_dir)
    manifest = json.loads(_read_text(path / "manifest.json"))
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != SCHEMA_VERSION):
        raise ValueError("unsupported bundle schema")
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != BUNDLE_FILES:
        raise ValueError("bundle files missing or unknown")
    identity = manifest.get("identity")
    if (not isinstance(identity, dict) or identity.get("schema_version") != SCHEMA_VERSION
            or not isinstance(identity.get("arrays"), dict)
            or set(identity["arrays"]) != ARRAY_FILES
            or "config" not in identity or "events_sha256" not in identity
            or manifest.get("run_id") != compute_run_id(identity)):
        raise ValueError("invalid bundle identity")
    for name, expected in manifest["files"].items():
        if not (path / name).is_file() or _hash_file(path / name) != expected:
            raise ValueError(f"missing or corrupt bundle file: {name}")

    arrays = {}
    for name in ARRAY_FILES:
        try:
            with np.load(path / name, allow_pickle=False) as data:
                arrays[name] = dict(data)
        except (ValueError, OSError, BadZipFile) as exc:
            raise ValueError(f"invalid bundle array file: {name}") from exc
        if _array_identity(arrays[name]) != identity["arrays"].get(name):
            raise ValueError(f"unexpected bundle arrays in {name}")

    config = config_from_dict(json.loads(_read_text(path / "config.json")))
    event_text = _read_text(path / "events.json")
    if _json(asdict(config)) != _json(identity["config"]) or _hash_bytes(event_text.encode()) != identity[
        "events_sha256"
    ]:
        raise ValueError("bundle content disagrees with identity")
    try:
        return RunBundle(manifest["run_id"], path, config, _record(Terrain, arrays["terrain.npz"]),
                         _record(Trajectory, arrays["trajectory.npz"]),
                         _record(BladeRoots, arrays["grass_state.npz"], "roots_"),
                         _record(CutState, arrays["grass_state.npz"], "state_"),
                         events_from_json(event_text), _freeze(manifest))
    except (KeyError, TypeError) as exc:
        raise ValueError("bundle is missing required data") from exc
