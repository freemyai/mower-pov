"""Atomic per-case files and explicit manifest verification."""

from dataclasses import asdict, dataclass
from hashlib import file_digest
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..cache import _datasets
from ..events import events_to_json
from .resolve import canonical_digest


class CorruptEvidence(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactManifest:
    request_id: str
    files: dict[str, str]


def atomic_write(path, write, mode="w"):
    # Write beside the destination for an atomic same-filesystem rename.
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, mode) as stream:
            write(stream)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, data):
    atomic_write(path, lambda out: json.dump(data, out, sort_keys=True, indent=2, allow_nan=False))


def hash_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_trace(trace, directory):
    write_json(directory/"trace.json", {
        "state_id": trace.state_id, "scenario_id": trace.scenario_id, "status": trace.status,
        "reason": trace.reason, "config": asdict(trace.config), "planned_world_ok": trace.planned_world_ok,
        "events_sha256": trace.events_sha256,
    })
    if trace.trajectory is not None:
        data = _datasets(trace.terrain, trace.trajectory, trace.roots, trace.cut_state, trace.story)
        flattened = {f"{name[:-4]}/{key}":array for name, arrays in data.items() for key, array in arrays.items()}
        atomic_write(directory/"trace.npz", lambda stream: np.savez_compressed(stream, **flattened), "wb")
        atomic_write(directory/"events.json", lambda stream: stream.write(events_to_json(trace.events)))


def save_episode(trace, verdict, directory):
    # The worker checkpoints the trace before running a potentially expensive judge.
    if not (directory / "trace.json").exists():
        write_trace(trace, directory)
    write_json(directory/"verdict.json", asdict(verdict))


def seal(directory, request_id):
    files = {p.name:hash_file(p) for p in sorted(directory.iterdir()) if p.is_file() and not p.name.startswith(".")}
    manifest = ArtifactManifest(request_id, files)
    write_json(directory/"manifest.json", asdict(manifest))
    return manifest


def verify_episode(directory, request_id):
    try:
        manifest = json.loads((directory/"manifest.json").read_text())
        if manifest["request_id"] != request_id or not isinstance(manifest["files"], dict):
            raise CorruptEvidence(f"invalid request identity in {directory}")
        files = manifest["files"]
        if "verdict.json" not in files:
            raise CorruptEvidence(f"missing verdict.json in {directory}")
        names = {p.name for p in directory.iterdir() if p.name != "manifest.json"}
        if names != set(files):
            raise CorruptEvidence(f"artifact inventory differs from manifest in {directory}")
        for name, expected in files.items():
            path = directory/name
            if Path(name).name != name or path.is_symlink() or not path.is_file() or hash_file(path) != expected:
                raise CorruptEvidence(f"missing or damaged artifact {name} in {directory}")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CorruptEvidence(f"invalid episode manifest in {directory}") from exc
    return canonical_digest(manifest)
