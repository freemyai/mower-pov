"""Launch verified Blender batches and recover frames by render fingerprint."""

import errno
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..cache import _code_identity, compute_run_id, load_bundle
from .blender_scene import PRESETS


@dataclass(frozen=True)
class RenderRequest:
    run_dir: Path
    frame_indices: tuple[int, ...]
    preset: str
    output_dir: Path
    blender: Path
    device: str = "CPU"
    gpu_index: int = 0
    denoise: bool = True
    exr: bool = False
    view: str = "onboard"


@dataclass(frozen=True)
class RenderResult:
    output_dir: Path
    produced_frames: tuple[int, ...]
    failed_frames: tuple[int, ...]
    worker_log: Path
    timings: dict[int, float]
    memory: dict[int, int]
    device: tuple[str, ...]
    artifacts: tuple[Path, ...]


def validate_frame(path: Path, manifest: dict) -> bool:
    expected = manifest.get("files", {}).get(path.name)
    signature = b"\x89PNG\r\n\x1a\n" if path.suffix == ".png" else b"\x76\x2f\x31\x01"
    if not expected:
        return False
    try:
        with path.open("rb") as file:
            if file.read(len(signature)) != signature:
                return False
            file.seek(0)
            return hashlib.file_digest(file, "sha256").hexdigest() == expected
    except OSError:
        return False


def pending_frames(output, frames, stats, suffix, linear):
    records = {record["frame"]: record for record in stats if "frame" in record}
    pending = []
    for frame in frames:
        path = output / f"frame-{frame:06d}{suffix}"
        record = records.get(frame, {})
        valid = validate_frame(path, record)
        if linear:
            valid = valid and validate_frame(path.with_suffix(".exr"), record)
        if not valid:
            pending.append(frame)
    return tuple(pending)


def require_render_space(output, estimate_bytes):
    if shutil.disk_usage(output).free < estimate_bytes:
        raise OSError(errno.ENOSPC, "Insufficient disk space for render request", str(output))


def _stats(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Interrupted writes leave a partial record. Its frame can be regenerated.
            continue
    return records


def _write_manifest(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        json.dump(value, file, indent=2)
        partial = Path(file.name)
    partial.replace(path)


def _render_locked(request, executable, output, settings, manifest, lock_fd):
    frames = request.frame_indices
    manifest_path = output / "request.json"
    worker_log = output / "blender.log"
    stats_path = output / "frames.jsonl"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"output directory belongs to a different render fingerprint: {output}")
    else:
        if any(path.name != ".render.lock" for path in output.iterdir()):
            raise ValueError(f"new render needs an empty output directory: {output}")
        _write_manifest(manifest_path, manifest)
    stats = _stats(stats_path)
    suffix = ".exr" if request.exr else ".png"
    missing = pending_frames(output, frames, stats, suffix, settings["linear"])
    exit_code = 0
    if missing:
        estimate = (48 if settings["linear"] else 4) * 1024**2 * len(missing)
        require_render_space(output, estimate)
        # End any partial log line before the new batch appends further checksums.
        if stats_path.exists():
            with stats_path.open("a") as log:
                log.write("\n")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf8") as args:
            json.dump({**settings, "frames": missing}, args)
            args.flush()
            worker = Path(__file__).with_name("blender_worker.py")
            with worker_log.open("a", encoding="utf8") as log:
                exit_code = subprocess.run([str(executable), "-b", "-t", "5", "--python-exit-code", "1",
                                            "--python", str(worker), "--", args.name],
                                           stdout=log, stderr=subprocess.STDOUT, check=False,
                                           pass_fds=(lock_fd,)).returncode
        stats = _stats(stats_path)
    failed = pending_frames(output, frames, stats, suffix, settings["linear"])
    if exit_code and not failed:
        raise RuntimeError(f"Blender returned {exit_code}. See {worker_log}")
    produced = tuple(frame for frame in frames if frame not in set(failed))
    timings = {r["frame"]: r["render_and_write_s"] for r in stats if "frame" in r}
    memory = {r["frame"]: r["process_peak_rss_kib"] for r in stats if "frame" in r}
    devices = [record["device"] for record in stats if "device" in record]
    images = tuple(output / f"frame-{frame:06d}{suffix}" for frame in produced)
    linear = tuple(path.with_suffix(".exr") for path in images) if settings["linear"] else ()
    logs = tuple(path for path in (manifest_path, worker_log, stats_path) if path.exists())
    return RenderResult(output, produced, failed, worker_log, timings, memory,
                        tuple(devices[-1]) if devices else (), logs + images + linear)


def render_frames(request: RenderRequest) -> RenderResult:
    state = load_bundle(request.run_dir)
    config = state.resolved_config
    frames = request.frame_indices
    if not frames or any(type(i) is not int or not 0 <= i < config.frame_count for i in frames):
        raise ValueError("frame indices are outside the absolute timeline")
    if len(set(frames)) != len(frames):
        raise ValueError("duplicate frame index")
    if request.view not in {"onboard", "showcase"}:
        raise ValueError(f"unknown camera view {request.view}")
    if request.preset not in PRESETS:
        raise ValueError(f"unknown preset {request.preset}")
    executable = Path(shutil.which(str(request.blender)) or request.blender).resolve()
    version = subprocess.check_output([str(executable), "--version"], timeout=30, text=True)
    if not re.search(rf"^Blender {re.escape(config.render.engine_version)}(?:\s|$)", version):
        raise ValueError(f"Blender version differs from bundle pin {config.render.engine_version}")

    output = request.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        "run_dir": str(state.run_dir.resolve()), "output_dir": str(output),
        "preset": request.preset, "frames": list(frames), "device": request.device,
        "gpu_index": request.gpu_index, "denoise": request.denoise, "view": request.view,
        "format": "OPEN_EXR_MULTILAYER" if request.exr else "PNG",
        "linear": request.preset == "final" and request.denoise and not request.exr,
    }
    identity = {k: v for k, v in settings.items() if k not in ("run_dir", "output_dir")}
    build_hash = next((line.split(":", 1)[1].strip() for line in version.splitlines()
                       if line.lstrip().startswith("build hash:")), None)
    identity.update(run_id=state.run_id, source_sha256=_code_identity(),
                    blender=version.splitlines()[0], blender_build=build_hash, fps=config.render.fps)
    manifest = {"fingerprint": compute_run_id(identity), "identity": identity}
    with (output / ".render.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _render_locked(request, executable, output, settings, manifest, lock.fileno())


def parse_frames(spec: str) -> tuple[int, ...]:
    """Comma-separated indices or half-open ranges such as 480:720."""
    frames = []
    for token in spec.split(","):
        first, sep, stop = token.strip().partition(":")
        frames.extend(range(int(first), int(stop)) if sep else (int(first),))
    return tuple(frames)
