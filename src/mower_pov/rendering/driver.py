"""Launch a checked Blender batch with an explicit device and timeline."""

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


def render_frames(request: RenderRequest) -> RenderResult:
    state = load_bundle(request.run_dir)
    config = state.resolved_config
    frames = request.frame_indices
    if not frames or any(type(i) is not int or not 0 <= i < config.frame_count for i in frames):
        raise ValueError("frame indices are outside the absolute timeline")
    if len(set(frames)) != len(frames):
        raise ValueError("duplicate frame index")
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
        "gpu_index": request.gpu_index, "denoise": request.denoise,
        "format": "OPEN_EXR_MULTILAYER" if request.exr else "PNG",
    }
    identity = {k: v for k, v in settings.items() if k not in ("run_dir", "output_dir")}
    identity.update(run_id=state.run_id, source_sha256=_code_identity(),
                    blender=version.splitlines()[0], fps=config.render.fps)
    fingerprint = compute_run_id(identity)
    manifest = {"fingerprint": fingerprint, "identity": identity}
    manifest_path = output / "request.json"
    worker_log = output / "blender.log"
    # Treat each request directory as an independent preview. Do not infer completion from existing frames.
    if any(output.iterdir()):
        raise ValueError(f"render request needs an empty output directory: {output}")
    with manifest_path.open("x", encoding="utf8") as file:
        json.dump(manifest, file, indent=2)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf8") as args:
        json.dump(settings, args)
        args.flush()
        worker = Path(__file__).with_name("blender_worker.py")
        with worker_log.open("w", encoding="utf8") as log:
            process = subprocess.run([str(executable), "-b", "-t", "5", "--python-exit-code", "1",
                                      "--python", str(worker), "--", args.name],
                                     stdout=log, stderr=subprocess.STDOUT, check=False)
    stats_path = output / "frames.jsonl"
    stats = [json.loads(line) for line in stats_path.read_text().splitlines()] if stats_path.exists() else []
    completed = {record["frame"] for record in stats if "frame" in record}
    suffix = ".exr" if request.exr else ".png"
    produced = tuple(frame for frame in frames if frame in completed and
                     (output / f"frame-{frame:06d}{suffix}").is_file())
    failed = tuple(frame for frame in frames if frame not in produced)
    if process.returncode and not failed:
        raise RuntimeError(f"Blender returned {process.returncode}. See {worker_log}")
    timings = {record["frame"]: record["render_and_write_s"] for record in stats if "frame" in record}
    memory = {record["frame"]: record["process_peak_rss_kib"] for record in stats if "frame" in record}
    devices = tuple(stats[0]["device"]) if stats else ()
    images = tuple(output / f"frame-{frame:06d}{suffix}" for frame in produced)
    logs = tuple(path for path in (manifest_path, worker_log, stats_path) if path.exists())
    return RenderResult(output, produced, failed, worker_log, timings, memory, devices, logs + images)


def parse_frames(spec: str) -> tuple[int, ...]:
    """Comma-separated indices or half-open ranges such as 480:720."""
    frames = []
    for token in spec.split(","):
        first, sep, stop = token.strip().partition(":")
        frames.extend(range(int(first), int(stop)) if sep else (int(first),))
    return tuple(frames)
