"""Probe local tool availability and verify the selected Cycles device."""

import json
import platform
import re
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from .contracts import EnvironmentReport

BLENDER_VERSION = "5.2.4"


def _run(command, timeout=20):
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.returncode == 0, proc.stdout + proc.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def _tool(name, args, explicit_path=None):
    location = str(explicit_path) if explicit_path is not None else shutil.which(name)
    ok, output = _run([location, *args]) if location else (False, "")
    return {"path": location, "available": ok, "version": output.splitlines()[0] if ok else None}


def _cycles_check(path, device):
    script = """
import bpy, json, sys
backend = sys.argv[-1]
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
scene.cycles.samples = 1
names = ['CPU']
if backend != 'CPU':
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = backend
    prefs.refresh_devices()
    devices = [d for d in prefs.devices if d.type == backend]
    if not devices:
        raise RuntimeError('no Cycles devices for ' + backend)
    for d in prefs.devices:
        d.use = d == devices[0]
    names = [devices[0].name]
    scene.cycles.device = 'GPU'
bpy.ops.render.render()
print('MOWER_DEVICE=' + json.dumps(names))
"""
    ok, output = _run([path, "-b", "-t", "4", "--python-exit-code", "1",
                       "--python-expr", script, "--", device], timeout=120)
    marker = next((line for line in output.splitlines() if line.startswith("MOWER_DEVICE=")), None)
    return (json.loads(marker.split("=", 1)[1]), None) if ok and marker else ([], output[-3500:])


def probe_environment(blender_path: Path | None = None, device: str = "CPU",
                      output_dir: Path = Path("."), blender_version: str = BLENDER_VERSION) -> EnvironmentReport:
    blender = _tool("blender", ["--version"], blender_path)
    ffmpeg = _tool("ffmpeg", ["-version"])
    ffprobe = _tool("ffprobe", ["-version"])
    gpu_tool = shutil.which("nvidia-smi")
    ok, gpus = _run([gpu_tool, "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"]) if gpu_tool else (False, "")
    encoders = []
    if ffmpeg["available"]:
        ok_encoders, output = _run([ffmpeg["path"], "-hide_banner", "-encoders"])
        if ok_encoders:
            encoders = [name for name in ("libx264", "aac")
                        if re.search(rf"^\s*[VA][.A-Z]{{5}}\s+{name}\s", output, re.M)]
    warnings = []
    devices = []
    if not blender["available"]:
        warnings.append("Blender executable unavailable")
    elif not re.match(rf"Blender {re.escape(blender_version)}(?:\s|$)", blender["version"]):
        warnings.append(f"Expected Blender {blender_version}")
    else:
        devices, error = _cycles_check(blender["path"], device)
        if error is not None:
            warnings.append(f"Cycles {device} smoke failed: {error}")
    if not ffprobe["available"] or len(encoders) != 2:
        warnings.append("FFmpeg libx264/AAC or ffprobe unavailable")
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < 1024 ** 3:
        warnings.append("Less than 1 GiB available for local smoke output")
    return {
        "os": platform.platform(), "python": sys.version.split()[0],
        "python_dependencies": {"numpy": version("numpy")},
        "blender": blender, "ffmpeg": ffmpeg, "ffprobe": ffprobe, "encoders": encoders,
        "nvidia_smi": gpus.splitlines() if ok else [], "disk_free_bytes": free_bytes,
        "disk_path": str(output_dir.resolve()), "requested_device": device,
        "verified_device": device if devices else None, "cycles_devices": devices,
        "warnings": warnings, "usable": not warnings,
    }
