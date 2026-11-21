"""Background Blender entry point for a batch of absolute frame indices."""

import hashlib
import json
import os
import resource
import sys
import time
from tempfile import TemporaryDirectory
from pathlib import Path

import bpy

# Blender's Python uses its bundled libraries, with only our own source added.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mower_pov.rendering.blender_scene import build_scene


def configure_device(scene, backend, gpu_index):
    if backend == "CPU":
        return ["CPU"]
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = backend
    prefs.refresh_devices()
    available = [device for device in prefs.devices if device.type == backend]
    if not 0 <= gpu_index < len(available):
        raise RuntimeError(f"{backend} index {gpu_index} unavailable, found {len(available)}")
    selected = available[gpu_index]
    for device in prefs.devices:
        device.use = device == selected
    scene.cycles.device = "GPU"
    scene.cycles.denoising_use_gpu = True
    return [selected.name, selected.id]


def _linear_output(scene, directory):
    tree = bpy.data.node_groups.new("Linear render evidence", "CompositorNodeTree")
    scene.compositing_node_group = tree
    render = tree.nodes.new("CompositorNodeRLayers")
    file = tree.nodes.new("CompositorNodeOutputFile")
    file.directory = str(directory)
    file.file_name = "linear"
    file.format.file_format = "OPEN_EXR_MULTILAYER"
    file.format.color_depth = "16"
    file.format.exr_codec = "ZIP"
    file.save_as_render = False
    for input_name, channel_name, kind in (
        ("Noisy Image", "RawBeauty", "RGBA"),
        ("Denoising Albedo", "Albedo", "RGBA"),
        ("Denoising Normal", "Normal", "VECTOR"),
    ):
        file.file_output_items.new(kind, channel_name)
        tree.links.new(render.outputs[input_name], file.inputs[channel_name])
    return file


def file_sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()



def main():
    settings = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    started = time.perf_counter()
    replay = build_scene(settings["run_dir"], settings["preset"], yard=settings["yard"],
                         asset_digest=settings["asset_digest"])
    scene = replay.scene
    showcase = None
    if settings["view"] == "showcase":
        from mower_pov.rendering.blender_showcase import make_showcase

        showcase = make_showcase(bpy, replay)
    device = configure_device(scene, settings["device"], settings["gpu_index"])
    scene.cycles.use_denoising = settings["denoise"]
    scene.cycles.seed = replay.bundle.resolved_config.seed % 2147483647
    scene.cycles.use_animated_seed = False
    scene.render.use_persistent_data = settings["device"] != "CPU"
    if settings["format"] == "OPEN_EXR_MULTILAYER":
        scene.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
    scene.render.image_settings.file_format = settings["format"]
    scene.render.image_settings.color_mode = "RGB" if settings["format"] == "PNG" else "RGBA"
    if settings["format"] != "PNG":
        scene.render.image_settings.color_depth = "16"
        scene.render.image_settings.exr_codec = "ZIP"
    output = Path(settings["output_dir"])
    suffix = ".png" if settings["format"] == "PNG" else ".exr"
    with (output / "frames.jsonl").open("a", encoding="utf8") as log, TemporaryDirectory(
        prefix=".linear-", dir=output,
    ) as aux:
        aux = Path(aux)
        if settings["linear"]:
            _linear_output(scene, aux)
        def record(data):
            print(json.dumps(data, allow_nan=False), file=log, flush=True)

        record({"device": device, "backend": settings["device"], "pid": os.getpid(),
                "blender": bpy.app.version_string, "blender_numpy": __import__("numpy").__version__,
                "scene_setup_s": time.perf_counter() - started})
        for frame in settings["frames"]:
            start = time.perf_counter()
            replay.apply(frame)
            if showcase is not None:
                showcase.apply(frame / scene.render.fps)
            updated = time.perf_counter()
            dst = output / f"frame-{frame:06d}{suffix}"
            tmp = dst.with_name(f".{dst.stem}.pending{suffix}")
            scene.render.filepath = str(tmp)
            bpy.ops.render.render(write_still=True)
            if not tmp.is_file() or tmp.stat().st_size < 100:
                raise RuntimeError(f"Blender did not write frame {frame}")
            files = [dst]
            if settings["linear"]:
                linear = list(aux.glob("*.exr"))
                if len(linear) != 1:
                    raise RuntimeError(f"missing or ambiguous linear output for frame {frame}: {linear}")
                sidecar = dst.with_suffix(".exr")
                os.replace(linear[0], sidecar)
                files.append(sidecar)
            os.replace(tmp, dst)
            record({"frame": frame, "absolute_s": frame / scene.render.fps,
                    "state_update_s": updated - start,
                    "render_and_write_s": time.perf_counter() - updated,
                    "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "bytes": dst.stat().st_size,
                    "files": {path.name: file_sha256(path) for path in files}})


if __name__ == "__main__":
    main()
