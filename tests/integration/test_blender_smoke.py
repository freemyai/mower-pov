import os
import shutil
from pathlib import Path

import pytest
from PIL import Image

from mower_pov.cache import prepare_run, save_bundle
from mower_pov.config import load_config
from mower_pov.rendering.driver import RenderRequest, render_frames


@pytest.fixture
def blender():
    path = os.environ.get("MOWER_BLENDER") or shutil.which("blender")
    if path is None:
        pytest.skip("Blender unavailable. Set MOWER_BLENDER to verify the scene.")
    return Path(path)


def test_blender_smoke_outputs_valid_frame(blender, tmp_path):
    cfg = load_config("configs/smoke.toml", {"world.duration_s": 4., "grass.roots_per_m2": 25})
    run = save_bundle(prepare_run(cfg, tmp_path / "runs"))
    result = render_frames(RenderRequest(run, (25,), "smoke", tmp_path / "image", blender))
    assert not result.failed_frames
    assert result.produced_frames == (25,)
    assert result.device == ("CPU",)
    assert result.timings[25] > 0 and result.memory[25] > 0
    assert result.output_dir / "frame-000025.png" in result.artifacts
    with Image.open(result.output_dir / "frame-000025.png") as image:
        image.load()
        assert image.size == (640, 360)
        assert max(high - low for low, high in image.convert("RGB").getextrema()) > 64


def test_blender_subframe_state_uses_cached_roots(blender, tmp_path):
    import subprocess
    import textwrap

    cfg = load_config("configs/smoke.toml", {"world.duration_s": 4., "grass.roots_per_m2": 50})
    run = save_bundle(prepare_run(cfg, tmp_path / "runs"))
    src = Path("src").resolve()
    snippet = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(src)!r})
        import bpy
        import numpy as np
        from mower_pov.rendering.blender_scene import build_scene

        shot = build_scene({str(run)!r}, "smoke")
        state = shot.bundle.cut_state
        i = int(np.flatnonzero(np.isfinite(state.root_first_cut_s))[0])
        f = state.root_first_cut_s[i] * shot.scene.render.fps
        mesh = shot.grass.mesh
        location = tuple(mesh.vertices[i].co)
        stable_id = mesh.attributes["root_id"].data[i].value
        shot.apply(f - .01)
        original = mesh.attributes["leaf_scale"].data[i].vector.z / 10
        shot.apply(f + .01)
        cut = mesh.attributes["leaf_scale"].data[i].vector.z / 10
        assert abs(original - shot.bundle.roots.h0_m[i]) < 1e-7
        assert abs(cut - state.cut_height_m) < 1e-7
        assert stable_id == mesh.attributes["root_id"].data[i].value
        assert location == tuple(mesh.vertices[i].co)
        leaves = shot.foliage[-1]
        # Each leaf base must follow the center of its stem ring when the plant bends.
        flower_vertices = np.array([tuple(vertex.co) for vertex in leaves.mesh.vertices[:30]])
        np.testing.assert_allclose(flower_vertices[24], flower_vertices[6:12].mean(axis=0), atol=1e-4)
        np.testing.assert_allclose(flower_vertices[29], flower_vertices[12:18].mean(axis=0), atol=1e-4)
        move = int(leaves.flexibility.argmax())
        assert leaves.flexibility[move] > 0
        first_position = tuple(leaves.mesh.vertices[move].co)
        shot.apply(f + 8.0)
        assert first_position != tuple(leaves.mesh.vertices[move].co)
        shot.apply(f + .01)
        assert first_position == tuple(leaves.mesh.vertices[move].co)
        instances = sum(
            instance.is_instance and instance.parent.name == "Grass from the work-state cache"
            for instance in bpy.context.evaluated_depsgraph_get().object_instances
        )
        assert instances == len(shot.bundle.roots.root_id), instances
        assert shot.yard_life is not None
        animal = shot.yard_life.hedgehog[0]
        shot.apply(35.7)
        first_pose = tuple(animal.location)
        shot.apply(93.4)
        assert first_pose != tuple(animal.location)
        shot.apply(35.7)
        assert first_pose == tuple(animal.location)
        tires = [obj for obj in shot.chassis.children if obj.name.startswith("Drive tire")]
        assert len(tires) == 2
        assert all(obj.location.x == 0 for obj in tires)
        track = shot.bundle.resolved_config.mower.track_m
        assert abs(max(obj.location.y for obj in tires) - track / 2) < 1e-7
        shot.scene.render.use_motion_blur = True
        shot.apply(f + .01)
        assert cut == mesh.attributes["leaf_scale"].data[i].vector.z / 10
    """)
    script = tmp_path / "verify_scene.py"
    script.write_text(snippet)
    proc = subprocess.run([str(blender), "-b", "-t", "4", "--python-exit-code", "1",
                           "--python", str(script)], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout


def test_linear_exr_retains_auxiliary_layers(blender, tmp_path):
    cfg = load_config("configs/smoke.toml", {"world.duration_s": 4., "grass.roots_per_m2": 25})
    run = save_bundle(prepare_run(cfg, tmp_path / "runs"))
    result = render_frames(RenderRequest(run, (25,), "smoke", tmp_path / "raw", blender,
                                         denoise=False, exr=True))
    assert not result.failed_frames, result.worker_log.read_text()
    header = (result.output_dir / "frame-000025.exr").read_bytes()[:8192]
    assert header.startswith(b"\x76\x2f\x31\x01")
    for name in (b"Combined", b"Denoising Albedo", b"Denoising Normal"):
        assert name in header


def test_resume_replaces_one_corrupt_frame_without_touching_its_neighbor(blender, tmp_path):
    cfg = load_config("configs/smoke.toml", {"world.duration_s": 4., "grass.roots_per_m2": 25})
    run = save_bundle(prepare_run(cfg, tmp_path / "runs"))
    req = RenderRequest(run, (25, 26), "smoke", tmp_path / "images", blender)
    first = render_frames(req)
    assert first.produced_frames == (25, 26)
    good = first.output_dir / "frame-000025.png"
    corrupt = first.output_dir / "frame-000026.png"
    stamp = good.stat().st_mtime_ns
    original = corrupt.read_bytes()
    corrupt.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    result = render_frames(req)
    assert result.produced_frames == (25, 26)
    assert good.stat().st_mtime_ns == stamp
    with Image.open(corrupt) as image:
        image.load()
        assert image.size == (640, 360)
    ledger = (req.output_dir / "frames.jsonl").read_text()
    assert ledger.count('"frame": 25,') == 1
    assert ledger.count('"frame": 26,') == 2
