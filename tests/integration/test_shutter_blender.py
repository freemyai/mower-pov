import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from mower_pov.cache import prepare_run, save_bundle
from mower_pov.config import load_config


@pytest.fixture
def blender():
    executable = os.environ.get("MOWER_BLENDER") or shutil.which("blender")
    if executable is None:
        pytest.skip("Blender unavailable. Set MOWER_BLENDER to verify the scene.")
    return executable


def _run_blender(blender, script, source):
    script.write_text(textwrap.dedent(source))
    proc = subprocess.run([str(blender), "-b", "-t", "3", "--python-exit-code", "1",
                           "--python", str(script)], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout


def test_temporal_render_accumulates_linear_light_before_display_transform(blender, tmp_path):
    src = Path("src").resolve()
    _run_blender(blender, tmp_path / "check_exposure.py", f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(src)!r})
        import bpy
        import numpy as np
        from mower_pov.rendering.shutter import BlenderShutter, Exposure

        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        scene.render.resolution_x = 16
        scene.render.resolution_y = 16
        scene.render.dither_intensity = 0
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.exposure = .14
        world = bpy.data.worlds.new("Uniform emitter")
        world.use_nodes = True
        scene.world = world
        color = world.node_tree.nodes.get("Background").inputs["Color"]
        camera = bpy.data.objects.new("Test camera", bpy.data.cameras.new("Test camera"))
        scene.collection.objects.link(camera)
        scene.camera = camera

        output = Path({str(tmp_path)!r})
        work = output / "shutter"
        work.mkdir()
        frames = Exposure(8, 3).sample_frames(607, 30, 40)
        raw_values = [.25, .75, 4.]
        actual_frames = []

        def replay(frame):
            actual_frames.append(frame)
            value = raw_values[frames.index(frame)]
            color.default_value = (value, value, value, 1)

        temporal = BlenderShutter(scene, work, png=True, linear=False)
        path = output / "integrated.png"
        temporal.render(607, frames, replay, path)
        assert np.allclose(actual_frames, [606.92, 607, 607.08])

        reference = output / "reference.png"
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '8'
        scene.render.filepath = str(reference)
        scene.frame_set(607)
        mean = sum(raw_values) / len(raw_values)
        color.default_value = (mean, mean, mean, 1)
        bpy.ops.render.render(write_still=True)

        integrated_image = bpy.data.images.load(str(path))
        reference_image = bpy.data.images.load(str(reference))
        assert integrated_image.size[:] == (16, 16)
        assert path.read_bytes()[24] == 8
        np.testing.assert_allclose(integrated_image.pixels[:], reference_image.pixels[:], atol=1/255)
    """)


def test_shutter_replays_vehicle_actor_and_cut_grass_at_render_time(blender, tmp_path):
    cfg = load_config("configs/smoke.toml", {
        "world.duration_s": 4., "grass.roots_per_m2": 25,
        "render.width": 96, "render.height": 54, "render.samples": 2,
    })
    run = save_bundle(prepare_run(cfg, tmp_path / "bundles"))
    src = Path("src").resolve()
    _run_blender(blender, tmp_path / "check_scene_state.py", f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(src)!r})
        import bpy
        import numpy as np
        from mower_pov.rendering.blender_scene import build_scene
        from mower_pov.rendering.shutter import BlenderShutter, Exposure

        shot = build_scene({str(run)!r}, "final")
        scene = shot.scene
        shot.camera.data.dof.use_dof = False
        scene.cycles.use_denoising = False
        grass = shot.grass.mesh
        frame = 61
        exposure = Exposure(8, 3)
        samples = exposure.sample_frames(frame, scene.render.fps, 4)
        cut_frames = shot.bundle.cut_state.root_first_cut_s * scene.render.fps
        root = int(np.flatnonzero((samples[0] < cut_frames) & (cut_frames < samples[1])
                                  & (shot.grass.cut_variants >= 0))[0])
        actor = shot.yard_life.hedgehog[0]
        rendered = []

        def observe_at_render(current_scene):
            if current_scene != scene:
                return
            rendered.append((
                scene.frame_current_final,
                tuple(shot.chassis.matrix_world.translation),
                tuple(actor.location),
                tuple(grass.attributes["leaf_angles"].data[root].vector),
                grass.attributes["leaf_variant"].data[root].value,
                np.asarray(shot.chassis.matrix_world.inverted() @ shot.camera.matrix_world),
            ))

        bpy.app.handlers.render_pre.append(observe_at_render)
        output = Path({str(tmp_path)!r})
        temporal = BlenderShutter(scene, output, png=True, linear=False)
        for f, name in ((frame, "ordered"), (41, "different"), (frame, "repeated")):
            positions = exposure.sample_frames(f, scene.render.fps, 4)
            temporal.render(f, positions, shot.apply, output / f"{{name}}.png")
        assert (output / "ordered.png").read_bytes() == (output / "repeated.png").read_bytes()
        np.testing.assert_allclose([sample[0] for sample in rendered[:3]], samples, atol=1e-5, rtol=0)
        for channel in (1, 2, 3):
            assert len({{record[channel] for record in rendered[:3]}}) == 3
        assert [record[4] for record in rendered[:3]] == [
            shot.grass.original_variants[root], shot.grass.cut_variants[root],
            shot.grass.cut_variants[root],
        ]
        for record in rendered[1:3]:
            np.testing.assert_allclose(record[5], rendered[0][5], atol=1e-6, rtol=0)
    """)
