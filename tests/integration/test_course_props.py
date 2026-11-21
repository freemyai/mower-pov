import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


def test_rendered_course_props_fit_the_grounded_collision_footprints(tmp_path):
    blender = os.environ.get("MOWER_BLENDER") or shutil.which("blender")
    if blender is None:
        pytest.skip("Blender unavailable. Set MOWER_BLENDER to verify the scene.")

    source = Path("src").resolve()
    snippet = textwrap.dedent(f"""
        import sys
        from types import SimpleNamespace

        import bpy
        import numpy as np

        sys.path.insert(0, {str(source)!r})
        from mower_pov.config import load_config
        from mower_pov.course_layout import COURSE_OBSTACLES
        from mower_pov.obstacles import distance_to_segment
        from mower_pov.rendering.blender_course import make_course_props
        from mower_pov.terrain import build_terrain, sample_height

        bpy.ops.wm.read_factory_settings(use_empty=True)
        terrain = build_terrain(load_config('configs/backyard.toml'))
        roots = make_course_props(bpy, SimpleNamespace(terrain=terrain))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for root, obstacle in zip(roots, COURSE_OBSTACLES, strict=True):
            vertices = []
            for part in root.children_recursive:
                evaluated = part.evaluated_get(depsgraph)
                vertices.extend(tuple(evaluated.matrix_world @ vertex.co)
                                for vertex in evaluated.to_mesh().vertices)
                evaluated.to_mesh_clear()
            world = np.asarray(vertices)
            points = np.asarray(obstacle.points)
            if len(points) == 1:
                distance = np.linalg.norm(world[:, :2] - points[0], axis=1)
            else:
                distance = np.min([distance_to_segment(world[:, :2], start, end)
                                   for start, end in zip(points[:-1], points[1:])], axis=0)
            relative_height = world[:, 2] - sample_height(terrain, world[:, :2])
            assert distance.max() <= obstacle.radius_m + .0004, (obstacle.name, distance.max())
            assert -.015 < relative_height.min() <= .006, (obstacle.name, relative_height.min())
    """)
    script = tmp_path / "verify_course_props.py"
    script.write_text(snippet)
    result = subprocess.run([blender, "-b", "-t", "2", "--python-exit-code", "1", "--python", str(script)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    assert result.returncode == 0, result.stdout
