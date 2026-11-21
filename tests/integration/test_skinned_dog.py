import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("config,start", [("backyard", 20.2), ("backyard-dynamic", 10.1)])
def test_skinned_dog_deforms_one_mesh_and_holds_ground_contacts(tmp_path, config, start):
    blender = os.environ.get("MOWER_BLENDER") or shutil.which("blender")
    if blender is None:
        pytest.skip("Blender unavailable. Set MOWER_BLENDER to verify the dog rig.")
    src = Path("src").resolve()
    script = tmp_path / "check_dog.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(src)!r})
        import bmesh
        import bpy
        import numpy as np
        from mower_pov.actors.behavior import DOG_TRACK, story_tracks, tracks_from_rows
        from mower_pov.actors.gait import DOG_LEGS, evaluate_dog
        from mower_pov.config import load_config
        from mower_pov.rendering.blender_dog import make_dog
        from mower_pov.terrain import build_terrain, sample_height

        config = load_config({f"configs/{config}.toml"!r})
        track = (DOG_TRACK if config.scenario == 'legacy'
                 else tracks_from_rows(story_tracks(config.scenario))[0])
        terrain = build_terrain(config)
        dog = make_dog(bpy, 42)
        skin = bmesh.new()
        skin.from_mesh(dog.surface.data)
        assert all(edge.is_manifold for edge in skin.edges)
        seen = set()
        pending = [next(iter(skin.verts))]
        while pending:
            vertex = pending.pop()
            if vertex in seen:
                continue
            seen.add(vertex)
            pending.extend(edge.other_vert(vertex) for edge in vertex.link_edges)
        assert len(seen) == len(skin.verts), 'the dog skin must be one welded mesh'
        skin.free()
        assert all(abs(sum(g.weight for g in v.groups) - 1.) < 1e-6
                   for v in dog.surface.data.vertices)

        first = None
        for time in np.linspace({start!r}, {start + .1!r}, 6):
            pose = evaluate_dog(time, terrain, track)
            dog.apply(pose)
            bpy.context.view_layer.update()
            evaluated = dog.surface.evaluated_get(bpy.context.evaluated_depsgraph_get())
            vertices = np.array([tuple(evaluated.matrix_world @ v.co)
                                 for v in evaluated.data.vertices])
            for spec in DOG_LEGS:
                target = pose.legs[spec.name]
                upper = dog.rig.pose.bones[spec.name + '.upper']
                lower = dog.rig.pose.bones[spec.name + '.lower']
                np.testing.assert_allclose(upper.head, target.hip, atol=1e-6)
                np.testing.assert_allclose(lower.tail, target.ankle, atol=1e-6)
                assert abs((upper.tail - upper.head).length - spec.upper) < 1e-6
                assert abs((lower.tail - lower.head).length - spec.lower) < 1e-6
                if target.planted:
                    xy = np.array(target.foot_world[:2])
                    contact = vertices[np.linalg.norm(vertices[:, :2] - xy, axis=1) < .027]
                    assert len(contact)
                    assert abs((contact[:, 2] - sample_height(terrain, contact[:, :2])).min()) < .005
            group = dog.surface.vertex_groups['front_right.paw'].index
            sole = np.array([v.index for v in dog.surface.data.vertices if v.co.z < .003
                             and any(g.group == group and g.weight > .999 for g in v.groups)])
            assert len(sole) > 4 and pose.legs['front_right'].planted
            if first is None:
                first = vertices.copy()
            else:
                assert np.linalg.norm(vertices[sole] - first[sole], axis=1).max() < .002
        assert np.linalg.norm(vertices - first, axis=1).max() > .035
    """))
    result = subprocess.run([str(blender), "-b", "-t", "4", "--python-exit-code", "1", "--python", str(script)],
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
