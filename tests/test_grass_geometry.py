import numpy as np

from mower_pov.rendering.grass_geometry import clip_blades


def test_clipping_retains_lower_strip_and_uncut_short_leaf():
    verts = [(0, -.1, 0), (0, .1, 0), (1, .07, .08), (1, -.07, .08),
             (3, -.02, 0), (3, .02, 0), (3.1, .02, .02), (3.1, -.02, .02)]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
    rgba = [(i / 10., .3, .1, 1.) for i in range(len(verts))]
    clipped, surfaces, colors = clip_blades(verts, faces, rgba, .045)
    assert len(surfaces) == 2
    for i in (0, 1, 4, 5, 6, 7):
        assert colors[clipped.index(verts[i])] == rgba[i]
    tips = [v for v in clipped if v[2] == .045]
    assert len(tips) == 2
    np.testing.assert_allclose([tip[0] for tip in tips], .045 / .08)
    assert max(v[2] for v in clipped) == .045
