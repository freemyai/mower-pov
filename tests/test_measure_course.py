from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np


spec = spec_from_file_location("measure_course", Path(__file__).resolve().parents[1] / "scripts/measure_course.py")
measure_course = module_from_spec(spec)
spec.loader.exec_module(measure_course)


def test_continuous_clearance_detects_crossing_between_poses_and_degenerate_segments():
    start = np.array([[-1., 0.], [1., 2.], [2., 0.]])
    end = np.array([[1., 0.], [1., 2.], [3., 0.]])

    distances = measure_course.segment_distances(start, end, (0., -1.), (0., 1.))
    point_distances = measure_course.segment_distances(start, end, (0., 0.), (0., 0.))

    np.testing.assert_allclose(distances, [0., np.sqrt(2.), 2.])
    np.testing.assert_allclose(point_distances, [0., np.sqrt(5.), 2.])


def test_coverage_uses_nominal_capsules_and_separates_unmowable_cells():
    terrain = SimpleNamespace(origin_xy_m=np.array([0., 0.]), cell_size_m=1.,
                              mowable_mask=np.array([[True, True, False], [True, True, True]]))
    first_cut = np.array([[1., np.inf, np.inf], [3., np.inf, np.inf]])
    rows = [((.5, .5), (2.5, .5))]

    coverage, nominal = measure_course.coverage_metrics(terrain, first_cut, rows, .45)

    np.testing.assert_array_equal(nominal, [[True, True, False], [False, False, False]])
    assert coverage["actual_m2"] == 2.
    assert coverage["actual_in_nominal_m2"] == 1.
    assert coverage["actual_outside_nominal_m2"] == 1.
    assert coverage["nominal_fraction_cut"] == .5


def test_previous_cut_mask_requires_a_cut_at_the_time_of_the_turn():
    first_cut = np.array([2., 3., 4., 5., np.inf])
    visible = np.array([True, False, True, True, True])

    short, tall = measure_course.visible_cut_masks(first_cut, visible, turn_started=3., viewed_at=5.)

    np.testing.assert_array_equal(short, [True, False, False, False, False])
    np.testing.assert_array_equal(tall, [False, False, False, False, True])


def test_camera_normalization_uses_render_aspect_and_preserves_signed_depth():
    cfg = SimpleNamespace(camera=SimpleNamespace(hfov_deg=90.), render=SimpleNamespace(width=100, height=50))
    optical_points = np.array([[.5, .5, 1.], [.1, .1, -1.], [0., 0., 0.]])

    x, y, depth = measure_course.optical_coordinates(cfg, optical_points)

    np.testing.assert_allclose([x[0], y[0]], [.5, 1.])
    np.testing.assert_array_equal(depth, [1., -1., 0.])
    assert np.isfinite(x).all() and np.isfinite(y).all()
