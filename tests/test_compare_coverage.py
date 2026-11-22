from dataclasses import replace
from pathlib import Path

import numpy as np

from mower_pov.config import load_config
from mower_pov.contracts import Terrain, Trajectory


def _load_metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import compare_coverage

    return compare_coverage


def test_cutter_visits_count_disjoint_physical_passes(monkeypatch):
    metrics = _load_metrics(monkeypatch)
    config = load_config("configs/mowing-course.toml")
    config = replace(config, mower=replace(config.mower, blade_radius_m=.2, blade_xy_m=(.1, 0.)))
    terrain = Terrain(np.zeros((1, 1)), np.array([0., 0.]), 1., np.array([[True]]),
                      (0., 0., 0., 0.))
    # The offset disc first crosses the cell, dwells through an on/off cycle,
    # then crosses twice between samples. The last crossing is blade-off.
    x = np.array([-.6, .4, .4, .4, 1.4, -.6, 1.4])
    zero = np.zeros(len(x))
    trajectory = Trajectory(
        np.arange(len(x), dtype=float), np.column_stack((x, zero + .5, zero)),
        np.broadcast_to(np.eye(3), (len(x), 3, 3)), zero, zero, zero, zero, zero,
        np.zeros((len(x), 2)), np.array([True, True, False, True, True, False, False]),
    )
    np.testing.assert_array_equal(metrics.cutter_visits(config, terrain, trajectory), [[2]])


def test_travel_bins_preserve_distance_and_attribute_boundary_cuts(monkeypatch):
    metrics = _load_metrics(monkeypatch)
    lengths, counts = metrics.travel_bins(2.5, np.array([0., 2., 2.5]), 1.)

    np.testing.assert_array_equal(lengths, [1., 1., .5])
    np.testing.assert_array_equal(counts, [1, 0, 2])
