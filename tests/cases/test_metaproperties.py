from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from mower_pov.cache import compute_run_id, load_bundle, prepare_run, save_bundle, _thaw
from mower_pov.config import load_config
from mower_pov.cases.contracts import ObstacleSpec
from mower_pov.cases.geometry import swept_obstacle_clearance


def test_planar_clearance_is_invariant_under_rigid_transform():
    layout = ObstacleSpec("slanted", "polyline_sweep", ((-.2, .30), (.24, .58)), .018)
    path = np.column_stack((np.linspace(-.9, .9, 61), np.zeros(61)))
    yaw = np.linspace(-.4, .3, len(path))
    theta = 1.21
    c, s = np.cos(theta), np.sin(theta)
    rotate = np.array([[c, -s], [s, c]])
    shift = np.array([.82, -.19])
    shape = (.50, .42)
    moved = replace(layout, points_xy_m=tuple(map(tuple, np.asarray(layout.points_xy_m) @ rotate.T + shift)))
    original = swept_obstacle_clearance(path, yaw, shape, layout)
    transformed = swept_obstacle_clearance(path @ rotate.T + shift, yaw + theta, shape, moved)
    np.testing.assert_allclose(original, transformed, atol=2e-15, rtol=0)


def test_loading_old_source_identity_preserves_evidence(tmp_path):
    config = load_config(Path(__file__).resolve().parents[2]/"configs/smoke.toml")
    original = prepare_run(config, tmp_path)
    manifest = _thaw(original.manifest)
    manifest["identity"]["source_sha256"] = "a" * 64
    assert manifest["identity"]["config"] == json.loads(json.dumps(asdict(config)))
    old_id = compute_run_id(manifest["identity"])
    manifest["run_id"] = old_id
    path = save_bundle(replace(original, run_id=old_id, run_dir=tmp_path/old_id, manifest=manifest))
    before = (path/"manifest.json").read_bytes()
    loaded = load_bundle(path)
    assert loaded.run_id == old_id
    assert loaded.manifest["identity"]["source_sha256"] == "a" * 64
    assert (path/"manifest.json").read_bytes() == before
