import json

import numpy as np
import pytest

from mower_pov.cache import compute_run_id, load_bundle, prepare_run, save_bundle
from mower_pov.config import load_config
from mower_pov.cutting import sample_grass_heights
from mower_pov.events import events_through
from mower_pov.motion import sample_pose


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    base = tmp_path_factory.mktemp("bundle")
    return prepare_run(load_config("configs/smoke.toml", {"world.duration_s": 4.0}), base)


def test_sequential_and_random_access_agree(bundle):
    initial = [0., .5, 1.5, 2.75, 3.873]
    run_dir = save_bundle(bundle)
    restored = load_bundle(run_dir)
    assert not restored.cut_state.root_first_cut_s.flags.writeable
    expected = {}
    for t in initial:
        expected[t] = (sample_pose(bundle.trajectory, t),
                       sample_grass_heights(bundle.roots, bundle.cut_state, t),
                       events_through(bundle.events, t))
    for t in [3.873, .5, 2.75, 0., 1.5]:
        pose, heights, events = expected[t]
        np.testing.assert_array_equal(pose, sample_pose(restored.trajectory, t))
        np.testing.assert_array_equal(heights, sample_grass_heights(restored.roots, restored.cut_state, t))
        assert events == events_through(restored.events, t)
    assert save_bundle(bundle) == run_dir


@pytest.mark.parametrize(("field", "value"), [
    ("grass.min_height_m", .07), ("grass.cut_height_m", .05), ("seed", 2),
    ("camera.mount_xyz_m", (.36, 0., .22)), ("mower.row_spacing_m", .27),
    ("render.engine_version", "different-renderer"),
])
def test_cache_invalidates_on_semantic_change(bundle, field, value):
    updated = load_config("configs/smoke.toml", {"world.duration_s": 4., field: value})
    assert prepare_run(updated, bundle.run_dir.parent).run_id != bundle.run_id


def test_report_timestamp_is_outside_run_identity():
    assert compute_run_id({"scene": 1, "report_timestamp": 1}) == compute_run_id(
        {"scene": 1, "report_timestamp": 2}
    )
    assert compute_run_id({"trajectory": "a"}) != compute_run_id({"trajectory": "b"})


def test_partial_or_unknown_schema_rejected(tmp_path, bundle):
    with pytest.raises(ValueError, match="manifest"):
        load_bundle(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(ValueError, match="schema"):
        load_bundle(tmp_path)
    manifest.write_text(json.dumps({"schema_version": 1, "files": {}}))
    with pytest.raises(ValueError, match="files"):
        load_bundle(tmp_path)


def test_same_size_corruption_rejected(bundle):
    path = save_bundle(bundle) / "trajectory.npz"
    original = path.read_bytes()
    try:
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with pytest.raises(ValueError, match="trajectory.npz"):
            load_bundle(path.parent)
    finally:
        path.write_bytes(original)
