import json

from mower_pov.cache import prepare_run
from mower_pov.config import load_config
from mower_pov.diagnostics import write_diagnostics


def test_diagnostics_create_inspectable_artifacts(tmp_path):
    cfg = load_config("configs/smoke.toml", {"world.duration_s": 4.})
    bundle = prepare_run(cfg, tmp_path / "bundles")
    output = tmp_path / "evidence"
    write_diagnostics(bundle, output)
    for name in ["terrain", "trajectory", "first-cut", "final-height"]:
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    report = json.loads((output / "state-checks.json").read_text())
    assert report["run_id"] == bundle.run_id
    assert report["camera_mount_max_abs_error"] < 1e-6
    assert report["random_access_matches_sequential"] is True
