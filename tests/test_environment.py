from mower_pov.environment import probe_environment


def test_missing_blender_reported(tmp_path):
    report = probe_environment(tmp_path / "missing-blender")
    assert report["blender"]["available"] is False
    assert report["usable"] is False


def test_cpu_fallback_is_explicit(tmp_path):
    report = probe_environment(tmp_path / "missing-blender", "OPTIX")
    assert report["requested_device"] == "OPTIX"
    assert report["verified_device"] is None
