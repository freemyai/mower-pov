import shutil

import pytest

from mower_pov.rendering.blender_house import assets_for_house


def test_house_checks_license_and_texture_bytes(tmp_path):
    shutil.copytree("assets", tmp_path / "assets")
    manifest = tmp_path / "assets/catalog.lock.json"
    assets, identity = assets_for_house(manifest)
    assert set(assets) == {"red_brick", "concrete_floor_worn_001"}
    assert len(identity) == 64
    path = assets["red_brick"]["channels"]["roughness"]["path"]
    path.write_bytes(b"updated")
    with pytest.raises(ValueError, match="digest"):
        assets_for_house(manifest)
