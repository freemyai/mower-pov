
import pytest
from PIL import Image

from mower_pov.rendering.driver import pending_frames, require_render_space, validate_frame


def test_resume_renders_missing_frames_only(tmp_path):
    path = tmp_path / "frame-000001.png"
    Image.new("RGB", (64, 36), (30, 60, 90)).save(path)
    from hashlib import sha256
    record = {"frame": 1, "files": {path.name: sha256(path.read_bytes()).hexdigest()}}
    broken = tmp_path / "frame-000002.png"
    broken.write_bytes(path.read_bytes()[:15])
    assert validate_frame(path, record)
    assert not validate_frame(broken, record)
    assert pending_frames(tmp_path, (1, 2), [record], ".png", False) == (2,)


def test_same_size_wrong_run_frame_rejected(tmp_path):
    from hashlib import sha256
    path = tmp_path / "frame-000001.png"
    Image.new("RGB", (64, 36), (30, 60, 90)).save(path)
    body = path.read_bytes()
    record = {"frame": 1, "files": {path.name: sha256(body).hexdigest()}}
    path.write_bytes(body[:-1] + b"!")
    assert path.stat().st_size == len(body)
    assert pending_frames(tmp_path, (1,), [record], ".png", False) == (1,)


def test_no_disk_space_fails_cleanly(tmp_path):
    with pytest.raises(OSError, match="space"):
        require_render_space(tmp_path, 1 << 50)
