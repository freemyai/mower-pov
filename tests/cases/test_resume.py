from pathlib import Path

import pytest

from mower_pov.cases.evidence import CorruptEvidence
from mower_pov.cases.runner import RunRequest, load_suite, resume_suite, run_suite


def test_resume_reuses_exact_request_and_rejects_corruption(two_case_suite, tmp_path):
    suite = load_suite(two_case_suite)
    request = RunRequest(suite, "legacy-joint")
    out = tmp_path/"run"
    first = run_suite(suite, request.policy_id, out)
    assert resume_suite(out, request) == first
    artifact = out/first.rows[0].directory/"trace.npz"
    artifact.write_bytes(b"corrupt")
    with pytest.raises(CorruptEvidence, match="trace.npz"):
        resume_suite(out, request)


def test_change_policy_refuses_reuse_of_output(two_case_suite, tmp_path):
    suite = load_suite(two_case_suite)
    out = tmp_path/"run"
    run_suite(suite, "legacy-joint", out, jobs=2)
    with pytest.raises(ValueError, match="different request"):
        resume_suite(out, RunRequest(suite, "legacy-greedy"))


def test_interrupted_trace_write_does_not_publish_complete_identity(tmp_path, simple_trace, monkeypatch):
    from mower_pov.cases import evidence

    write = evidence.atomic_write

    def interrupted(path, content, mode="w"):
        if path.name == "events.json":
            raise OSError("injected interrupt")
        return write(path, content, mode)

    monkeypatch.setattr(evidence, "atomic_write", interrupted)
    with pytest.raises(OSError, match="injected interrupt"):
        evidence.write_trace(simple_trace, tmp_path)
    assert not (tmp_path/"trace.json").exists()
