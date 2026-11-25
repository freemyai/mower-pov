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
