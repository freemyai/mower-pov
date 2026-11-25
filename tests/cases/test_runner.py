import json

from mower_pov.cases.runner import RunRequest, load_suite, read_suite_result, run_suite


def test_same_seed_replay_and_order_do_not_change_evidence(two_case_suite, tmp_path):
    suite = load_suite(two_case_suite)
    one = run_suite(suite, "legacy-joint", tmp_path/"one", jobs=2)
    data = json.loads(two_case_suite.read_text())
    data["cases"].reverse()
    two_case_suite.write_text(json.dumps(data))
    two = run_suite(load_suite(two_case_suite), "legacy-joint", tmp_path/"two")
    assert {r.id:r.state_id for r in one.rows} == {r.id:r.state_id for r in two.rows}
    assert all(r.passed for r in one.rows + two.rows)
    assert [r.id for r in two.rows] == ["S00-beta", "S00-alpha"]


def test_missing_case_is_incomplete(two_case_suite, tmp_path):
    suite = load_suite(two_case_suite)
    result = read_suite_result(tmp_path/"empty", RunRequest(suite, "legacy-joint"))
    assert result.incomplete == ("S00-alpha", "S00-beta")


def test_invalid_input_remains_in_summary(two_case_suite, tmp_path):
    (tmp_path/"S00-beta.json").write_text("{")
    result = run_suite(load_suite(two_case_suite), "legacy-joint", tmp_path/"run")
    assert len(result.rows) == 2
    assert result.rows[1].outcome == "INVALID_INPUT"


def test_timeout_remains_in_summary(two_case_suite, tmp_path):
    result = run_suite(load_suite(two_case_suite), "legacy-joint", tmp_path/"abort", timeout_s=.001)
    assert [r.outcome for r in result.rows] == ["TIMEOUT", "TIMEOUT"]
