from dataclasses import replace
from pathlib import Path
import json

from mower_pov.cases.evaluation import evaluate_episode
from mower_pov.cases.geometry import mission_impossibility
from mower_pov.cases.legacy import run_offline
from mower_pov.cases.runner import load_suite, validate_suite
from mower_pov.policies.legacy import LegacyPlanner

ROOT = Path(__file__).resolve().parents[2]
LIST = ROOT/"scenarios/suites/static-v1.json"


def test_static_suite_has_exactly_30_unique_ids():
    cases = validate_suite(load_suite(LIST))
    assert len(cases) == 30
    assert len({case.id for case in cases}) == 30
    assert {case.id.split("-")[0] for case in cases} == {f"S{i:02}" for i in range(1, 7)}


def test_all_cases_have_assertions_and_declared_witnesses():
    cases = validate_suite(load_suite(LIST))
    witnesses = json.loads((ROOT/"scenarios/static/reference-witnesses.json").read_text())
    assert set(witnesses) == {case.id for case in cases}
    for case in cases:
        assert case.assertions
        witness = witnesses[case.id]
        if case.expected_outcome == "SAFE_REJECTION":
            assert mission_impossibility(case), case.id
            assert witness["kind"] == "impossibility"
        else:
            assert witness["kind"] == "routed"
            assert len(witness["offset_m"]) == len(case.world.row_y_m)
            assert not mission_impossibility(case), case.id


def test_valid_infeasible_case_is_not_invalid_input():
    cases = validate_suite(load_suite(LIST))
    case = next(c for c in cases if c.id == "S03-05-closed-gap")
    result = evaluate_episode(case, run_offline(case, LegacyPlanner("legacy-greedy")))
    assert result.validity == "VALID" and result.outcome == "SAFE_REJECTION"
    assert result.passed


def test_current_policy_cannot_win_by_deleting_obstacles(simple_case):
    case = replace(simple_case, world=replace(simple_case.world, obstacles=(
        obstacle_from_suite(),
    )))

    class DeletesObstacle:
        id = "deletes-obstacle"
        version = "1"

        def plan(self, problem):
            cleared = replace(problem, world=replace(problem.world, obstacles=()))
            return LegacyPlanner("legacy-joint").plan(cleared)

    verdict = evaluate_episode(case, run_offline(case, DeletesObstacle()))
    assert not verdict.passed
    assert next(c for c in verdict.assertions if c.name == "no_obstacles_removed_to_enable_success").status == "FAIL"


def obstacle_from_suite():
    return validate_suite(load_suite(LIST))[0].world.obstacles[0]
