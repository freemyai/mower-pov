from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import json

from mower_pov.cache import _array_identity, _datasets
from mower_pov.cases.legacy import run_offline
from mower_pov.cases.resolve import resolve_case
from mower_pov.cases.schema import load_spec
from mower_pov.course_layout import COURSE_OBSTACLES, COURSE_ROW_Y, COURSE_X_LIMITS
from mower_pov.policies.legacy import LegacyPlanner

import numpy as np
import pytest


@lru_cache(maxsize=2)
def _phase_one(case, policy):
    return run_offline(case, LegacyPlanner(policy))


@pytest.mark.parametrize("policy", ["legacy-joint", "legacy-greedy"])
def test_adapter_matches_104s_golden_state(scenario, write_scenario, policy):
    scenario["world"].update(
        width_m=12., depth_m=10., x_limits_m=COURSE_X_LIMITS, row_y_m=COURSE_ROW_Y,
        flower_bed_xyxy_m=[4.45, -1.6, 5.6, 1.6],
        obstacles=[
            {"id": o.name, "type": "disc" if len(o.points) == 1 else "polyline_sweep",
             "points_xy_m": o.points, "radius_m": o.radius_m} for o in COURSE_OBSTACLES
        ],
    )
    scenario["termination"]["max_sim_time_s"] = 104
    case = resolve_case(load_spec(write_scenario(scenario)), 20261120)
    trace = _phase_one(case, policy)
    assert trace.status == "COMPLETED"
    fixture = Path(__file__).parent / "fixtures/phase1-state-hashes.json"
    expected = json.loads(fixture.read_text())[policy]
    arrays = _datasets(trace.terrain, trace.trajectory, trace.roots, trace.cut_state, trace.story)
    assert {k: _array_identity(v) for k, v in arrays.items()} == expected["arrays"]
    assert trace.events_sha256 == expected["events_sha256"]


def test_camera_change_does_not_change_cut_state(scenario, write_scenario):
    scenario["world"]["obstacles"] = []
    case = resolve_case(load_spec(write_scenario(scenario)), 42)
    moved = replace(case, camera=replace(case.camera, mount_xyz_m=(.36, 0., .22)))
    planner = LegacyPlanner("legacy-joint")
    first, second = run_offline(case, planner), run_offline(moved, planner)
    assert first.state_id == second.state_id
    assert first.scenario_id != second.scenario_id
    np.testing.assert_array_equal(first.cut_state.t_first_cut_s, second.cut_state.t_first_cut_s)


def test_no_plan_is_not_empty_success(scenario, write_scenario):
    scenario["termination"]["max_sim_time_s"] = 2
    case = resolve_case(load_spec(write_scenario(scenario)), 42)
    trace = run_offline(case, LegacyPlanner("legacy-joint"))
    assert trace.status == "BUDGET_EXCEEDED"
    assert trace.trajectory is None


def test_planning_input_excludes_expected_outcome_and_assertions(scenario, write_scenario):
    case = resolve_case(load_spec(write_scenario(scenario)), 42)

    class Spy:
        id = "spy"
        version = "1"

        def plan(self, problem):
            assert not hasattr(problem, "expected_outcome")
            assert not hasattr(problem, "assertions")
            return LegacyPlanner("legacy-joint").plan(problem)

    assert run_offline(case, Spy()).status == "COMPLETED"
