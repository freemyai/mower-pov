from dataclasses import replace

import numpy as np

from mower_pov.cases.contracts import ObstacleSpec
from mower_pov.cases.evaluation import evaluate_episode
from mower_pov.cases.legacy import run_offline
from mower_pov.policies.legacy import LegacyPlanner


def test_safe_rejection_has_independent_barrier(simple_case):
    obstruction = ObstacleSpec("full-width", "polyline_sweep", ((0., -2.), (0., 2.)), .04)
    case = replace(simple_case, world=replace(simple_case.world, obstacles=(obstruction,)),
                   expected_outcome="SAFE_REJECTION")
    trace = run_offline(case, LegacyPlanner("legacy-joint"))
    verdict = evaluate_episode(case, trace)
    assert verdict.validity == "VALID"
    assert verdict.outcome == "SAFE_REJECTION"
    assert verdict.passed


def test_parked_forever_cannot_pass_a_reachable_route(simple_case, simple_trace):
    trajectory = simple_trace.trajectory
    xy = np.tile(trajectory.position_m[0], (len(trajectory.time_s), 1))
    trace = replace(simple_trace, trajectory=replace(trajectory, position_m=xy))
    verdict = evaluate_episode(simple_case, trace)
    assert not verdict.passed
    assert verdict.outcome == "INCOMPLETE"


def test_simple_plan_passes_all_checks(simple_case, simple_trace):
    verdict = evaluate_episode(simple_case, simple_trace)
    assert verdict.passed
    assert verdict.metrics.target_fraction_cut > .99
