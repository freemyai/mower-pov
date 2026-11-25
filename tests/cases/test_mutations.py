from dataclasses import replace

import numpy as np
import pytest

from mower_pov.cases.contracts import ObstacleSpec
from mower_pov.cases.evaluation import evaluate_episode
from mower_pov.cases.resolve import scenario_digest


def assert_failed(case, trace, name):
    verdict = evaluate_episode(case, trace)
    check = {check.name: check for check in verdict.assertions}[name]
    assert check.status == "FAIL"
    assert not verdict.passed


def test_inject_penetration(simple_case, simple_trace):
    item = ObstacleSpec("penetrated", "disc", ((0., 0.),), .05)
    case = replace(simple_case, world=replace(simple_case.world, obstacles=(item,)))
    assert_failed(case, replace(simple_trace, scenario_id=scenario_digest(case)),
                  "continuous_body_clearance")


def test_inject_boundary_intrusion(simple_case, simple_trace):
    pos = simple_trace.trajectory.position_m.copy()
    pos[len(pos)//2, 1] = simple_case.world.depth_m / 2
    assert_failed(simple_case, replace(simple_trace, trajectory=replace(
        simple_trace.trajectory, position_m=pos)), "no_boundary_violation")


def test_inject_overspeed(simple_case, simple_trace):
    speed = simple_trace.trajectory.v_mps.copy()
    speed[len(speed)//2] = .6
    assert_failed(simple_case, replace(simple_trace, trajectory=replace(
        simple_trace.trajectory, v_mps=speed)), "kinematic_limits")


@pytest.mark.parametrize("mode", ["early-cut", "blade-off"])
def test_inject_impossible_grass_history(simple_case, simple_trace, mode):
    trace = simple_trace
    if mode == "early-cut":
        grid = trace.cut_state.t_first_cut_s.copy()
        grid[np.isfinite(grid)] = 0
        trace = replace(trace, cut_state=replace(trace.cut_state, t_first_cut_s=grid))
    else:
        trace = replace(trace, trajectory=replace(trace.trajectory,
                        blade_enabled=np.zeros_like(trace.trajectory.blade_enabled)))
    assert_failed(simple_case, trace, "no_cutting_when_rotor_inactive")


def test_inject_wrong_target_denominator(simple_case, simple_trace):
    mask = simple_trace.terrain.mowable_mask.copy()
    mask[mask.shape[0]//2, mask.shape[1]//2] = False
    trace = replace(simple_trace, terrain=replace(simple_trace.terrain, mowable_mask=mask))
    assert_failed(simple_case, trace, "target_geometry")
