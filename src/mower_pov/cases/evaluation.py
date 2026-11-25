"""Validate a static run from its actual state and the scenario's independent truth."""

from dataclasses import dataclass

import numpy as np

from ..controller import CUTTING_ROTOR_FRACTION, SPIN_DOWN_S, SPIN_UP_S
from .contracts import STATIC_CUT_HEIGHT_M
from .geometry import (WAYPOINT_TOL_M, bed_distance, interval_lower_bound, mission_impossibility,
                       swept_obstacle_clearance, swept_yard_clearance)
from .metrics import Metrics, measure_episode, target_grid, trace_sweep
from .resolve import scenario_digest

_TOL = 1e-8


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str = ""
    first_violation_s: float | None = None
    sample_index: int | None = None


@dataclass(frozen=True)
class Verdict:
    validity: str
    outcome: str
    expected_outcome: str
    passed: bool
    assertions: tuple[Check, ...]
    first_violation_s: float | None
    metrics: Metrics | None
    evidence: tuple[str, ...] = ("trace.npz", "events.json")


def _check(name, truth, detail="", time=None, index=None):
    return Check(name, "PASS" if truth else "FAIL", detail, None if truth else time, None if truth else index)


def _series(name, bad, times, detail):
    indices = np.flatnonzero(bad)
    index = int(indices[0]) if len(indices) else None
    return _check(name, index is None, detail, float(times[index]) if index is not None else None, index)


def _verdict(case, outcome, checks, metrics=None, validity="VALID"):
    times = [check.first_violation_s for check in checks if check.first_violation_s is not None]
    passed = validity == "VALID" and outcome == case.expected_outcome and all(
        check.status == "PASS" or (outcome == "SAFE_REJECTION" and check.status == "NOT_EVALUATED")
        for check in checks
    )
    return Verdict(validity, outcome, case.expected_outcome, passed, tuple(checks),
                   min(times) if times else None, metrics,
                   ("trace.npz", "events.json") if outcome not in {"SAFE_REJECTION", "NO_FEASIBLE_PLAN",
                                                                  "BUDGET_EXCEEDED"} else ())


def _motion_checks(case, tr, story):
    t = tr.time_s
    xy, yaw, dt = tr.position_m[:, :2], tr.unwrapped_yaw, np.diff(t)
    robot = case.robot
    speed = np.maximum(abs(tr.v_mps[:-1]), abs(tr.v_mps[1:]))
    yaw_speed = np.maximum(abs(tr.omega_radps[:-1]), abs(tr.omega_radps[1:]))
    velocity = np.diff(xy, axis=0) / dt[:, None]
    heading = (yaw[1:] + yaw[:-1]) / 2
    forward = np.column_stack((np.cos(heading), np.sin(heading)))
    lateral = np.column_stack((-forward[:, 1], forward[:, 0]))
    along = np.einsum("ij,ij->i", velocity, forward)
    sideways = np.einsum("ij,ij->i", velocity, lateral)
    actual_speed = np.linalg.norm(velocity, axis=1)
    yaw_velocity = np.diff(yaw) / dt
    actual_yaw_speed = abs(yaw_velocity)
    bad = ((speed > robot.max_speed_mps + _TOL) | (actual_speed > robot.max_speed_mps + 4e-4)
           | (yaw_speed > robot.max_omega_radps + _TOL)
           | (actual_yaw_speed > robot.max_omega_radps + 4e-4)
           | (abs(np.diff(tr.v_mps)) / dt > robot.max_accel_mps2 + _TOL)
           | (abs(np.diff(tr.omega_radps)) / dt > robot.max_alpha_radps2 + _TOL))
    # Position uses linear chords, and heading/controls are interpolated
    # separately. These allowances cover the resulting sampling residuals.
    bad |= abs(along - (tr.v_mps[1:] + tr.v_mps[:-1]) / 2) > .001
    bad |= abs(sideways) > .010
    bad |= abs(yaw_velocity - (tr.omega_radps[1:] + tr.omega_radps[:-1]) / 2) > .150
    wheels = tr.v_mps[:, None] + tr.omega_radps[:, None] * np.array([-1., 1.]) * robot.track_m / 2
    wheel_bad = np.any(~np.isclose(tr.wheel_speeds, wheels, rtol=0, atol=_TOL), axis=1)
    bad |= wheel_bad[:-1] | wheel_bad[1:]
    rotor, command = story.rotor_fraction, story.blade_command
    rate = np.where(command[:-1], 1 / SPIN_UP_S, -1 / SPIN_DOWN_S)
    predicted = np.clip(rotor[:-1] + rate * dt, 0., 1.)
    bad_rotor = (command[:-1] == command[1:]) & (abs(rotor[1:] - predicted) > _TOL)
    sample_bad = ((rotor >= CUTTING_ROTOR_FRACTION) != tr.blade_enabled)
    sample_bad |= ~np.isfinite(rotor) | (rotor < 0) | (rotor > 1)
    sample_bad[0] |= rotor[0] > _TOL
    sample_bad[:-1] |= bad_rotor
    return (
        _series("kinematic_limits", bad, t, "commanded rates, pose-derived speeds and acceleration limits"),
        _series("rotor_state", sample_bad, t, "rotor envelope and effective cutting must follow blade commands"),
    )


def _sweep_checks(case, trajectory, clearance_m):
    xy, yaw, time = trajectory.position_m[:, :2], trajectory.unwrapped_yaw, trajectory.time_s
    world, robot = case.world, case.robot
    size = robot.length_m, robot.width_m
    edge = swept_yard_clearance(xy, yaw, size, (world.width_m, world.depth_m))
    yield _series("no_boundary_violation", edge < -_TOL, time,
                  f"conservative yard edge clearance {float(edge.min()):.6f} m")
    checks = [(item.id, swept_obstacle_clearance(xy, yaw, size, item)) for item in world.obstacles]
    if world.flower_bed_xyxy_m is not None:
        distance = bed_distance(xy, yaw, size, world.flower_bed_xyxy_m)
        checks.append(("flower bed rectangle", interval_lower_bound(distance, xy, yaw, size)))
    if not checks:
        yield _check("continuous_body_clearance", True, "empty obstacle map")
        return
    margins = np.minimum.reduce([values for _, values in checks])
    bad = (margins < clearance_m - _TOL) | (margins <= 0.)
    index = int(np.argmax(bad)) if bad.any() else int(np.argmin(margins))
    nearest, _ = min(checks, key=lambda item: item[1][index])
    detail = f"{nearest}, conservative clearance {margins[index]:.6f} m, required {clearance_m:.6f} m"
    yield _series("continuous_body_clearance", bad, time, detail)


def _complete(case, tr, story):
    x0, x1 = case.world.x_limits_m
    xy = tr.position_m[:, :2]
    start = 0
    for row, y in enumerate(case.world.row_y_m):
        ends = (x0, x1) if row % 2 == 0 else (x1, x0)
        for x in ends:
            close = np.flatnonzero(np.linalg.norm(xy[start:] - (x, y), axis=1) <= WAYPOINT_TOL_M)
            if not len(close):
                return False
            start += int(close[0])
    return bool(abs(tr.v_mps[-1]) < _TOL and abs(tr.omega_radps[-1]) < _TOL
                and not story.blade_command[-1] and story.rotor_fraction[-1] <= _TOL
                and np.linalg.norm(xy[-1] - (x, y)) <= WAYPOINT_TOL_M)


def _first_cut_mismatch(actual, expected):
    if actual.shape != expected.shape:
        return True, None
    bad = ~np.isclose(actual, expected, rtol=0, atol=1e-9)
    if not bad.any():
        return False, None
    candidates = np.minimum(actual[bad], expected[bad])
    return True, float(np.min(candidates[np.isfinite(candidates)])) if np.isfinite(candidates).any() else None


def evaluate_episode(case, trace):
    checks = [
        _check("scenario_identity", trace.scenario_id == scenario_digest(case)),
        _check("no_obstacles_removed_to_enable_success", trace.planned_world_ok),
    ]
    outcome = trace.status
    if trace.trajectory is None:
        proof = mission_impossibility(case)
        proven = outcome in {"NO_FEASIBLE_PLAN", "BUDGET_EXCEEDED"} and proof is not None
        if proven:
            outcome = "SAFE_REJECTION"
        elif outcome not in {"NO_FEASIBLE_PLAN", "BUDGET_EXCEEDED"}:
            outcome = "INVALID_TRACE"
        checks.append(_check("rejection_certificate", proven, proof or trace.reason))
        names = ("continuous_body_clearance", "no_boundary_violation", "no_cutting_when_rotor_inactive",
                 "kinematic_limits", "complete_static_work", "minimum_target_coverage")
        checks.extend(Check(name, "NOT_EVALUATED", "no executed trajectory") for name in names)
        return _verdict(case, outcome, checks, measure_episode(case, trace))

    tr, story = trace.trajectory, trace.story
    times = tr.time_s
    period = 1 / case.world.step_hz
    clock_ok = (len(times) > 1 and np.all(np.isfinite(times)) and abs(times[0]) < _TOL
                and abs(times[-1] - case.duration_s) < _TOL
                and np.allclose(np.diff(times), period, rtol=0, atol=1e-12))
    checks.append(_check("simulation_clock", clock_ok, "absolute v1 simulation grid and terminal time"))
    if not clock_ok or not all(np.isfinite(values).all() for values in (
        tr.position_m, tr.unwrapped_yaw, tr.v_mps, tr.omega_radps
    )):
        checks.append(_check("finite_pose_trace", False, "invalid pose stream or time grid"))
        return _verdict(case, "INVALID_TRACE", checks)

    requested = {item.name:item for item in case.assertions}
    minimum = case.world.clearance_m
    if "continuous_body_clearance" in requested:
        minimum = max(minimum, requested["continuous_body_clearance"].minimum_m)
    checks.extend(_sweep_checks(case, tr, minimum))
    checks.extend(_motion_checks(case, tr, story))
    complete = _complete(case, tr, story)
    checks.append(_check("complete_static_work", complete, "all fixed lane waypoints in order and final stop",
                         float(times[-1])))

    terrain = trace.terrain
    origin, grass_mask, _ = target_grid(case.world, case.robot.blade_radius_m)
    truth_grid = (terrain.cell_size_m == case.world.cell_m and np.array_equal(terrain.origin_xy_m, origin)
                  and np.array_equal(terrain.mowable_mask, grass_mask))
    checks.append(_check("target_geometry", truth_grid, "nominal rows, yard and bed determine the target"))
    metrics, first, roots = trace_sweep(case, trace, check_roots=True)
    grid_bad, grid_time = _first_cut_mismatch(trace.cut_state.t_first_cut_s, first)
    root_bad, root_time = _first_cut_mismatch(trace.cut_state.root_first_cut_s, roots)
    candidates = [time for time in (grid_time, root_time) if time is not None]
    first_bad = min(candidates) if candidates else None
    index = int(np.searchsorted(times, first_bad, side="right") - 1) if first_bad is not None else None
    checks.append(_check("no_cutting_when_rotor_inactive",
                         not (grid_bad or root_bad) and trace.cut_state.cut_height_m == STATIC_CUT_HEIGHT_M,
                         "cell and root first contacts recomputed from the effective cutter", first_bad, index))
    if "minimum_target_coverage" in requested:
        fraction = requested["minimum_target_coverage"].minimum_fraction
        checks.append(_check("minimum_target_coverage", metrics.target_fraction_cut >= fraction,
                             f"coverage {metrics.target_fraction_cut:.6f}, required {fraction:.6f}", float(times[-1])))
    return _verdict(case, "COMPLETED" if complete else "INCOMPLETE", checks, metrics)
