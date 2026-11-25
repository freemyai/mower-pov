"""Check fixed feasibility witnesses without invoking either coverage policy."""

import json
from pathlib import Path

import numpy as np

from mower_pov.cases.geometry import (
    bed_distance, interval_lower_bound, swept_obstacle_clearance, swept_yard_clearance,
)
from mower_pov.cases.runner import load_suite, validate_suite

ROOT = Path(__file__).resolve().parents[2]


def _paths(world, offsets):
    x0, x1 = world.x_limits_m
    u = np.linspace(0., 1., 1001)
    span = x1 - x0
    for i, (y, offset) in enumerate(zip(world.row_y_m, offsets)):
        x = x0 + span * u
        curve = y + offset * np.sin(np.pi * u)**2
        yaw = np.arctan(offset * np.pi / span * np.sin(2 * np.pi * u))
        xy = np.column_stack((x, curve))
        yield (xy[::-1], yaw[::-1] - np.pi) if i % 2 else (xy, yaw)
        if i + 1 < len(world.row_y_m):
            radius = (y - world.row_y_m[i+1]) / 2
            sign = -1 if i % 2 else 1
            arc = np.column_stack(((x0 if i % 2 else x1) + sign * radius * np.sin(np.pi * u),
                                  y + radius * (np.cos(np.pi * u) - 1)))
            yield arc, (-np.pi + np.pi * u) if i % 2 else -np.pi * u


def _time_bound(case, xy, yaw):
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    curvature = np.gradient(yaw, distance, edge_order=2)
    bend = max(.001, abs(curvature).max())
    bend_change = max(.001, abs(np.gradient(curvature, distance, edge_order=2)).max())
    robot = case.robot
    speed = min(robot.max_speed_mps, .98 * robot.max_omega_radps / bend,
                np.sqrt(.49 * robot.max_alpha_radps2 / bend_change))
    accel = min(.98 * robot.max_accel_mps2, .49 * robot.max_alpha_radps2 / bend)
    speed = min(speed, np.sqrt(distance[-1] * accel))
    return distance[-1] / speed + speed / accel


def test_reference_curves_and_turns_have_clearance_and_time():
    suite = load_suite(ROOT/"scenarios/suites/static-v1.json")
    witnesses = json.loads((ROOT/"scenarios/static/reference-witnesses.json").read_text())
    for case in validate_suite(suite):
        witness = witnesses[case.id]
        if witness["kind"] != "routed":
            continue
        world = case.world
        size = case.robot.length_m, case.robot.width_m
        time = .75 + .72 + .35
        for xy, yaw in _paths(world, witness["offset_m"]):
            assert swept_yard_clearance(xy, yaw, size, (world.width_m, world.depth_m)).min() >= 0, case.id
            for obstacle in world.obstacles:
                gap = swept_obstacle_clearance(xy, yaw, size, obstacle).min()
                assert gap >= world.clearance_m, (case.id, obstacle.id, gap)
            if world.flower_bed_xyxy_m is not None:
                distance = bed_distance(xy, yaw, size, world.flower_bed_xyxy_m)
                gap = interval_lower_bound(distance, xy, yaw, size).min()
                assert gap >= world.clearance_m, (case.id, gap)
            time += _time_bound(case, xy, yaw)
        assert time <= case.duration_s, (case.id, time)
