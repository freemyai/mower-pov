"""Coverage planning, steering limits, and replay for the multi-row yard course."""

from dataclasses import dataclass

import numpy as np

from .actors.behavior import story_tracks
from .contracts import StoryState
from .controller import (CUTTING_ROTOR_FRACTION, SPIN_DOWN_S, SPIN_UP_S,
                         _half_turn, _smoothstep)
from .course_layout import (BODY_DESIGN_CLEARANCE_M, COURSE_OBSTACLES,
                            COURSE_ROW_Y, COURSE_X_LIMITS, obstacle_clearance,
                            obstacle_swept_clearance)
from .motion import trajectory_from_path
from .obstacles import validate_story_layout


@dataclass
class Row:
    xy: np.ndarray
    s: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray
    speed: np.ndarray
    times: np.ndarray
    offset_m: float
    nominal_blockers: tuple
    bypass_x: tuple | None
    candidate_count: int

    @property
    def duration(self):
        return self.times[-1]


@dataclass(frozen=True)
class CoursePlan:
    rows: tuple[Row, ...]
    turn_duration: float
    required_time: float

    def report(self):
        start = .75
        report = []
        for i, row in enumerate(self.rows):
            stop = start + row.duration
            report.append({
                "row": i + 1, "nominal_y_m": COURSE_ROW_Y[i],
                "t_start_s": float(start), "t_end_s": float(stop),
                "route_length_m": float(row.s[-1]), "bypass_offset_m": row.offset_m,
                "nominal_blockers": row.nominal_blockers, "compared_candidates": row.candidate_count,
            })
            start = stop + self.turn_duration
        return report


def _row_geometry(x, y, eastbound):
    if not eastbound:
        x, y = x[::-1], y[::-1]
    xy = np.column_stack((x, y))
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    slope = np.gradient(y, x, edge_order=2)
    yaw = np.arctan(slope) - (0. if eastbound else np.pi)
    yaw[[0, -1]] = 0. if eastbound else -np.pi
    curvature = np.gradient(yaw, s, edge_order=2)
    return xy, s, yaw, curvature


def _profile(s, curvature, obstacle_margin, mower):
    magnitude = np.abs(curvature)
    # Use both neighbors so the bound covers interpolation along each source chord.
    slope = np.maximum.reduce((magnitude, np.r_[magnitude[0], magnitude[:-1]],
                               np.r_[magnitude[1:], magnitude[-1]]))
    segment_change = np.abs(np.diff(curvature) / np.diff(s))
    change = np.maximum(np.r_[segment_change[0], segment_change],
                        np.r_[segment_change, segment_change[-1]])
    omega_limit = .998 * mower.max_omega_radps / np.maximum(slope, .001)
    # d(omega)/dt = curvature * drive acceleration + curvature' * speed**2.
    bend_limit = np.sqrt(.57 * mower.max_alpha_radps2 / np.maximum(change, .001))
    obstacle_limit = np.interp(obstacle_margin, [.040, .11, .33], [.160, .215, .30])
    speed = np.minimum.reduce((np.full_like(s, mower.max_speed_mps), omega_limit, bend_limit,
                               np.where(obstacle_margin < .33, obstacle_limit, mower.max_speed_mps)))
    acceleration = np.minimum(mower.max_accel_mps2 * .96,
                              .40 * mower.max_alpha_radps2 / np.maximum(slope, .001))
    speed[[0, -1]] = 0.
    for i in range(1, len(speed)):
        distance = s[i] - s[i - 1]
        speed[i] = min(speed[i], np.sqrt(speed[i - 1] ** 2 + 2 * distance *
                                       min(acceleration[i - 1], acceleration[i])))
    for i in range(len(speed) - 2, -1, -1):
        distance = s[i + 1] - s[i]
        speed[i] = min(speed[i], np.sqrt(speed[i + 1] ** 2 + 2 * distance *
                                       min(acceleration[i + 1], acceleration[i])))
    times = np.r_[0., np.cumsum(2 * np.diff(s) / (speed[:-1] + speed[1:]))]
    return speed, times


def _all_pose_clearances(xy, yaw, config):
    return np.stack([obstacle_swept_clearance(xy, yaw, config, p) for p in COURSE_OBSTACLES])


def _plan_row(config, x, lane_y, eastbound, prior_rows):
    nominal_xy = np.column_stack((x, np.full_like(x, lane_y)))
    nominal = np.stack([obstacle_clearance(nominal_xy, np.zeros(len(x)), config, item)
                        for item in COURSE_OBSTACLES])
    planning_clearance = BODY_DESIGN_CLEARANCE_M + .001
    blocked = nominal < planning_clearance + .002
    blockers = tuple(obstacle.name for obstacle, collision in zip(COURSE_OBSTACLES, blocked)
                     if collision.any())
    baseline = np.full_like(x, lane_y)
    if not blockers:
        xy, s, yaw, curvature = _row_geometry(x, baseline, eastbound)
        clearance = _all_pose_clearances(xy, yaw, config).min(axis=0)
        speed, times = _profile(s, curvature, clearance, config.mower)
        return Row(xy, s, yaw, curvature, speed, times, 0., (), None, 1)

    threat_x = x[blocked.any(axis=0)]
    winner = None
    tried = 0
    ramp_pairs = ((.45, .51), (.58, .66), (.72, .72), (.92, .92),
                  (.48, .89), (.89, .48))
    for expansion, (left, right) in ((pad, pair) for pad in (.025, .13, .22) for pair in ramp_pairs):
        entry = max(x[0] + .03, threat_x[0] - expansion)
        exit = min(x[-1] - .03, threat_x[-1] + expansion)
        start = max(x[0], entry - left)
        stop = min(x[-1], exit + right)
        shape = _smoothstep(x, start, entry) - _smoothstep(x, exit, stop)
        for magnitude in np.arange(.10, .66, .03):
            for sign in (-1, 1):
                offset = float(sign * magnitude)
                y = baseline + offset * shape
                xy, s, yaw, curvature = _row_geometry(x, y, eastbound)
                tried += 1
                # Reject obvious near collisions on a coarser path before checking the sweep.
                coarse = [obstacle_clearance(xy[::7], yaw[::7], config, item).min()
                          for item in COURSE_OBSTACLES]
                if min(coarse) < planning_clearance:
                    continue
                clearance = _all_pose_clearances(xy, yaw, config).min(axis=0)
                if clearance.min() < planning_clearance:
                    continue
                speed, times = _profile(s, curvature, clearance, config.mower)
                extra_distance = s[-1] - (x[-1] - x[0])
                # Prefer safe and shorter returns to the original lane; give redundant sweeps a cost.
                already = np.zeros(len(xy), dtype=bool)
                for previous in prior_rows:
                    px, py = previous.xy.T
                    order = slice(None) if px[0] < px[-1] else slice(None, None, -1)
                    previous_y = np.interp(xy[:, 0], px[order], py[order])
                    already |= np.abs(xy[:, 1] - previous_y) < config.mower.blade_radius_m * 1.20
                repeated_m = float(np.sum(np.diff(s) * already[1:]))
                score = times[-1] + .85 * abs(offset) + extra_distance + 1.40 * repeated_m
                if winner is None or score < winner[0]:
                    winner = (score, Row(xy, s, yaw, curvature, speed, times, offset, blockers,
                                         (start, entry, exit, stop), tried))
    if winner is None:
        raise ValueError(f"No clear course candidate for row y={lane_y:.2f} m around {', '.join(blockers)}")
    winner[1].candidate_count = tried
    return winner[1]


def plan_course(config):
    """Check the course's known obstacles and choose drivable bypasses."""
    x = np.linspace(*COURSE_X_LIMITS, 527)
    rows = []
    for index, lane in enumerate(COURSE_ROW_Y):
        rows.append(_plan_row(config, x, lane, index % 2 == 0, rows))
    mower = config.mower
    turn_radius = abs(COURSE_ROW_Y[1] - COURSE_ROW_Y[0]) * .5
    omega = min(mower.max_omega_radps, mower.max_speed_mps / turn_radius)
    alpha = min(mower.max_alpha_radps2, mower.max_accel_mps2 / turn_radius)
    peak = min(omega, np.sqrt(np.pi * alpha))
    turn_duration = np.pi / peak + peak / alpha
    required = .75 + sum(row.duration for row in rows) + 4 * turn_duration + SPIN_DOWN_S + .35
    if config.world.duration_s < required:
        raise ValueError(f"mowing course needs {required:.2f} seconds with the configured motion limits")
    return CoursePlan(tuple(rows), turn_duration, required)


def _evaluate_row(row, seconds):
    index = np.minimum(np.searchsorted(row.times, seconds, side="right") - 1, len(row.times) - 2)
    u = seconds - row.times[index]
    acceleration = (row.speed[index + 1] - row.speed[index]) / (row.times[index + 1] - row.times[index])
    v = row.speed[index] + acceleration * u
    s = row.s[index] + u * (row.speed[index] + .5 * acceleration * u)
    xy = np.column_stack([np.interp(s, row.s, values) for values in row.xy.T])
    yaw = np.interp(s, row.s, row.yaw)
    omega = v * np.interp(s, row.s, row.curvature)
    modes = np.full(len(s), "MOW", dtype="U8")
    names = np.full(len(s), "", dtype="U12")
    if row.bypass_x is not None:
        start, entry, exit, stop = row.bypass_x
        x = xy[:, 0]
        if row.xy[0, 0] > row.xy[-1, 0]:
            x, start, entry, exit, stop = -x, -stop, -exit, -entry, -start
        lead = max(x[0] - .001, start - .19)
        plan_end = start + .21 * (entry - start)
        modes[(x >= lead) & (x < plan_end)] = "REPLAN"
        modes[(x >= plan_end) & (x < exit)] = "BYPASS"
        modes[(x >= exit) & (x <= stop)] = "REJOIN"
        relevant = [item for item in COURSE_OBSTACLES if item.name in row.nominal_blockers]
        pivot = np.array([np.mean(np.asarray(item.points)[:, 0]) for item in relevant])
        closest = np.argmin(np.abs(xy[:, :1] - pivot), axis=1)
        on_change = modes != "MOW"
        names[on_change] = np.array(row.nominal_blockers)[closest[on_change]]
    return xy, yaw, v, omega, modes, names


def _evaluate_turn(config, row, seconds):
    m = config.mower
    radius = .5 * abs(COURSE_ROW_Y[1] - COURSE_ROW_Y[0])
    cap_omega = min(m.max_omega_radps, m.max_speed_mps / radius)
    cap_alpha = min(m.max_alpha_radps2, m.max_accel_mps2 / radius)
    forward = row.xy[0, 0] < row.xy[-1, 0]
    direction = -1. if forward else 1.
    angle, speed = np.array([_half_turn(t, cap_omega, cap_alpha) for t in seconds]).T
    yaw0 = row.yaw[-1]
    yaw = yaw0 + direction * angle
    xy = row.xy[-1] + direction * radius * np.column_stack(
        (np.sin(yaw) - np.sin(yaw0), np.cos(yaw0) - np.cos(yaw)))
    return xy, yaw, radius * speed, direction * speed


def run_course(config, terrain):
    plan = plan_course(config)
    times = np.arange(round(config.world.duration_s * config.world.step_hz) + 1) / config.world.step_hz
    xy = np.tile(plan.rows[0].xy[0], (len(times), 1))
    yaw, v, omega = (np.zeros(len(times)) for _ in range(3))
    mode = np.full(len(times), "PARK", dtype="U8")
    obstacle = np.full(len(times), "", dtype="U12")
    start = .75
    for index, row in enumerate(plan.rows):
        end = start + row.duration
        moving = (times >= start) & (times < end)
        xy[moving], yaw[moving], v[moving], omega[moving], mode[moving], obstacle[moving] = (
            _evaluate_row(row, times[moving] - start))
        start = end
        if index + 1 == len(plan.rows):
            xy[times >= end] = row.xy[-1]
            yaw[times >= end] = row.yaw[-1]
            break
        end = start + plan.turn_duration
        turning = (times >= start) & (times < end)
        xy[turning], yaw[turning], v[turning], omega[turning] = _evaluate_turn(
            config, row, times[turning] - start)
        mode[turning] = "TURN"
        start = end

    validate_story_layout(config, terrain, xy)
    clearance = _all_pose_clearances(xy, yaw, config).min(axis=0)
    if clearance.min() < BODY_DESIGN_CLEARANCE_M:
        i = int(np.argmin(clearance))
        raise ValueError(f"mowing course sweep has only {clearance[i]:.3f} m clearance at {times[i]:.3f}s")

    command = (times >= .08) & (times < end)
    rotor = np.clip((times - .08) / SPIN_UP_S, 0., 1.)
    rotor = np.minimum(rotor, np.clip(1 - (times - end) / SPIN_DOWN_S, 0., 1.))
    blade = rotor >= CUTTING_ROTOR_FRACTION
    trajectory = trajectory_from_path(config, terrain, times, xy, yaw, v, omega, blade, rotor)
    story = StoryState(story_tracks(config.scenario), mode, np.isin(mode, ["REPLAN", "BYPASS"]),
                       obstacle, clearance, command, rotor)
    return trajectory, story, plan
