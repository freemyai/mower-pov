"""Coverage planning, steering limits, and replay for the multi-row yard course."""

from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from .actors.behavior import story_tracks
from .contracts import StoryState
from .controller import (CUTTING_ROTOR_FRACTION, SPIN_DOWN_S, SPIN_UP_S,
                         _half_turn, _smoothstep)
from .course_coverage import SweepGrid, SweepOption, idle_blade_commands, select_coverage_sequence
from .cutting import compute_cut_state
from .course_layout import CourseLayout, DEFAULT_COURSE, obstacle_clearance, obstacle_swept_clearance
from .motion import segments_hit_box, trajectory_from_path
from .obstacles import FENCE_ENDPOINTS, FENCE_Y, distance_to_segment, validate_story_layout


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
    coverage_report: dict
    layout: CourseLayout = DEFAULT_COURSE

    def report(self):
        start = .75
        report = []
        for i, row in enumerate(self.rows):
            stop = start + row.duration
            report.append({
                "row": i + 1, "nominal_y_m": self.layout.row_y_m[i],
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


def _profile(s, curvature, obstacle_margin, mower, *, yaw=None):
    magnitude = np.abs(curvature)
    # Use both neighbors so the bound covers interpolation along each source chord.
    slope = np.maximum.reduce((magnitude, np.r_[magnitude[0], magnitude[:-1]],
                               np.r_[magnitude[1:], magnitude[-1]]))
    if yaw is not None:
        # The executor interpolates heading linearly in arclength. Bound the
        # slope of each actual interpolant at both velocity knots of its chord.
        chord_slope = abs(np.diff(yaw) / np.diff(s))
        slope = np.maximum.reduce((slope, np.r_[chord_slope[0], chord_slope],
                                   np.r_[chord_slope, chord_slope[-1]]))
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


def _all_pose_clearances(xy, yaw, config, layout):
    distances = [obstacle_swept_clearance(xy, yaw, config, item) for item in layout.obstacles]
    return np.minimum.reduce(distances) if distances else np.full(len(xy), config.world.width_m + config.world.depth_m)


def _cycloid_step(x, begin, end):
    u = np.clip((x - begin) / (end - begin), 0., 1.)
    return u - np.sin(2 * np.pi * u) / (2 * np.pi)


def _row_shapes(x, lane_y, nominal, collision_columns, obstacles):
    if not collision_columns.any():
        yield np.zeros_like(x), 0., None
        if not obstacles:
            return
        # An unobstructed lane may collect grass skipped by another lane's detour.
        for entry, exit in ((-.85, -.20), (-.18, .40), (.32, .67), (-.77, .52)):
            if not x[0] < entry < exit < x[-1]:
                continue
            shape = (_smoothstep(x, max(x[0], entry - .72), entry)
                     - _smoothstep(x, exit, min(x[-1], exit + .72)))
            for offset in (-.22, -.15, -.08, .08, .15, .22):
                yield shape * offset, offset, None
    else:
        threat_x = x[collision_columns]
        ramp_pairs = ((.45, .51), (.58, .66), (.72, .72), (.92, .92),
                      (.48, .89), (.89, .48))
        for expansion, (left, right) in product((.025, .13, .22), ramp_pairs):
            entry = max(x[0] + .03, threat_x[0] - expansion)
            exit = min(x[-1] - .03, threat_x[-1] + expansion)
            start = max(x[0], entry - left)
            stop = min(x[-1], exit + right)
            shape = _smoothstep(x, start, entry) - _smoothstep(x, exit, stop)
            for magnitude in np.arange(.10, .66, .03):
                for sign in (-1, 1):
                    offset = float(sign * magnitude)
                    yield shape * offset, offset, (start, entry, exit, stop)

    if not obstacles:
        return
    # A second family can begin and end at different distances from an obstacle.
    # Its hold points follow the prop itself, with the full box sweep checked below.
    obstacle = obstacles[int(np.argmin(nominal.min(axis=1)))]
    points = np.asarray(obstacle.points)
    close = points[abs(points[:, 1] - lane_y) < .26 + obstacle.radius_m]
    pivot = float(np.median(close[:, 0])) if len(close) else float(points[0, 0])
    ramp_lengths = (.53, .73, .85, .97, 1.18, 1.48, 1.90)
    seen = set()
    for skew, half_width, left, right in product((-.16, 0., .16), (.12, .26), ramp_lengths, ramp_lengths):
        entry, exit = pivot + skew - half_width, pivot + skew + half_width
        start, stop = max(x[0], entry - left), min(x[-1], exit + right)
        if min(entry - start, stop - exit) < .33:
            continue
        knots = (start, entry, exit, stop)
        if knots in seen:
            continue
        seen.add(knots)
        offsets = np.arange(-.65, .681, .06) if collision_columns.any() else (-.19, -.12, -.06, .06, .12, .19)
        transitions = (_cycloid_step, _smoothstep) if half_width == .12 else (_cycloid_step,)
        for curve in transitions:
            shape = curve(x, start, entry) - curve(x, exit, stop)
            for offset in offsets:
                if abs(offset) > .025:
                    yield shape * offset, float(offset), knots if collision_columns.any() else None


def _candidate_clearance(xy, yaw, config, bounds, design_clearance, yard_context=True):
    corner = .5 * np.hypot(config.mower.length_m, config.mower.width_m)
    chords = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    step = .5 * (chords + corner * abs(np.diff(yaw)))
    pad = np.maximum(np.r_[step, 0.], np.r_[0., step])
    # Include the configured garden boundary and flower bed before scoring. A
    # bad candidate must not win the search and fail only at final validation.
    margin = corner + .025 + .5 * chords.max()
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    edge = np.array([config.world.width_m, config.world.depth_m]) * .5 - margin
    if np.any(lower <= -edge) or np.any(upper >= edge):
        return None
    bed = np.asarray(config.world.flower_bed_xyxy_m)
    bed_min, bed_max = bed[:2] - margin, bed[2:] + margin
    overlaps_bounds = np.all((upper >= bed_min) & (lower <= bed_max))
    if overlaps_bounds and segments_hit_box(xy[:-1], xy[1:], bed_min, bed_max).any():
        return None
    # The play rail belongs to the surrounding yard rather than the course props.
    if yard_context:
        a, b = FENCE_ENDPOINTS
        rail = distance_to_segment(xy, (a, FENCE_Y), (b, FENCE_Y))
        if np.min(rail - pad) < corner + .10:
            return None
    nearest = np.full(len(xy), np.inf)
    for obstacle, low, high in bounds:
        box_delta = np.maximum(np.maximum(low - xy, xy - high), 0.)
        lower_bound = np.linalg.norm(box_delta, axis=1) - corner - obstacle.radius_m - pad
        relevant = np.flatnonzero(lower_bound < .33)
        if not len(relevant):
            continue
        coarse = relevant[::7]
        if np.min(obstacle_clearance(xy[coarse], yaw[coarse], config, obstacle) - pad[coarse]) < design_clearance:
            return None
        exact = obstacle_clearance(xy[relevant], yaw[relevant], config, obstacle) - pad[relevant]
        if exact.min() < design_clearance:
            return None
        nearest[relevant] = np.minimum(nearest[relevant], exact)
    return nearest


def _row_candidates(config, grid, x, lane_y, eastbound, layout):
    nominal_xy = np.column_stack((x, np.full_like(x, lane_y)))
    nominal = np.asarray([obstacle_clearance(nominal_xy, np.zeros(len(x)), config, item)
                          for item in layout.obstacles]).reshape(-1, len(x))
    planning_clearance = layout.clearance_m + .001
    blocked = nominal < planning_clearance + .002
    columns = blocked.any(axis=0)
    if not layout.yard_context:
        x0, y0, x1, y1 = config.world.flower_bed_xyxy_m
        half_length, half_width = config.mower.length_m / 2, config.mower.width_m / 2
        if y0 - half_width <= lane_y <= y1 + half_width:
            columns |= (x >= x0 - half_length) & (x <= x1 + half_length)
    blockers = tuple(obstacle.name for obstacle, collision in zip(layout.obstacles, blocked)
                     if collision.any())
    unique = {}
    tried = 0
    bounds = [(item, np.min(item.points, axis=0), np.max(item.points, axis=0))
              for item in reversed(layout.obstacles)]
    for deviation, offset, bypass in _row_shapes(x, lane_y, nominal, columns, layout.obstacles):
        xy, s, yaw, curvature = _row_geometry(x, lane_y + deviation, eastbound)
        tried += 1
        # Reject obvious collisions before checking all chords of the candidate.
        clearance = _candidate_clearance(xy, yaw, config, bounds, planning_clearance, layout.yard_context)
        if clearance is None:
            continue
        speed, times = _profile(s, curvature, clearance, config.mower,
                                yaw=None if layout.yard_context else yaw)
        mask = grid.footprint(xy, yaw)
        former = unique.get(mask)
        if former is None or times[-1] < former.duration:
            unique[mask] = Row(xy, s, yaw, curvature, speed, times, offset, blockers,
                               bypass if blockers else None, 0)
    if not unique:
        raise NoFeasiblePlan(f"No clear course candidate for row y={lane_y:.2f} m around {', '.join(blockers)}")
    options = []
    for mask, row in unique.items():
        row.candidate_count = tried
        options.append((SweepOption(mask, row.duration, float(row.s[-1])), row))
    return options


class NoFeasiblePlan(ValueError):
    """No drivable candidate in the supplied route family."""


class CourseBudgetExceeded(ValueError):
    """The route exceeds the supplied time budget."""


def turn_duration_s(mower, turn_radius):
    if not turn_radius:
        return 0.
    omega = min(mower.max_omega_radps, mower.max_speed_mps / turn_radius)
    alpha = min(mower.max_alpha_radps2, mower.max_accel_mps2 / turn_radius)
    peak = min(omega, np.sqrt(np.pi * alpha))
    return np.pi / peak + peak / alpha


def check_course_turns(config, rows, layout, duration):
    bounds = [(o, np.min(o.points, axis=0), np.max(o.points, axis=0)) for o in layout.obstacles]
    for row in rows[:-1]:
        xy, yaw, _, _ = _evaluate_turn(config, row, np.linspace(0., duration, 181), layout)
        if _candidate_clearance(xy, yaw, config, bounds, layout.clearance_m, layout.yard_context) is None:
            raise NoFeasiblePlan("a fixed end-of-row turn conflicts with the static layout")


def plan_course(config, layout=DEFAULT_COURSE):
    """Check the course's known obstacles and choose drivable bypasses."""
    mower = config.mower
    turn_duration = turn_duration_s(mower, layout.turn_radius_m)
    count = len(layout.row_y_m)
    overhead = .75 + (count - 1) * turn_duration + SPIN_DOWN_S + .35
    span = layout.x_limits_m[1] - layout.x_limits_m[0]
    lower_time = overhead + count * span / mower.max_speed_mps
    if config.world.duration_s < lower_time:
        raise CourseBudgetExceeded(f"mowing course needs at least {lower_time:.2f} seconds with the configured motion limits")
    x = np.linspace(*layout.x_limits_m, 527 if layout.yard_context else 181)
    grid = SweepGrid(config, layout)
    candidates = [_row_candidates(config, grid, x, lane, index % 2 == 0, layout)
                  for index, lane in enumerate(layout.row_y_m)]
    options = [tuple(option for option, _ in row) for row in candidates]
    fastest = overhead + sum(min(option.duration for option in row) for row in options)
    turn_masks = []
    if not layout.yard_context:
        check_course_turns(config, tuple(row[0][1] for row in candidates), layout, turn_duration)
    for row in candidates[:-1]:
        xy, yaw, _, _ = _evaluate_turn(config, row[0][1], np.linspace(0., turn_duration, 181), layout)
        turn_masks.append(grid.footprint(xy, yaw))
    indices = select_coverage_sequence(options, grid.target, grid.cell_m ** 2,
                                       config.world.duration_s - overhead, fixed_sweeps=turn_masks,
                                       repeat_zone=grid.interior)
    if indices is None:
        raise CourseBudgetExceeded(f"mowing course needs {fastest:.2f} seconds with the configured motion limits")
    rows = tuple(candidates[i][j][1] for i, j in enumerate(indices))
    seen = repeated_mask = repeat = 0
    gains = []
    for index, (options_row, option_index) in enumerate(zip(options, indices)):
        mask = options_row[option_index].mask
        gains.append((mask & grid.target & ~seen).bit_count() * grid.cell_m ** 2)
        repeated_mask |= seen & mask
        repeat += (seen & mask).bit_count()
        seen |= mask
        if index < len(turn_masks):
            repeated_mask |= seen & turn_masks[index]
            repeat += (seen & turn_masks[index]).bit_count()
            seen |= turn_masks[index]
    report = {"objective": "marginal-swept-area", "planning_cell_m": grid.cell_m,
              "distinct_valid_options_per_row": [len(row) for row in options],
              "revisited_area_m2": repeated_mask.bit_count() * grid.cell_m ** 2,
              "new_target_area_per_row_m2": gains, "repeated_sweep_m2": repeat * grid.cell_m ** 2}
    return CoursePlan(rows, turn_duration, overhead + sum(row.duration for row in rows), report, layout)


def _evaluate_row(row, seconds, layout):
    index = np.minimum(np.searchsorted(row.times, seconds, side="right") - 1, len(row.times) - 2)
    u = seconds - row.times[index]
    acceleration = (row.speed[index + 1] - row.speed[index]) / (row.times[index + 1] - row.times[index])
    v = row.speed[index] + acceleration * u
    s = row.s[index] + u * (row.speed[index] + .5 * acceleration * u)
    xy = np.column_stack([np.interp(s, row.s, values) for values in row.xy.T])
    yaw = np.interp(s, row.s, row.yaw)
    omega = v * np.interp(s, row.s, row.curvature)
    modes = np.full(len(s), "MOW", dtype="U8")
    width = max([12, *(len(name) for name in row.nominal_blockers)])
    names = np.full(len(s), "", dtype=f"U{width}")
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
        relevant = [item for item in layout.obstacles if item.name in row.nominal_blockers]
        pivot = np.array([np.mean(np.asarray(item.points)[:, 0]) for item in relevant])
        closest = np.argmin(np.abs(xy[:, :1] - pivot), axis=1)
        on_change = modes != "MOW"
        names[on_change] = np.array(row.nominal_blockers)[closest[on_change]]
    return xy, yaw, v, omega, modes, names


def _evaluate_turn(config, row, seconds, layout):
    m = config.mower
    radius = layout.turn_radius_m
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


def run_course(config, terrain, plan=None):
    plan = plan_course(config) if plan is None else plan
    layout = plan.layout
    times = np.arange(round(config.world.duration_s * config.world.step_hz) + 1) / config.world.step_hz
    xy = np.tile(plan.rows[0].xy[0], (len(times), 1))
    yaw, v, omega = (np.zeros(len(times)) for _ in range(3))
    mode = np.full(len(times), "PARK", dtype="U8")
    name_width = max([12, *(len(item.name) for item in layout.obstacles)])
    obstacle = np.full(len(times), "", dtype=f"U{name_width}")
    start = .75
    for index, row in enumerate(plan.rows):
        end = start + row.duration
        moving = (times >= start) & (times < end)
        xy[moving], yaw[moving], v[moving], omega[moving], mode[moving], obstacle[moving] = (
            _evaluate_row(row, times[moving] - start, layout))
        start = end
        if index + 1 == len(plan.rows):
            xy[times >= end] = row.xy[-1]
            yaw[times >= end] = row.yaw[-1]
            break
        end = start + plan.turn_duration
        turning = (times >= start) & (times < end)
        xy[turning], yaw[turning], v[turning], omega[turning] = _evaluate_turn(
            config, row, times[turning] - start, layout)
        mode[turning] = "TURN"
        start = end

    if layout.yard_context:
        validate_story_layout(config, terrain, xy)
    clearance = _all_pose_clearances(xy, yaw, config, layout)
    if layout.yard_context and clearance.min() < layout.clearance_m:
        i = int(np.argmin(clearance))
        raise NoFeasiblePlan(f"mowing course sweep has only {clearance[i]:.3f} m clearance at {times[i]:.3f}s")

    command = (times >= .08) & (times < end)
    rotor = np.clip((times - .08) / SPIN_UP_S, 0., 1.)
    rotor = np.minimum(rotor, np.clip(1 - (times - end) / SPIN_DOWN_S, 0., 1.))
    blade = rotor >= CUTTING_ROTOR_FRACTION
    trajectory = trajectory_from_path(config, terrain, times, xy, yaw, v, omega, blade, rotor)
    tracks = story_tracks(config.scenario) if layout.yard_context else np.empty((0, 6))
    story = StoryState(tracks, mode, np.isin(mode, ["REPLAN", "BYPASS"]),
                       obstacle, clearance, command, rotor)
    return trajectory, story, plan


def coast_over_cut_grass(config, terrain, trajectory, story, roots, state):
    times = trajectory.time_s
    cutters = np.concatenate((state.t_first_cut_s.ravel(), state.root_first_cut_s))
    command, windows = idle_blade_commands(times, story.blade_command, cutters)
    if not windows:
        return trajectory, story, state, []

    rotor = story.rotor_fraction.copy()
    dt = times[1] - times[0]
    for first, restart in windows:
        settle = min(len(times), restart + int(np.ceil(SPIN_UP_S / dt)) + 2)
        for i in range(first + 1, settle):
            rate = 1 / SPIN_UP_S if command[i - 1] else -1 / SPIN_DOWN_S
            rotor[i] = np.clip(rotor[i - 1] + dt * rate, 0., 1.)
    replay = trajectory_from_path(config, terrain, times, trajectory.position_m[:, :2],
                                  trajectory.unwrapped_yaw, trajectory.v_mps, trajectory.omega_radps,
                                  rotor >= CUTTING_ROTOR_FRACTION, rotor)
    cut = compute_cut_state(config, terrain, replay, roots)
    # Noncentral blades can shift slightly as rotor vibration changes. Keep
    # every first-cut timestamp if a proposed coast would affect the sweep.
    original = (state.t_first_cut_s, state.root_first_cut_s)
    updated = (cut.t_first_cut_s, cut.root_first_cut_s)
    if not all(np.array_equal(old, new) for old, new in zip(original, updated)):
        return trajectory, story, state, []
    evidence = [{"command_off_s": float(times[i]), "command_restart_s": float(times[j]),
                 "stopped_s": float(np.count_nonzero(rotor[i:j] == 0) * dt)} for i, j in windows]
    return replay, replace(story, blade_command=command, rotor_fraction=rotor), cut, evidence
