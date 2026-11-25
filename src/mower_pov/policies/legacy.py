"""Adapters for the frozen row-by-row policy and its joint-search successor."""

from dataclasses import asdict, dataclass

import numpy as np

from ..cases.contracts import STATIC_CUT_HEIGHT_M
from ..contracts import AudioConfig, CameraConfig, GrassConfig, MowerConfig, RenderConfig, RunConfig, WorldConfig
from ..controller import SPIN_DOWN_S, _smoothstep
from ..course import (
    CourseBudgetExceeded, CoursePlan, NoFeasiblePlan, Row, _all_pose_clearances,
    _candidate_clearance, _profile, _row_geometry, check_course_turns, plan_course, turn_duration_s,
)
from ..course_layout import DEFAULT_COURSE, CourseLayout, CourseObstacle, obstacle_clearance
from .base import PlanResult, StaticPlanningInput

POLICY_IDS = ("legacy-greedy", "legacy-joint")
_DEFAULT_CAMERA = CameraConfig((.18, 0., .22), 100., 12.)
_PHASE_ONE_BED = (4.45, -1.6, 5.6, 1.6)


def case_config(problem, seed=0, camera=_DEFAULT_CAMERA):
    world = problem.world
    spacing = world.row_y_m[0] - world.row_y_m[1] if len(world.row_y_m) > 1 else .25
    outside = (world.width_m, world.depth_m, world.width_m + 1, world.depth_m + 1)
    return RunConfig(
        seed=seed,
        world=WorldConfig(world.width_m, world.depth_m, float(problem.duration_s), world.step_hz,
                          world.cell_m, world.flower_bed_xyxy_m or outside),
        mower=MowerConfig(**asdict(problem.robot), blade_xy_m=(0., 0.), row_spacing_m=spacing),
        camera=camera, grass=GrassConfig(.08, .12, STATIC_CUT_HEIGHT_M, 1800),
        render=RenderConfig(1920, 1080, 30, "yuv420p", "5.2.4", 28),
        audio=AudioConfig(48000, 2), scenario="mowing-course",
    )


def case_layout(problem):
    world = problem.world
    obstacles = tuple(CourseObstacle(o.id, o.points_xy_m, o.radius_m) for o in world.obstacles)
    layout = CourseLayout(world.x_limits_m, world.row_y_m, obstacles, world.clearance_m,
                          yard_context=False)
    phase_one = CourseLayout(yard_context=False)
    if layout == phase_one and (world.width_m, world.depth_m, world.flower_bed_xyxy_m) == (
        12., 10., _PHASE_ONE_BED
    ):
        return DEFAULT_COURSE
    return layout


@dataclass(frozen=True)
class LegacyPlanner:
    id: str
    version: str = "1"

    def __post_init__(self):
        if self.id not in POLICY_IDS:
            raise ValueError(f"unknown static policy {self.id!r}")

    def plan(self, problem: StaticPlanningInput) -> PlanResult:
        config, layout = case_config(problem), case_layout(problem)
        try:
            result = (plan_course(config, layout) if self.id == "legacy-joint"
                      else _greedy_course(config, layout))
        except CourseBudgetExceeded as exc:
            return PlanResult("BUDGET_EXCEEDED", reason=str(exc))
        except NoFeasiblePlan as exc:
            return PlanResult("NO_FEASIBLE_PLAN", reason=str(exc))
        return PlanResult("COMPLETED", result)


def _greedy_course(config, layout):
    """Retain f577518's centreline repeat penalty and tie-break order."""
    turn = turn_duration_s(config.mower, layout.turn_radius_m)
    overhead = .75 + (len(layout.row_y_m) - 1) * turn + SPIN_DOWN_S + .35
    span = layout.x_limits_m[1] - layout.x_limits_m[0]
    if config.world.duration_s < overhead + len(layout.row_y_m) * span / config.mower.max_speed_mps:
        raise CourseBudgetExceeded("time budget below straight-line travel and turn lower bound")
    samples = np.linspace(*layout.x_limits_m, 527 if layout.yard_context else 181)
    rows = []
    for index, lane in enumerate(layout.row_y_m):
        rows.append(_greedy_row(config, layout, samples, lane, index % 2 == 0, rows))
    required = overhead + sum(row.duration for row in rows)
    if config.world.duration_s < required:
        raise CourseBudgetExceeded(f"greedy row choices require {required:.2f}s")
    if not layout.yard_context:
        check_course_turns(config, rows, layout, turn)
    return CoursePlan(tuple(rows), turn, required, {}, layout)


def _greedy_row(config, layout, x, lane_y, eastbound, previous_rows):
    nominal_xy = np.column_stack((x, np.full_like(x, lane_y)))
    clearance = layout.clearance_m + .001
    obstructed = [obstacle_clearance(nominal_xy, np.zeros(len(x)), config, item) < clearance + .002
                  for item in layout.obstacles]
    blockers = tuple(o.name for o, mask in zip(layout.obstacles, obstructed) if mask.any())
    blocked = np.any(obstructed, axis=0) if obstructed else np.zeros(len(x), bool)
    bounds = [(o, np.min(o.points, axis=0), np.max(o.points, axis=0)) for o in layout.obstacles]
    bed = config.world.flower_bed_xyxy_m
    if not layout.yard_context and bed[1] - config.mower.width_m / 2 <= lane_y <= bed[3] + config.mower.width_m / 2:
        blocked |= (x >= bed[0] - config.mower.length_m / 2) & (x <= bed[2] + config.mower.length_m / 2)

    def make_row(offset, knots, profile, count):
        xy, s, yaw, curvature = _row_geometry(x, lane_y + offset * profile, eastbound)
        if layout.yard_context:
            # Exact historical geometry and speed sampling for the published yard.
            if any(obstacle_clearance(xy[::7], yaw[::7], config, o).min() < clearance for o in layout.obstacles):
                return None
            margin = _all_pose_clearances(xy, yaw, config, layout)
            if margin.min() < clearance:
                return None
        else:
            margin = _candidate_clearance(xy, yaw, config, bounds, clearance, False)
            if margin is None:
                return None
        speed, times = _profile(s, curvature, margin, config.mower,
                                yaw=None if layout.yard_context else yaw)
        return Row(xy, s, yaw, curvature, speed, times, offset, blockers, knots, count)

    if not blocked.any():
        straight = make_row(0., None, np.zeros_like(x), 1)
        if straight is not None:
            return straight
        raise NoFeasiblePlan(f"nominal row {lane_y:.3f}m lies outside the drivable yard")

    threat = x[blocked]
    winner, best, tried = None, np.inf, 0
    ramp_pairs = ((.45, .51), (.58, .66), (.72, .72), (.92, .92), (.48, .89), (.89, .48))
    for padding in (.025, .13, .22):
        entry, exit_ = max(x[0] + .03, threat[0] - padding), min(x[-1] - .03, threat[-1] + padding)
        for left, right in ramp_pairs:
            start, stop = max(x[0], entry - left), min(x[-1], exit_ + right)
            profile = _smoothstep(x, start, entry) - _smoothstep(x, exit_, stop)
            for magnitude in np.arange(.10, .66, .03):
                for sign in (-1, 1):
                    tried += 1
                    knots = (start, entry, exit_, stop) if blockers else None
                    candidate = make_row(float(sign * magnitude), knots, profile, tried)
                    if candidate is None:
                        continue
                    already = np.zeros(len(x), bool)
                    for previous in previous_rows:
                        px, py = previous.xy.T
                        order = slice(None) if px[0] < px[-1] else slice(None, None, -1)
                        prior_y = np.interp(candidate.xy[:, 0], px[order], py[order])
                        already |= abs(candidate.xy[:, 1] - prior_y) < config.mower.blade_radius_m * 1.20
                    repeat = np.sum(np.diff(candidate.s) * already[1:])
                    score = (candidate.duration + .85 * magnitude
                             + (candidate.s[-1] - (x[-1] - x[0])) + 1.40 * repeat)
                    if score < best:
                        best, winner = score, candidate
    if winner is None:
        raise NoFeasiblePlan(f"no greedy course candidate for row {lane_y:.3f}m")
    winner.candidate_count = tried
    return winner
