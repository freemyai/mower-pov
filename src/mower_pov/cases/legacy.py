"""Replay an offline plan with the original motion and persistent cutting APIs."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from ..cache import _array_identity, _datasets
from ..contracts import BladeRoots, CutState, EventLog, RunConfig, StoryState, Terrain, Trajectory
from ..course import NoFeasiblePlan, coast_over_cut_grass, run_course
from ..cutting import compute_cut_state
from ..events import build_events, events_to_json
from ..policies.base import OfflinePlanner, StaticPlanningInput
from ..policies.legacy import case_config
from ..terrain import build_terrain, generate_blade_roots
from .contracts import ResolvedCase
from .resolve import scenario_digest


@dataclass(frozen=True)
class EpisodeTrace:
    scenario_id: str
    state_id: str
    status: str
    reason: str
    config: RunConfig
    planned_world_ok: bool
    terrain: Terrain | None = None
    trajectory: Trajectory | None = None
    roots: BladeRoots | None = None
    cut_state: CutState | None = None
    story: StoryState | None = None
    events: EventLog | None = None
    events_sha256: str | None = None


def run_offline(case: ResolvedCase, planner: OfflinePlanner) -> EpisodeTrace:
    from ..policies.legacy import case_layout

    problem = StaticPlanningInput(case.robot, case.world, case.duration_s)
    config = case_config(problem, case.seed, case.camera)
    scenario = scenario_digest(case)
    identity = {
        "seed": case.seed, "mode": case.mode, "problem": asdict(problem),
        "policy": [planner.id, planner.version],
    }
    result = planner.plan(problem)
    if result.plan is None:
        if result.status not in {"NO_FEASIBLE_PLAN", "BUDGET_EXCEEDED"}:
            raise ValueError(f"missing trajectory for policy outcome {result.status!r}")
        identity["status"] = result.status
        return EpisodeTrace(scenario, _digest(identity), result.status, result.reason, config, True)

    terrain = build_terrain(config)
    valid_layout = result.plan.layout == case_layout(problem)
    try:
        trajectory, story, _ = run_course(config, terrain, result.plan)
    except NoFeasiblePlan as exc:
        identity["status"] = "NO_FEASIBLE_PLAN"
        return EpisodeTrace(scenario, _digest(identity), "NO_FEASIBLE_PLAN", str(exc), config, valid_layout)
    roots = generate_blade_roots(config, terrain)
    cut = compute_cut_state(config, terrain, trajectory, roots)
    if planner.id != "legacy-greedy":
        trajectory, story, cut, _ = coast_over_cut_grass(config, terrain, trajectory, story, roots, cut)
    events = build_events(config, trajectory, cut, roots, story=story)
    event_hash = sha256(events_to_json(events).encode()).hexdigest()
    data = _datasets(terrain, trajectory, roots, cut, story)
    identity["arrays"] = {name:_array_identity(arrays) for name, arrays in data.items()}
    identity["events_sha256"] = event_hash
    return EpisodeTrace(scenario, _digest(identity), "COMPLETED", "", config, valid_layout, terrain,
                        trajectory, roots, cut, story, events, event_hash)


def _digest(identity):
    return sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
