"""Inputs visible to a static policy and its typed outcomes."""

from dataclasses import dataclass
from typing import Literal, Protocol

from ..cases.contracts import RobotSpec, WorldSpec
from ..course import CoursePlan

PlanStatus = Literal["COMPLETED", "NO_FEASIBLE_PLAN", "BUDGET_EXCEEDED"]


@dataclass(frozen=True)
class StaticPlanningInput:
    robot: RobotSpec
    world: WorldSpec
    duration_s: int


@dataclass(frozen=True)
class PlanResult:
    status: PlanStatus
    plan: CoursePlan | None = None
    reason: str = ""


class OfflinePlanner(Protocol):
    id: str
    version: str

    def plan(self, problem: StaticPlanningInput) -> PlanResult: ...
