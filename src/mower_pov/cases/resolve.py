"""Canonical scenario identity, independent of the file's checkout path."""

from dataclasses import asdict, replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from .contracts import ResolvedCase, ScenarioSpec


def resolve_case(spec: ScenarioSpec, seed: int) -> ResolvedCase:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an unsigned 63-bit integer")
    return replace(spec.case, seed=seed)


def scenario_digest(case: ResolvedCase) -> str:
    return canonical_digest(asdict(case))


def canonical_digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


_KERNEL_FILES = (
    "actors/behavior.py", "cache.py", "contracts.py", "controller.py", "course.py",
    "course_coverage.py", "course_layout.py", "cutting.py", "events.py", "motion.py",
    "obstacles.py", "terrain.py", "policies/base.py", "policies/legacy.py",
    "cases/contracts.py", "cases/legacy.py", "cases/resolve.py",
)
_JUDGE_FILES = ("cases/evaluation.py", "cases/geometry.py", "cases/metrics.py")
_RUNNER_FILES = ("cases/evidence.py", "cases/runner.py", "cases/schema.py", "cases/scenario-v1.schema.json")


@lru_cache(maxsize=3)
def source_digest(component):
    """Hash numeric and adjudication sources independently of Blender assets."""
    selected = {"kernel": _KERNEL_FILES, "judge": _JUDGE_FILES, "runner": _RUNNER_FILES}[component]
    root = Path(__file__).resolve().parents[1]
    return canonical_digest({name: sha256((root / name).read_bytes()).hexdigest()
                             for name in selected})
