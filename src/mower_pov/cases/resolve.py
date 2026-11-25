"""Canonical scenario identity, independent of the file's checkout path."""

from dataclasses import asdict, replace
from hashlib import sha256
import json

from .contracts import ResolvedCase, ScenarioSpec


def resolve_case(spec: ScenarioSpec, seed: int) -> ResolvedCase:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an unsigned 63-bit integer")
    return replace(spec.case, seed=seed)


def scenario_digest(case: ResolvedCase) -> str:
    return sha256(json.dumps(asdict(case), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()
