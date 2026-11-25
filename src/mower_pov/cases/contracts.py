"""Immutable SI contracts for the v1 static scenario suite."""

from dataclasses import dataclass

from ..contracts import CameraConfig

STATIC_CUT_HEIGHT_M = .045


@dataclass(frozen=True)
class RobotSpec:
    length_m: float
    width_m: float
    track_m: float
    blade_radius_m: float
    max_speed_mps: float
    max_accel_mps2: float
    max_omega_radps: float
    max_alpha_radps2: float


@dataclass(frozen=True)
class ObstacleSpec:
    id: str
    type: str
    points_xy_m: tuple[tuple[float, float], ...]
    radius_m: float


@dataclass(frozen=True)
class WorldSpec:
    width_m: float
    depth_m: float
    step_hz: int
    cell_m: float
    x_limits_m: tuple[float, float]
    row_y_m: tuple[float, ...]
    obstacles: tuple[ObstacleSpec, ...]
    flower_bed_xyxy_m: tuple[float, float, float, float] | None
    clearance_m: float


@dataclass(frozen=True)
class AssertionSpec:
    name: str
    minimum_m: float | None = None
    minimum_fraction: float | None = None


@dataclass(frozen=True)
class ResolvedCase:
    version: int
    id: str
    seed: int
    mode: str
    fidelity: str
    units: tuple[str, str, str]
    frame: str
    robot: RobotSpec
    world: WorldSpec
    actors: tuple
    observation: str
    faults: tuple
    duration_s: int
    expected_outcome: str
    assertions: tuple[AssertionSpec, ...]
    assets: tuple[str, ...]
    camera: CameraConfig


@dataclass(frozen=True)
class ScenarioSpec:
    case: ResolvedCase
