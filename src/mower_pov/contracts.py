"""Shared data contracts. Time is absolute seconds, coordinates are right-handed."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class WorldConfig:
    width_m: float
    depth_m: float
    duration_s: float
    step_hz: int
    cell_m: float
    flower_bed_xyxy_m: tuple[float, float, float, float]


@dataclass(frozen=True)
class MowerConfig:
    length_m: float
    width_m: float
    track_m: float
    max_speed_mps: float
    max_accel_mps2: float
    max_omega_radps: float
    max_alpha_radps2: float
    blade_radius_m: float
    blade_xy_m: tuple[float, float]
    row_spacing_m: float


@dataclass(frozen=True)
class CameraConfig:
    mount_xyz_m: tuple[float, float, float]
    hfov_deg: float
    pitch_deg: float


@dataclass(frozen=True)
class GrassConfig:
    min_height_m: float
    max_height_m: float
    cut_height_m: float
    roots_per_m2: int


@dataclass(frozen=True)
class RenderConfig:
    width: int
    height: int
    fps: int
    pixel_format: str
    engine_version: str
    samples: int


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int
    channels: int


@dataclass(frozen=True)
class RunConfig:
    seed: int
    world: WorldConfig
    mower: MowerConfig
    camera: CameraConfig
    grass: GrassConfig
    render: RenderConfig
    audio: AudioConfig
    scenario: str = "legacy"

    @property
    def frame_count(self) -> int:
        return round(self.world.duration_s * self.render.fps)

    @property
    def audio_frames(self) -> int:
        return round(self.world.duration_s * self.audio.sample_rate_hz)


EnvironmentReport = dict[str, object]




@dataclass(frozen=True)
class Terrain:
    # Cell centers, arrays indexed [row along +Y, column along +X].
    height_m: np.ndarray
    origin_xy_m: np.ndarray
    cell_size_m: float
    mowable_mask: np.ndarray
    obstacle_xyxy_m: tuple[float, float, float, float]


@dataclass(frozen=True)
class BladeRoots:
    root_id: np.ndarray
    root_xyz_m: np.ndarray
    h0_m: np.ndarray
    width_m: np.ndarray
    bend: np.ndarray
    phase: np.ndarray
    azimuth_rad: np.ndarray
    tint: np.ndarray
    dry_tip: np.ndarray


@dataclass(frozen=True)
class ControlSchedule:
    time_s: np.ndarray
    v_mps: np.ndarray
    omega_radps: np.ndarray
    blade_enabled: np.ndarray


@dataclass(frozen=True)
class Trajectory:
    time_s: np.ndarray
    position_m: np.ndarray
    orientation: np.ndarray
    unwrapped_yaw: np.ndarray
    pitch_rad: np.ndarray
    roll_rad: np.ndarray
    v_mps: np.ndarray
    omega_radps: np.ndarray
    wheel_speeds: np.ndarray
    blade_enabled: np.ndarray


@dataclass(frozen=True)
class Violation:
    time_s: float
    reason: str


@dataclass(frozen=True)
class CutState:
    t_first_cut_s: np.ndarray
    cut_height_m: float
    root_first_cut_s: np.ndarray
    root_grid_rc: np.ndarray
    wheel_contact_time_s: np.ndarray
    wheel_contact_xy_m: np.ndarray
    wheel_width_m: float

    @property
    def wheel_contact_count(self) -> int:
        return 2 * len(self.wheel_contact_time_s)


@dataclass(frozen=True)
class Event:
    time_s: float
    kind: str
    position_m: tuple[float, float, float]
    intensity: float
    payload: tuple[tuple[str, int | float], ...] = ()


@dataclass(frozen=True)
class EventLog:
    events: tuple[Event, ...]

@dataclass(frozen=True)
class StoryState:
    # Tracks hold actor rows [start_s, end_s, start_x, start_y, end_x, end_y].
    tracks: np.ndarray
    mode: np.ndarray
    blocked: np.ndarray
    obstacle_name: np.ndarray
    clearance_m: np.ndarray
    blade_command: np.ndarray
    rotor_fraction: np.ndarray


@dataclass(frozen=True)
class RunBundle:
    run_id: str
    run_dir: Path
    resolved_config: RunConfig
    terrain: Terrain
    trajectory: Trajectory
    roots: BladeRoots
    cut_state: CutState
    events: EventLog
    manifest: Mapping[str, object]
    story: StoryState | None = None
