"""Fixed-step differential drive and terrain contact pose."""

import numpy as np

from .contracts import ControlSchedule, RunConfig, Terrain, Trajectory, Violation
from .terrain import sample_height


def make_demo_schedule(config: RunConfig, terrain: Terrain) -> ControlSchedule:
    """A straight outbound pass, a 180-degree turn, and a neighboring return."""
    duration = config.world.duration_s
    t = np.array([0, 1, 2, 3, 15.8, 17, 17.75, 21.45, 22.2, 23.3, 36, 39, 40]) * duration / 40
    m = config.mower
    turn_omega = np.pi / (4.45 * duration / 40)
    turn_speed = m.row_spacing_m / 2 * turn_omega
    speed = np.array([0, 0, 0, m.max_speed_mps * .9, m.max_speed_mps * .9, 0,
                      turn_speed, turn_speed, 0, m.max_speed_mps * .9,
                      m.max_speed_mps * .9, .16, 0])
    omega = np.zeros_like(t)
    omega[6:8] = turn_omega
    blade = (t >= t[2]) & (t < duration)
    return ControlSchedule(t, speed, omega, blade)


def wheel_speeds(v, omega, track):
    delta = np.asarray(omega) * track / 2
    return np.stack((v - delta, v + delta), axis=-1)


def _slew(target, speed, delta):
    target = np.clip(target, -speed, speed)
    result = np.zeros_like(target)
    for i in range(1, len(result)):
        change = min(max(target[i] - result[i - 1], -delta), delta)
        result[i] = result[i - 1] + change
    return result


def _rotation(yaw, pitch, roll):
    cy, cp, cr = np.cos(yaw), np.cos(pitch), np.cos(roll)
    sy, sp, sr = np.sin(yaw), np.sin(pitch), np.sin(roll)
    rows = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return np.moveaxis(np.array(rows), (0, 1), (-2, -1))


def integrate_motion(config: RunConfig, terrain: Terrain, controls: ControlSchedule) -> Trajectory:
    m, w = config.mower, config.world
    n = len(controls.time_s)
    if (n == 0 or any(a.shape != (n,) for a in (controls.time_s, controls.v_mps,
            controls.omega_radps, controls.blade_enabled)) or controls.time_s[0] != 0
            or np.any(np.diff(controls.time_s) <= 0)
            or not all(np.isfinite(a).all() for a in (controls.time_s, controls.v_mps,
                                                    controls.omega_radps))):
        raise ValueError("controls require finite, strictly increasing times starting at zero")
    dt = 1 / w.step_hz
    t = np.arange(round(w.duration_s * w.step_hz) + 1) / w.step_hz
    v = _slew(np.interp(t, controls.time_s, controls.v_mps), m.max_speed_mps, m.max_accel_mps2 * dt)
    omega = _slew(np.interp(t, controls.time_s, controls.omega_radps),
                  m.max_omega_radps, m.max_alpha_radps2 * dt)
    blade = controls.blade_enabled[np.searchsorted(controls.time_s, t, side="right") - 1]
    angle_step = (omega[:-1] + omega[1:]) * (.5 * dt)
    yaw = np.r_[0., np.cumsum(angle_step)]
    halfway = yaw[:-1] + angle_step / 2
    distance = (v[:-1] + v[1:]) * (.5 * dt) * np.sinc(angle_step / (2 * np.pi))
    delta = distance[:, None] * np.column_stack((np.cos(halfway), np.sin(halfway)))
    xy = np.vstack((np.zeros(2), np.cumsum(delta, axis=0)))
    return trajectory_from_path(config, terrain, t, xy, yaw, v, omega, blade)

def trajectory_from_path(config, terrain, t, xy, yaw, v, omega, blade, rotor=None):
    m = config.mower
    dt = 1 / config.world.step_hz
    forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
    left = forward[:, ::-1] * [-1, 1]
    # Four contact points approximate the support plane under the chassis.
    half_length, half_track = .36 * m.length_m, .5 * m.track_m
    front, rear = forward * half_length, -forward * half_length
    side = left * half_track
    fl, fr, rl, rr = (
        sample_height(terrain, xy + offset)
        for offset in (front + side, front - side, rear + side, rear - side)
    )
    target = np.column_stack(((fl + fr + rl + rr) / 4,
                              -np.arctan2(fl + fr - rl - rr, 4 * half_length),
                              np.arctan2(fl + rl - fr - rr, 4 * half_track)))
    support = target.copy()
    gain = -np.expm1(-dt / .09)
    for i in range(1, len(t)):
        support[i] = support[i - 1] + gain * (target[i] - support[i - 1])
    vibration_level = blade if rotor is None else rotor
    vibration = vibration_level * np.sin(2 * np.pi * 16 * t)
    pitch = support[:, 1] + .0006 * vibration
    roll = support[:, 2] + .0004 * vibration_level * np.sin(2 * np.pi * 19 * t)
    return Trajectory(t, np.column_stack((xy, support[:, 0])), _rotation(yaw, pitch, roll),
                      yaw, pitch, roll, v, omega, wheel_speeds(v, omega, m.track_m), blade)


def sample_pose(trajectory: Trajectory, time_s: float) -> np.ndarray:
    t = trajectory.time_s
    if not t[0] <= time_s <= t[-1]:
        raise ValueError("pose time outside trajectory")
    i = min(np.searchsorted(t, time_s, side="right") - 1, len(t) - 2)
    u = (time_s - t[i]) / (t[i + 1] - t[i])
    pose = np.eye(4)
    angles = (trajectory.unwrapped_yaw, trajectory.pitch_rad, trajectory.roll_rad)
    pose[:3, :3] = _rotation(*(a[i] + u * (a[i + 1] - a[i]) for a in angles))
    pose[:3, 3] = trajectory.position_m[i] + u * (
        trajectory.position_m[i + 1] - trajectory.position_m[i])
    return pose


def segments_hit_box(start, end, lower, upper):
    """Slab intersection, including segments parallel to either axis."""
    delta = end - start
    entry = np.zeros(len(start))
    exit = np.ones(len(start))
    for axis in range(2):
        moving = delta[:, axis] != 0
        outside = (start[:, axis] < lower[axis]) | (start[:, axis] > upper[axis])
        entry[~moving & outside] = np.inf
        a = (lower[axis] - start[moving, axis]) / delta[moving, axis]
        b = (upper[axis] - start[moving, axis]) / delta[moving, axis]
        entry[moving] = np.maximum(entry[moving], np.minimum(a, b))
        exit[moving] = np.minimum(exit[moving], np.maximum(a, b))
    return entry <= exit


def validate_swept_body(config: RunConfig, terrain: Terrain, trajectory: Trajectory) -> list[Violation]:
    m, w = config.mower, config.world
    radius = np.hypot(m.length_m, m.width_m) / 2
    xy = trajectory.position_m[:, :2]
    low = terrain.origin_xy_m + radius
    high = terrain.origin_xy_m + [w.width_m, w.depth_m] - radius
    bed = np.asarray(terrain.obstacle_xyxy_m)
    hit_bed = segments_hit_box(xy[:-1], xy[1:], bed[:2] - radius, bed[2:] + radius)
    outside = np.any((xy < low) | (xy > high), axis=1)
    return ([Violation(float(trajectory.time_s[i]), "yard boundary") for i in np.flatnonzero(outside)]
            + [Violation(float(trajectory.time_s[i]), "swept flower bed")
               for i in np.flatnonzero(hit_bed)])
