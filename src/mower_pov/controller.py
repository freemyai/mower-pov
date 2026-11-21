"""Replayable truth-input controller for the backyard scenarios."""

import numpy as np

from .actors.behavior import ACTOR_NAMES, story_tracks
from .contracts import StoryState
from .motion import trajectory_from_path
from .obstacles import (ACTOR_RADII, DETOUR_TOY_XY, distance_to_segment,
                        validate_actor_paths, validate_story_layout)

CLEAR_CONFIRM_S = .65
SPIN_UP_S = .65
SPIN_DOWN_S = .72
CUTTING_ROTOR_FRACTION = .28


def _smoothstep(x, start, end):
    u = np.clip((x - start) / (end - start), 0., 1.)
    return u ** 3 * (10 + u * (-15 + 6 * u))


def _path(config, terrain):
    if config.scenario == "toy-detour":
        x = np.linspace(-2.35, 4.0, 2501)
        original_y = np.full_like(x, DETOUR_TOY_XY[1])
        try:
            validate_story_layout(config, terrain, np.column_stack((x, original_y)))
        except ValueError as error:
            if "toy footprint" not in str(error):
                raise
        else:
            return x, original_y, None
        shift = .83 * (_smoothstep(x, -1.90, .20) - _smoothstep(x, 1.62, 3.65))
        for sign in (1, -1):
            y = original_y + sign * shift
            try:
                validate_story_layout(config, terrain, np.column_stack((x, y)))
            except ValueError:
                continue
            return x, y, (-1.90, 3.65)
        raise ValueError("Backyard layout conflict — neither path around the toy clears the footprint")

    x = np.linspace(-1.40, 4.25, 2401)
    y = 3.58 - .72 * _smoothstep(x, -1.27, .60)
    validate_story_layout(config, terrain, np.column_stack((x, y)))
    return x, y, None


def _half_turn(elapsed, max_omega, max_alpha):
    peak = min(max_omega, np.sqrt(np.pi * max_alpha))
    ramp = peak / max_alpha
    cruise = np.pi / peak - ramp
    duration = 2 * ramp + cruise
    if elapsed < ramp:
        return .5 * max_alpha * elapsed ** 2, max_alpha * elapsed
    if elapsed < ramp + cruise:
        return peak * (elapsed - .5 * ramp), peak
    remaining = max(0., duration - elapsed)
    return np.pi - .5 * max_alpha * remaining ** 2, max_alpha * remaining


def _actor_positions(times, tracks):
    begin = tracks[:, 2:4]
    end = tracks[:, 4:6]
    progress = np.clip((times[:, None] - tracks[:, 0]) / (tracks[:, 1] - tracks[:, 0]), 0., 1.)
    blend = progress ** 2 * (3 - 2 * progress)
    return begin + blend[..., None] * (end - begin)


def run_controller(config, terrain):
    """Use forward occupancy, acceleration-limited chassis motion, and measured rotor state."""
    m, w = config.mower, config.world
    dt = 1 / w.step_hz
    times = np.arange(round(w.duration_s * w.step_hz) + 1) * dt
    tracks = story_tracks(config.scenario)
    validate_actor_paths(config, terrain, tracks)
    actor_xy = _actor_positions(times, tracks)
    path_x, path_y, detour = _path(config, terrain)
    path = np.column_stack((path_x, path_y))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    yaw_path = np.arctan(np.gradient(path_y, path_x))
    curvature = np.gradient(yaw_path, arc)

    def at(s):
        return np.array([np.interp(s, arc, path_x), np.interp(s, arc, path_y)])

    body_radius = .5 * np.hypot(m.length_m, m.width_m)
    radii = body_radius + np.array(ACTOR_RADII)
    position = np.zeros((len(times), 2))
    distance, velocity, rotor = (np.zeros_like(times) for _ in range(3))
    command = np.zeros(len(times), dtype=bool)
    blocked = np.zeros_like(command)
    modes = np.full(len(times), "MOW", dtype="U13")
    names = np.full(len(times), "", dtype="U12")
    path_direction = np.ones(len(times))
    turn_angle = np.zeros(len(times))
    turn_rate = np.zeros(len(times))
    position[0] = path[0]
    mode = "MOW"
    turning_since = None
    returning = False
    clear_since = 0.
    last_obstacle = ""
    lookahead = np.linspace(0, 1.22, 6) * w.step_hz
    footprint_margin = .105
    max_v = min(.30, m.max_speed_mps, m.max_omega_radps / max(.01, np.max(np.abs(curvature))))
    for i, time in enumerate(times):
        s = distance[i]
        center = position[i]
        speed = velocity[i]
        if (config.scenario != "toy-detour" and turning_since is None
                and arc[-1] - s < .008 and speed <= 1e-9):
            turning_since = time
        if turning_since is not None:
            turn_angle[i], turn_rate[i] = _half_turn(
                time - turning_since, m.max_omega_radps, m.max_alpha_radps2)
            returning = turn_angle[i] >= np.pi
        path_direction[i] = -1. if returning else 1.
        turning = turning_since is not None and not returning
        stopping = speed ** 2 / (2 * m.max_accel_mps2) + speed * .14
        reach = stopping + .20
        # Keep the clear-check at least as large as the approach region, even at rest.
        reach = max(reach, max_v ** 2 / (2 * m.max_accel_mps2) + .20)
        future = actor_xy[np.minimum(i + lookahead.astype(int), len(times) - 1)]
        # Evaluate successive legs of the forward path rather than crossing a bend's chord.
        look = 0. if turning else path_direction[i] * reach
        mid, tip = at(s + look / 2), at(s + look)
        clearance = np.minimum(distance_to_segment(future, center, mid),
                               distance_to_segment(future, mid, tip)) - radii
        occupied = np.min(clearance, axis=0) <= footprint_margin
        if occupied.any():
            last_obstacle = ACTOR_NAMES[int(np.argmin(clearance.min(axis=0)))]
            mode = "WAIT" if speed <= 1e-9 else "BRAKE"
        elif mode == "BRAKE" and speed > 1e-9:
            pass
        elif mode in {"WAIT", "BRAKE"}:
            clear_since = time
            mode = "CLEAR_CONFIRM"
        elif mode == "CLEAR_CONFIRM" and time - clear_since >= CLEAR_CONFIRM_S:
            mode = "RESUME"
        elif mode == "RESUME" and speed >= max_v - 1e-6:
            mode = "MOW"
            last_obstacle = ""
        blocked[i] = occupied.any()
        in_detour = detour is not None and detour[0] <= center[0] <= detour[1]
        remaining = s if returning else arc[-1] - s
        at_end = remaining < .10 or time >= w.duration_s - 1.05
        if mode == "MOW" and turning:
            modes[i] = "TURN"
        else:
            modes[i] = "REPLAN" if mode == "MOW" and in_detour else mode
        names[i] = "toy" if modes[i] == "REPLAN" else last_obstacle
        command[i] = (mode in {"MOW", "RESUME"} and not in_detour and not turning
                      and time >= 1.0 and not at_end)
        if i + 1 == len(times):
            break
        rotor_rate = 1 / SPIN_UP_S if command[i] else -1 / SPIN_DOWN_S
        rotor[i + 1] = np.clip(rotor[i] + rotor_rate * dt, 0, 1)
        can_move = not turning and (
            mode == "MOW" or (mode == "RESUME" and rotor[i] >= CUTTING_ROTOR_FRACTION))
        stop_by_end = np.sqrt(2 * m.max_accel_mps2 * max(0, remaining - .004))
        end_fade = max(0., (w.duration_s - .05 - time) * m.max_accel_mps2)
        goal = min(max_v, stop_by_end, end_fade) if can_move and time >= 1.35 else 0.
        velocity[i + 1] = speed + np.clip(goal - speed, -m.max_accel_mps2 * dt, m.max_accel_mps2 * dt)
        step = path_direction[i] * (speed + velocity[i + 1]) * (.5 * dt)
        distance[i + 1] = np.clip(s + step, 0., arc[-1])
        position[i + 1] = at(distance[i + 1])

    # The relative segment sweep conservatively covers moving animals between all ticks.
    relative = position[:, None, :] - actor_xy
    sweeps = distance_to_segment(np.zeros(2), relative[:-1], relative[1:]) - radii
    min_clearance = np.r_[np.min(sweeps, axis=1),
                          np.min(np.linalg.norm(relative[-1], axis=1) - radii)]
    if min_clearance.min() < .025:
        interval, actor = np.unravel_index(np.argmin(sweeps), sweeps.shape)
        raise ValueError(f"Backyard layout conflict — swept mower approaches {ACTOR_NAMES[actor]} "
                         f"within {sweeps[interval, actor]:.3f} m at {times[interval]:.3f} s")
    yaw = np.interp(distance, arc, yaw_path) - turn_angle
    omega = path_direction * velocity * np.interp(distance, arc, curvature) - turn_rate
    blade = rotor >= CUTTING_ROTOR_FRACTION
    trajectory = trajectory_from_path(config, terrain, times, position, yaw, velocity, omega, blade, rotor)
    story = StoryState(tracks, modes, blocked, names, min_clearance, command, rotor)
    return trajectory, story
