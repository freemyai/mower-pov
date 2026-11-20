"""Absolute-time blocking for the garden animals and props."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActorPose:
    xy: tuple[float, float]
    heading: float
    distance: float
    moving: float


def _travel(time, start_s, end_s, start, end):
    start_xy = np.asarray(start)
    delta = np.asarray(end) - start_xy
    u = float(np.clip((time - start_s) / (end_s - start_s), 0., 1.))
    progress = u ** 2 * (3. - 2. * u)
    return ActorPose(tuple(start_xy + progress * delta), float(np.arctan2(delta[1], delta[0])),
                     float(progress * np.linalg.norm(delta)), float(4 * u * (1 - u)))


def hedgehog_pose(time):
    return _travel(time, .9, 4.25, (1.28, -.71), (.99, -1.45))


def ball_pose(time):
    return _travel(time, 17.25, 22.25, (4.46, 2.91), (.40, 2.91))


def dog_pose(time):
    return _travel(time, 17.6, 22.8, (4.96, 3.14), (.90, 3.14))


TRAMPOLINE_XY = (3.5, 4.23)
TRAMPOLINE_RADIUS = .73
FENCE_Y = 2.18
FENCE_ENDPOINTS = (.47, 4.96)
TOY_XY = (1.74, 2.44)
TOY_RADIUS = .27
HEDGE_XY = (1.0, -1.49)
BALL_RADIUS = .096
HEDGEHOG_RADIUS = .235
DOG_RADIUS = .65


def trot_step(phase):
    p = np.asarray(phase) % 1.
    stance = .62
    recover = np.clip((p - stance) / (1. - stance), 0., 1.)
    step = .124
    x = np.where(p < stance, step * (1. - 2. * p / stance),
                 step * (-1. + 2. * recover * recover * (3 - 2 * recover)))
    z = .090 * np.sin(np.pi * recover) ** 2
    if p.ndim == 0:
        return float(x), float(z)
    return x, z


def _distance_to_segment(points, start_xy, end_xy):
    start, end = np.asarray(start_xy), np.asarray(end_xy)
    relative = points - start
    delta = end - start
    u = np.clip((relative @ delta) / (delta @ delta), 0., 1.)
    return np.linalg.norm(relative - u[:, None] * delta, axis=1)


def layout_clearance(bundle):
    """Conservative planar clearance over the full route, irrespective of timing."""
    points = bundle.trajectory.position_m[:, :2]
    mower = bundle.resolved_config.mower
    # A pair of adjacent 240 Hz poses can move by at most one integration step.
    footprint = (np.hypot(mower.length_m, mower.width_m) / 2
                 + mower.max_speed_mps / bundle.resolved_config.world.step_hz)
    values = {}
    for name, track, radius in (("hedgehog", hedgehog_pose, HEDGEHOG_RADIUS),
                                ("dog", dog_pose, DOG_RADIUS), ("ball", ball_pose, BALL_RADIUS)):
        values[name] = float(_distance_to_segment(points, track(0).xy, track(40).xy).min()
                             - footprint - radius)
    values["fence"] = float(_distance_to_segment(points, (FENCE_ENDPOINTS[0], FENCE_Y),
                                                  (FENCE_ENDPOINTS[1], FENCE_Y)).min()
                            - footprint - .075)
    for name, xy, radius in (("trampoline", TRAMPOLINE_XY, TRAMPOLINE_RADIUS + .06),
                             ("toys", TOY_XY, TOY_RADIUS), ("hedge", HEDGE_XY, .46)):
        values[name] = float(np.linalg.norm(points - xy, axis=1).min() - footprint - radius)
    return values


def layout_available(bundle):
    """Check that the fixed blocking fits the yard and its configured flower bed."""
    from ..motion import segments_hit_box

    world = bundle.resolved_config.world
    if world.width_m < 11.15 or world.depth_m < 9.95:
        return False
    if min(layout_clearance(bundle).values()) < .09:
        return False

    bed = np.asarray(bundle.terrain.obstacle_xyxy_m)
    bounds = [(track(0).xy, track(40).xy, radius)
              for track, radius in ((hedgehog_pose, HEDGEHOG_RADIUS), (dog_pose, DOG_RADIUS),
                                    (ball_pose, BALL_RADIUS))]
    bounds.append(((FENCE_ENDPOINTS[0], FENCE_Y), (FENCE_ENDPOINTS[1], FENCE_Y), .075))
    bounds.extend((xy, xy, radius) for xy, radius in (
        (TRAMPOLINE_XY, TRAMPOLINE_RADIUS + .06), (TOY_XY, TOY_RADIUS), (HEDGE_XY, .46),
    ))
    return all(not segments_hit_box(np.asarray([a]), np.asarray([b]), bed[:2] - r, bed[2:] + r)[0]
               for a, b, r in bounds)
