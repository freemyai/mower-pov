"""Absolute-time blocking for the garden animals and props."""

import numpy as np

from ..actors.behavior import ball_pose, dog_pose, hedgehog_pose


TRAMPOLINE_XY = (3.5, 4.12)
TRAMPOLINE_RADIUS = .73
TRAMPOLINE_OUTER_RADIUS = TRAMPOLINE_RADIUS + .06
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
    for name, xy, radius in (("trampoline", TRAMPOLINE_XY, TRAMPOLINE_OUTER_RADIUS),
                             ("toys", TOY_XY, TOY_RADIUS), ("hedge", HEDGE_XY, .46)):
        values[name] = float(np.linalg.norm(points - xy, axis=1).min() - footprint - radius)
    return values


def layout_violations(bundle):
    """Return concrete conflicts between the authored blocking and the yard."""
    from ..motion import segments_hit_box

    issues = []
    world = bundle.resolved_config.world
    if world.width_m < 11.15 or world.depth_m < 9.95:
        issues.append(f"yard too small for the authored layout ({world.width_m:g} × {world.depth_m:g} m)")
    fence_inner_edge = np.array([world.width_m, world.depth_m]) / 2 - .0605
    if np.any(np.abs(TRAMPOLINE_XY) + TRAMPOLINE_OUTER_RADIUS >= fence_inner_edge):
        issues.append("trampoline overlaps the perimeter fence")
    for name, clearance in layout_clearance(bundle).items():
        if clearance < .09:
            issues.append(f"{name} has only {clearance:.3f} m clearance from the mower sweep")

    bed = np.asarray(bundle.terrain.obstacle_xyxy_m)
    bounds = [(name, track(0).xy, track(40).xy, radius)
              for name, track, radius in (("hedgehog", hedgehog_pose, HEDGEHOG_RADIUS),
                                          ("dog", dog_pose, DOG_RADIUS), ("ball", ball_pose, BALL_RADIUS))]
    bounds.append(("fence", (FENCE_ENDPOINTS[0], FENCE_Y), (FENCE_ENDPOINTS[1], FENCE_Y), .075))
    bounds.extend((name, xy, xy, radius) for name, xy, radius in (
        ("trampoline", TRAMPOLINE_XY, TRAMPOLINE_OUTER_RADIUS),
        ("toys", TOY_XY, TOY_RADIUS), ("hedge", HEDGE_XY, .46),
    ))
    for name, a, b, radius in bounds:
        if segments_hit_box(np.asarray([a]), np.asarray([b]), bed[:2] - radius, bed[2:] + radius)[0]:
            issues.append(f"{name} overlaps the flower bed")
    return tuple(issues)


def layout_available(bundle):
    return not layout_violations(bundle)


def require_life_layout(bundle):
    issues = layout_violations(bundle)
    if issues:
        raise ValueError("Backyard layout conflict — " + " | ".join(issues))
