"""World-locked stance contacts and fixed-length inverse kinematics for the dog proxy."""

from dataclasses import dataclass
from math import floor, sqrt

import numpy as np

from ..terrain import sample_height
from .behavior import DOG_TRACK, ActorPose

STRIDE_M = .40
STANCE_FRACTION = .62
SOLE_TO_ANKLE_M = .029


@dataclass(frozen=True)
class LegSpec:
    name: str
    x: float
    y: float
    phase: float
    upper: float
    lower: float


DOG_LEGS = (
    LegSpec("front_right", .157, -.088, 0., .182, .194),
    LegSpec("hind_left", -.169, .088, 0., .194, .191),
    LegSpec("front_left", .157, .088, .5, .182, .194),
    LegSpec("hind_right", -.169, -.088, .5, .194, .191),
)


@dataclass(frozen=True)
class LegPose:
    foot_world: tuple[float, float, float]
    planted: bool
    hip: tuple[float, float, float]
    knee: tuple[float, float, float]
    ankle: tuple[float, float, float]


@dataclass(frozen=True)
class DogPose:
    root: ActorPose
    body_z: float
    legs: dict[str, LegPose]


def _contact_distance(distance, phase, total):
    cycle = floor(distance / STRIDE_M + phase)
    touchdown = (cycle - phase) * STRIDE_M
    liftoff = touchdown + STANCE_FRACTION * STRIDE_M
    next_touchdown = touchdown + STRIDE_M
    anchor = (touchdown + liftoff) * .5
    if distance < liftoff or (total - liftoff < .070 and total < next_touchdown):
        return anchor, 0., True

    # Land early if the track ends during a swing. A late, very short swing is
    # skipped above so the departing paw never snaps to a new resting point.
    end = min(next_touchdown, total)
    u = min(1., (distance - liftoff) / (end - liftoff))
    following = next_touchdown + STANCE_FRACTION * STRIDE_M * .5 if end == next_touchdown else end
    blend = u ** 3 * (10 + u * (-15 + 6 * u))
    lift = .064 * 16 * (u * (1 - u)) ** 2 * min(1., (end - liftoff) / .13)
    return anchor + (following - anchor) * blend, lift, u >= 1.


def solve_knee(hip, ankle, spec):
    delta = ankle - hip
    length = float(np.linalg.norm(delta))
    if not abs(spec.upper - spec.lower) < length < spec.upper + spec.lower:
        raise ValueError(f"{spec.name} contact target out of reach ({length:.3f} m)")
    axis = delta / length
    pole = np.array([-1. if spec.x > 0 else 1., 0., 0.])
    pole -= axis * (pole @ axis)
    pole_length = np.linalg.norm(pole)
    if pole_length < 1e-9:
        pole = np.array([0., 1., 0.])
    else:
        pole /= pole_length
    span = (spec.upper ** 2 - spec.lower ** 2 + length ** 2) / (2 * length)
    return hip + span * axis + sqrt(spec.upper ** 2 - span ** 2) * pole


def evaluate_dog(time_s, terrain, track=DOG_TRACK):
    root = track.evaluate(time_s)
    forward = track.direction
    lateral = np.array([-forward[1], forward[0]])
    phase = 2 * np.pi * root.distance / STRIDE_M
    body_z = float(sample_height(terrain, root.xy)) + .007 * root.moving * (1 - np.cos(2 * phase))
    legs = {}
    for spec in DOG_LEGS:
        anchor, lift, planted = _contact_distance(root.distance, spec.phase, track.length)
        xy = track.start_xy + forward * (anchor + spec.x) + lateral * spec.y
        z = float(sample_height(terrain, xy)) + lift
        offset = xy - root.xy
        hip = np.array([spec.x, spec.y * .79, .341])
        ankle = np.array([offset @ forward, offset @ lateral, z + SOLE_TO_ANKLE_M - body_z])
        knee = solve_knee(hip, ankle, spec)
        legs[spec.name] = LegPose((float(xy[0]), float(xy[1]), z), planted,
                                  tuple(hip), tuple(knee), tuple(ankle))
    return DogPose(root, body_z, legs)
