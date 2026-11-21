"""Renderer-independent absolute-time actor blocking."""

from dataclasses import dataclass
from math import atan2, hypot

import numpy as np


@dataclass(frozen=True)
class ActorPose:
    xy: tuple[float, float]
    heading: float
    distance: float
    moving: float


@dataclass(frozen=True)
class Track:
    start_s: float
    end_s: float
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]

    @property
    def length(self):
        return hypot(self.end_xy[0] - self.start_xy[0], self.end_xy[1] - self.start_xy[1])

    @property
    def direction(self):
        return (np.asarray(self.end_xy) - self.start_xy) / self.length

    def evaluate(self, time_s):
        delta = np.asarray(self.end_xy) - self.start_xy
        u = float(np.clip((time_s - self.start_s) / (self.end_s - self.start_s), 0., 1.))
        progress = u ** 2 * (3. - 2. * u)
        return ActorPose(tuple(np.asarray(self.start_xy) + progress * delta),
                         atan2(delta[1], delta[0]), progress * self.length, 4 * u * (1 - u))


HEDGEHOG_TRACK = Track(.9, 4.25, (1.28, -.71), (.99, -1.45))
BALL_TRACK = Track(17.25, 22.25, (4.46, 2.91), (.40, 2.91))
DOG_TRACK = Track(17.6, 22.8, (4.96, 3.14), (.90, 3.14))


def hedgehog_pose(time_s):
    return HEDGEHOG_TRACK.evaluate(time_s)


def ball_pose(time_s):
    return BALL_TRACK.evaluate(time_s)


def dog_pose(time_s):
    return DOG_TRACK.evaluate(time_s)
