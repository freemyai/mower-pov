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
ACTOR_NAMES = ("dog", "hedgehog", "ball")
LEGACY_TRACKS = (DOG_TRACK, HEDGEHOG_TRACK, BALL_TRACK)


def story_tracks(scenario):
    if scenario == "toy-detour":
        tracks = LEGACY_TRACKS
    else:
        dog = Track(8.2, 13.6, (2.15, 4.43), (2.15, 1.25))
        # The hedgehog occupies the lane until its own movement clears it.
        hedgehog = Track(100., 104., (3.92, 2.86), (3.92, 1.70)) if scenario == "hedgehog-stays" else (
            Track(25.1, 28.7, (3.92, 2.86), (3.92, 1.70)))
        ball = Track(1.1, 3.55, (.12, 3.79), (.49, 3.79))
        tracks = (dog, hedgehog, ball)
    return np.array([[track.start_s, track.end_s, *track.start_xy, *track.end_xy] for track in tracks])


def tracks_from_rows(rows):
    return tuple(Track(float(start), float(end), (float(x0), float(y0)), (float(x1), float(y1)))
                 for start, end, x0, y0, x1, y1 in rows)


def story_track(bundle, name):
    index = ACTOR_NAMES.index(name)
    if bundle.story is None:
        return LEGACY_TRACKS[index]
    start, end, x0, y0, x1, y1 = map(float, bundle.story.tracks[index])
    return Track(start, end, (x0, y0), (x1, y1))


def actor_pose(bundle, name, time_s):
    return story_track(bundle, name).evaluate(time_s)


def hedgehog_pose(time_s):
    return HEDGEHOG_TRACK.evaluate(time_s)


def ball_pose(time_s):
    return BALL_TRACK.evaluate(time_s)


def dog_pose(time_s):
    return DOG_TRACK.evaluate(time_s)
