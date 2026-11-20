"""Timing and camera blocking for the separate garden montage."""

from dataclasses import dataclass

import numpy as np

from .life_motion import TRAMPOLINE_XY, dog_pose, hedgehog_pose


@dataclass(frozen=True)
class ShowcaseShot:
    name: str
    first_frame: int
    end_frame: int


SHOTS = (
    ShowcaseShot("hedgehog", 0, 120),
    ShowcaseShot("dog", 525, 675),
    ShowcaseShot("trampoline", 780, 930),
    ShowcaseShot("cut-strip", 975, 1095),
)


def jump(time):
    """Height, knee compression and arm balance for five trampoline hops."""
    phase = (time - 26.35) / .84
    if not 0 <= phase < 5:
        return 0., 0., 0.
    step = int(phase)
    state = phase - step
    u = min(1., state / .68)
    strength = (0.80, 1.0, .93, .98, .77)[step]
    height = .242 * strength * 4 * u * (1 - u)
    crouch = .099 * np.sin(np.pi * np.clip((state - .68) / .32, 0, 1)) ** 2
    balance = float(np.sin(np.pi * u))
    return height, crouch, balance


def look(time):
    """Camera origin, aim, horizontal field of view and aperture."""
    if time < 5:
        x, y = hedgehog_pose(time).xy
        return ((x - .71 - .03 * time, y - .50, .29), (x - .035, y, .126), 43., 6.3)
    if time < 24:
        x, y = dog_pose(time).xy
        return ((x - 1.05, y - 1.92, .69), (x - .29, y - .07, .305), 53., 8.)
    if time < 32:
        x, y = TRAMPOLINE_XY
        u = np.clip((time - 26) / 5, 0, 1)
        return ((x - 2.35 + .17 * u, y - 2.05, 1.16), (x, y, 1.06), 51., 8.)
    u = np.clip((time - 32.5) / 4, 0, 1)
    return ((1.48 - .43*u, -.52, .32), (-.23, .06, .115), 58., 9.5)
