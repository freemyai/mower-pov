"""Continuous world-space breeze shared by grass and garden foliage."""

from dataclasses import dataclass

import numpy as np


def breeze(time_s, x, y):
    """Angular displacement in the XY plane, with broad gusts and smooth ripples."""
    along = .86 * x + .51 * y
    packet = .5 + .5 * np.sin(2.05 * time_s - 1.65 * along)
    ripple = np.sin(4.1 * time_s - 3.3 * along + .56 * y)
    dx = .025 + .127 * packet ** 2 + .018 * ripple
    dy = .05 * np.sin(1.68 * time_s - 1.4 * along) + .028 * packet
    return np.stack((dx, dy), axis=-1)


def leaf_tilt(world_xy, yaw):
    dx, dy = np.moveaxis(world_xy, -1, 0)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack((s * dx - c * dy, c * dx + s * dy), axis=-1)


def foliage_positions(reference, flexibility, time_s):
    """Replay displacement from anchored positions, with no frame-order dependence."""
    xy = breeze(time_s, reference[:, 0], reference[:, 1])
    positions = reference.copy()
    positions[:, :2] += flexibility[:, None] * xy
    positions[:, 2] -= .36 * flexibility * np.sum(xy ** 2, axis=1)
    return positions


@dataclass
class FoliageReplay:
    mesh: object
    reference: np.ndarray
    flexibility: np.ndarray

    def apply(self, time_s):
        coords = foliage_positions(self.reference, self.flexibility, time_s)
        self.mesh.vertices.foreach_set("co", coords.astype("f").ravel())
        self.mesh.update()
