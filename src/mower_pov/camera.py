"""Rigid body-to-optical transform (+X right, +Y down, +Z forward)."""

import numpy as np

from .contracts import RunConfig


def camera_mount(config: RunConfig) -> np.ndarray:
    tilt = np.deg2rad(config.camera.pitch_deg)
    c, s = np.cos(tilt), np.sin(tilt)
    mount = np.eye(4)
    mount[:3, :3] = [[0, -s, c], [-1, 0, 0], [0, -c, -s]]
    mount[:3, 3] = config.camera.mount_xyz_m
    return mount


def camera_pose(config: RunConfig, body_pose: np.ndarray) -> np.ndarray:
    return body_pose @ camera_mount(config)
