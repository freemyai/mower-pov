"""Shared metre-scale geometry for obstacles and the mowing lanes."""

from dataclasses import dataclass

import numpy as np


COURSE_X_LIMITS = (-1.65, 1.42)
COURSE_ROW_Y = (1.30, 1.05, .80, .55, .30)
BODY_DESIGN_CLEARANCE_M = .035


@dataclass(frozen=True)
class CourseObstacle:
    name: str
    points: tuple[tuple[float, float], ...]
    radius_m: float


def _hose():
    lead = ((.23, 1.33), (.36, 1.365), (.45, 1.42), (.51, 1.50),
            (.54, 1.60), (.54, 1.71), (.56, 1.81))
    angle = np.linspace(-2.75, -2.75 + 2.38 * np.pi, 61)
    radius = np.linspace(.220, .157, len(angle))
    coil = tuple(zip(.77 + radius * np.cos(angle), 1.91 + radius * np.sin(angle)))
    return lead + coil


COURSE_OBSTACLES = (
    CourseObstacle("garden_hose", _hose(), .031),
    CourseObstacle("watering_can", ((-.34, -.09),), .24),
    CourseObstacle("garden_ball", ((-.80, .92),), .115),
)


def course_layout_identity():
    return {
        "x_limits_m": list(COURSE_X_LIMITS),
        "row_y_m": list(COURSE_ROW_Y),
        "obstacles": [{"name": item.name, "points_m": [list(p) for p in item.points],
                       "radius_m": item.radius_m} for item in COURSE_OBSTACLES],
    }


def _pose_box_distance(centers, yaw, half, segments):
    """Exact point/segment distance to each oriented two-dimensional chassis box."""
    delta = np.asarray(segments)[None, :, :, :] - centers[:, None, None, :]
    c, s = np.cos(yaw)[:, None, None], np.sin(yaw)[:, None, None]
    local = np.stack((delta[..., 0] * c + delta[..., 1] * s,
                      -delta[..., 0] * s + delta[..., 1] * c), axis=-1)
    closest = np.linalg.norm(np.maximum(np.abs(local) - half, 0.), axis=-1).min(axis=2)
    a, b = local[:, :, 0], local[:, :, 1]
    edge = b - a
    if local.shape[1] == 1 and np.all(edge == 0):
        return closest[:, 0]

    corners = half * np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    direction = edge[:, :, None, :]
    origin_to_corner = corners - a[:, :, None, :]
    length_sq = (direction * direction).sum(axis=-1)
    along = (origin_to_corner * direction).sum(axis=-1) / np.maximum(length_sq, 1e-20)
    nearest = np.linalg.norm(origin_to_corner - np.clip(along, 0, 1)[..., None] * direction, axis=-1)
    closest = np.minimum(closest, nearest.min(axis=2))

    parallel = np.abs(edge) < 1e-14
    divide = np.where(parallel, 1., edge)
    t0, t1 = (-half - a) / divide, (half - a) / divide
    low = np.where(parallel, -np.inf, np.minimum(t0, t1)).max(axis=-1)
    high = np.where(parallel, np.inf, np.maximum(t0, t1)).min(axis=-1)
    parallel_miss = np.any(parallel & (np.abs(a) > half), axis=-1)
    through = ~parallel_miss & (np.maximum(low, 0.) <= np.minimum(high, 1.))
    return np.where(through, 0., closest).min(axis=1)


def obstacle_clearance(polyline_xy, yaw, config, obstacle):
    """Return signed metres from each chassis rectangle to an obstacle proxy."""
    centers = np.asarray(polyline_xy)
    angles = np.broadcast_to(np.asarray(yaw), (len(centers),))
    m = config.mower
    half = np.array([m.length_m, m.width_m]) * .5
    points = np.asarray(obstacle.points)
    segments = (np.stack((points[:-1], points[1:]), axis=1) if len(points) > 1 else
                np.stack((points, points), axis=1))
    return np.concatenate([_pose_box_distance(centers[i:i+2048], angles[i:i+2048], half, segments)
                           for i in range(0, len(centers), 2048)]) - obstacle.radius_m


def obstacle_swept_clearance(xy, yaw, config, obstacle):
    # The farthest box point moves no more than centre translation plus corner arc.
    # Taking half the interval and the worst endpoint bounds the complete sweep.
    radius = .5 * np.hypot(config.mower.length_m, config.mower.width_m)
    interval_pad = .5 * (np.linalg.norm(np.diff(xy, axis=0), axis=1) + radius * abs(np.diff(yaw)))
    limit = np.zeros(len(xy))
    limit[:-1] = interval_pad
    np.maximum(limit[1:], interval_pad, out=limit[1:])
    return obstacle_clearance(xy, yaw, config, obstacle) - limit
