"""Independent planar chassis geometry and continuous clearance bounds."""

import numpy as np

WAYPOINT_TOL_M = .015


def _local_points(points, centres, yaw):
    offset = np.asarray(points)[None, :, :] - centres[:, None, :]
    x, y = offset[..., 0], offset[..., 1]
    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return np.stack((cos * x + sin * y, cos * y - sin * x), axis=-1)


def _box_distances(points, half):
    return np.linalg.norm(np.maximum(abs(points) - half, 0.), axis=-1)


def _polyline_distance(centres, yaw, size, points):
    half = np.asarray(size) / 2
    p = _local_points(points, centres, yaw)
    nearest = _box_distances(p, half).min(axis=1)
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * half
    for a, b in zip(p[:, :-1].swapaxes(0, 1), p[:, 1:].swapaxes(0, 1)):
        v = b - a
        length_sq = np.sum(v * v, axis=1)
        if not np.any(length_sq):
            continue
        # Separating axes for a segment and a box: the two box axes and
        # the normal of the segment. Endpoint/corner distances cover disjoint shapes.
        normal = np.stack((-v[:, 1], v[:, 0]), axis=1)
        within_axes = np.all((np.minimum(a, b) <= half) & (np.maximum(a, b) >= -half), axis=1)
        on_line = abs(np.sum(normal * a, axis=1)) <= np.sum(abs(normal) * half, axis=1)
        corner_delta = corners[None, :, :] - a[:, None, :]
        projected = np.einsum("nki,ni->nk", corner_delta, v) / np.maximum(length_sq[:, None], 1e-30)
        distance = np.linalg.norm(corner_delta - np.clip(projected, 0, 1)[..., None] * v[:, None, :], axis=-1)
        nearest = np.minimum(nearest, np.where(within_axes & on_line, 0., distance.min(axis=1)))
    return nearest


def obstacle_distance(centres, yaw, size, obstacle):
    return _polyline_distance(centres, yaw, size, obstacle.points_xy_m) - obstacle.radius_m


def interval_lower_bound(samples, centres, yaw, size):
    radius = np.hypot(*size) / 2
    change = np.linalg.norm(np.diff(centres, axis=0), axis=1) + radius * abs(np.diff(yaw))
    # Each interval point lies within half the motion bound of an endpoint.
    return np.minimum(samples[:-1], samples[1:]) - change / 2


def swept_obstacle_clearance(centres, yaw, size, obstacle):
    return interval_lower_bound(obstacle_distance(centres, yaw, size, obstacle), centres, yaw, size)


def swept_yard_clearance(centres, yaw, size, world_size):
    length, width = np.asarray(size) / 2
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    extent = np.column_stack((c * length + s * width, s * length + c * width))
    samples = (np.asarray(world_size) / 2 - abs(centres) - extent).min(axis=1)
    return interval_lower_bound(samples, centres, yaw, size)


def bed_distance(centres, yaw, size, bed):
    x0, y0, x1, y1 = bed
    perimeter = ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))
    distance = _polyline_distance(centres, yaw, size, perimeter)
    inside = np.all((centres >= (x0, y0)) & (centres <= (x1, y1)), axis=1)
    return np.where(inside, 0., distance)


def mission_impossibility(case):
    """Sufficient certificates only. Search failure alone proves no impossibility."""
    world, robot = case.world, case.robot
    x0, x1 = world.x_limits_m
    if len(world.row_y_m) * max(0., x1 - x0 - 2 * WAYPOINT_TOL_M) > case.duration_s * robot.max_speed_mps:
        return "mandatory row displacement exceeds the time budget at maximum speed"
    size = robot.length_m, robot.width_m
    inside = min(size) / 2
    required = np.array([(x, y) for y in world.row_y_m for x in (x0, x1)])
    if np.any(abs(required) + inside - WAYPOINT_TOL_M > np.array([world.width_m, world.depth_m]) / 2):
        return "mandatory lane endpoint cannot fit inside the yard"
    # Any orientation contains the inscribed disc. Allow the entire accepted
    # waypoint neighbourhood when proving an endpoint cannot be visited.
    yaw = np.zeros(len(required))
    forbidden = inside + world.clearance_m - WAYPOINT_TOL_M
    for item in world.obstacles:
        if np.any(obstacle_distance(required, yaw, (0., 0.), item) < forbidden):
            return f"mandatory lane endpoint conflicts with {item.id}"
    if world.flower_bed_xyxy_m is not None:
        if np.any(bed_distance(required, yaw, (0., 0.), world.flower_bed_xyxy_m) < forbidden):
            return "mandatory lane endpoint conflicts with the flower bed"

    radius = inside + world.clearance_m
    low, high = -world.depth_m / 2 + inside, world.depth_m / 2 - inside
    # A vertical slice forbidden throughout the feasible centre domain separates
    # both required lane endpoints even when the chassis changes its orientation.
    slices = sorted({x for item in world.obstacles for x, _ in item.points_xy_m
                     if x0 + WAYPOINT_TOL_M < x < x1 - WAYPOINT_TOL_M})
    for x in slices:
        spans = []
        for item in world.obstacles:
            expanded = radius + item.radius_m
            for px, py in item.points_xy_m:
                if abs(px - x) < expanded:
                    half = np.sqrt(expanded ** 2 - (px - x) ** 2)
                    spans.append((py - half, py + half))
            for (ax, ay), (bx, by) in zip(item.points_xy_m, item.points_xy_m[1:]):
                if ax == bx and abs(ax - x) < expanded:
                    half = np.sqrt(expanded ** 2 - (ax - x) ** 2)
                    spans.append((min(ay, by) - half, max(ay, by) + half))
        covered = low
        for bottom, top in sorted(spans):
            if bottom >= covered:
                break
            covered = max(covered, top)
        if covered > high:
            return f"obstacle footprints block the full centre corridor at x={x:.3f} m"
    return None
