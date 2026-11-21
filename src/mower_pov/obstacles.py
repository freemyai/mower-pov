"""Shared yard geometry and conservative swept-footprint checks."""

import numpy as np

from .motion import segments_hit_box
from .actors.behavior import ACTOR_NAMES, tracks_from_rows

TRAMPOLINE_XY = (3.5, 4.12)
TRAMPOLINE_RADIUS = .73
TRAMPOLINE_OUTER_RADIUS = TRAMPOLINE_RADIUS + .06
FENCE_Y = 2.18
FENCE_ENDPOINTS = (.47, 4.96)
OPEN_RAILS = ((.47, 1.53), (2.77, 3.51), (4.33, 4.96))
TOY_XY = (1.74, 2.44)
STORY_TOY_XY = (1.19, 3.75)
DETOUR_TOY_XY = (.90, .60)
TOY_RADIUS = .27
HEDGE_XY = (1., -1.49)
STORY_HEDGE_XY = (3.92, 1.65)
ACTOR_RADII = (.46, .225, .096)


def play_rails(scenario):
    return OPEN_RAILS if scenario in {"backyard-story", "hedgehog-stays"} else (FENCE_ENDPOINTS,)


def toy_center(scenario):
    if scenario == "toy-detour":
        return DETOUR_TOY_XY
    if scenario in {"backyard-story", "hedgehog-stays"}:
        return STORY_TOY_XY
    return TOY_XY


def trampoline_uprights():
    angles = np.arange(6) * (np.pi / 3)
    return (TRAMPOLINE_RADIUS - .041) * np.column_stack((np.cos(angles), np.sin(angles)))


def distance_to_segment(points, start, end):
    vector = np.asarray(end) - start
    difference = np.asarray(points) - start
    length_sq = np.sum(vector * vector, axis=-1)
    fraction = np.sum(difference * vector, axis=-1) / np.maximum(length_sq, 1e-15)
    return np.linalg.norm(difference - np.clip(fraction, 0., 1.)[..., None] * vector, axis=-1)


def _check_route(points, config, terrain, radius):
    width, depth = config.world.width_m, config.world.depth_m
    edge = np.array([width, depth]) / 2
    if np.any(np.abs(np.asarray(TRAMPOLINE_XY)) + TRAMPOLINE_OUTER_RADIUS >= edge - .0605):
        raise ValueError("Backyard layout conflict — trampoline overlaps the perimeter")
    if np.any(np.abs(points) + radius >= edge):
        raise ValueError("Backyard layout conflict — mower swept volume reaches the perimeter")
    bed = np.asarray(terrain.obstacle_xyxy_m)
    if segments_hit_box(points[:-1], points[1:], bed[:2] - radius, bed[2:] + radius).any():
        raise ValueError("Backyard layout conflict — mower swept volume reaches the flower bed")
    for a, b in trampoline_uprights().reshape(3, 2, 2) * 1.050 + TRAMPOLINE_XY:
        if distance_to_segment(points, a, b).min() <= radius + .026:
            raise ValueError("Backyard layout conflict — mower intersects a trampoline lower bar")
    clearance = np.linalg.norm(points - TRAMPOLINE_XY, axis=1).min()
    if clearance <= radius + TRAMPOLINE_OUTER_RADIUS:
        raise ValueError("Backyard layout conflict — mower intersects the trampoline")
    for a, b in play_rails(config.scenario):
        if distance_to_segment(points, (a, FENCE_Y), (b, FENCE_Y)).min() <= radius + .075:
            raise ValueError("Backyard layout conflict — mower intersects a play-area rail")
    if np.linalg.norm(points - toy_center(config.scenario), axis=1).min() <= radius + TOY_RADIUS:
        raise ValueError("Backyard layout conflict — mower intersects the toy footprint")


def validate_story_layout(config, terrain, points):
    """Conservative footprint with padding for spacing between the route's samples."""
    body_radius = .5 * np.hypot(config.mower.length_m, config.mower.width_m)
    sampling_pad = .5 * np.linalg.norm(np.diff(points, axis=0), axis=1).max(initial=0.)
    _check_route(points, config, terrain, body_radius + sampling_pad + .025)


def validate_actor_paths(config, terrain, tracks):
    if config.scenario == "toy-detour":
        return
    limit = np.array([config.world.width_m, config.world.depth_m]) / 2 - .06
    bed = np.asarray(terrain.obstacle_xyxy_m)
    for name, track, radius in zip(ACTOR_NAMES, tracks_from_rows(tracks), ACTOR_RADII):
        path = np.linspace(track.start_xy, track.end_xy, 301)
        prefix = f"Backyard layout conflict — {name} track intersects "
        if (np.abs(path) + radius >= limit).any():
            raise ValueError(prefix + "yard perimeter")
        if segments_hit_box(path[:-1], path[1:], bed[:2] - radius, bed[2:] + radius).any():
            raise ValueError(prefix + "flower bed")
        if np.linalg.norm(path - TRAMPOLINE_XY, axis=1).min() < radius + TRAMPOLINE_OUTER_RADIUS:
            raise ValueError(prefix + "trampoline")
        if np.linalg.norm(path - toy_center(config.scenario), axis=1).min() < radius + TOY_RADIUS:
            raise ValueError(prefix + "toys")
        for a, b in play_rails(config.scenario):
            if distance_to_segment(path, (a, FENCE_Y), (b, FENCE_Y)).min() < radius + .075:
                raise ValueError(prefix + "play-area rail")
