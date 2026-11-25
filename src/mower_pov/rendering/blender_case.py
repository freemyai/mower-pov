"""Render exactly the flat obstacle proxies evaluated in a static case."""

import numpy as np

from ..terrain import sample_height
from .blender_scene import material, mesh_object, tube


def make_static_obstacles(bpy, bundle):
    paint = material(bpy, "Case obstacle", (.31, .102, .03), .72)
    soil = material(bpy, "Flower bed", (.13, .068, .033), .87)
    for item in bundle.manifest["identity"]["static_obstacles"]:
        points = np.asarray(item["points_xy_m"])
        radius = item["radius_m"]
        if len(points) == 1:
            height = min(.24, max(.035, radius))
            ground = float(sample_height(bundle.terrain, points[0]))
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=height,
                                                location=(*points[0], ground + height / 2))
            bpy.context.object.name = item["id"]
            bpy.context.object.data.materials.append(paint)
            continue
        caps = np.column_stack((points, sample_height(bundle.terrain, points) + radius))
        for a, b in zip(caps, caps[1:]):
            tube(bpy, item["id"], a, b, radius, radius, paint, sides=12)
        for p in caps:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=radius, location=p)
            bpy.context.object.name = item["id"]
            bpy.context.object.data.materials.append(paint)

    world = bundle.resolved_config.world
    x0, y0, x1, y1 = world.flower_bed_xyxy_m
    if x0 < world.width_m/2 and y0 < world.depth_m/2:
        xy = np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
        vertices = np.column_stack((xy, sample_height(bundle.terrain, xy) + .02))
        mesh_object(bpy, "Flower bed", vertices, [(0, 1, 2, 3)], soil)
