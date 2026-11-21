"""Grounded, procedural obstacles for the mower's garden course."""

import numpy as np

from ..terrain import sample_height
from .blender_life import _empty, _ring_curve
from .blender_scene import material, mesh_object, tube


def _finish(bpy, name, rgb, roughness, noise, metallic=0., tint=.16):
    mat = material(bpy, name, rgb, roughness, metallic, surface_noise=noise)
    ramp = next(node for node in mat.node_tree.nodes if node.type == "VALTORGB")
    for element, factor in zip(ramp.color_ramp.elements, (1 - tint, 1 + tint)):
        element.color = (*(value * factor for value in rgb), 1.)
    return mat


def _hose_fitting(bpy, root, end, inward, grip, gasket, metal, opening):
    # The hardware sleeves the end of the hose inside its collision footprint.
    axis = inward - end
    axis /= np.linalg.norm(axis)
    for name, start, finish, front, back, mat, sides in (
        ("Metal quick-fit nose", .001, .025, .0135, .0135, metal, 32),
        ("Molded hexagonal collar", .023, .034, .021, .021, grip, 12),
        ("Elastomer connector grip", .032, .060, .0184, .0184, gasket, 24),
        ("Orange locking sleeve", .059, .069, .0205, .0205, grip, 24),
        ("Tapered strain relief", .068, .085, .0175, .0137, gasket, 24),
    ):
        tube(bpy, name, end + start * axis, end + finish * axis, front, back, mat, sides, root)
    # A shallow cap at the end resolves as a bore from a low, oblique viewpoint.
    tube(bpy, "Recessed connector opening", end, end + .0015 * axis,
         .0082, .0082, opening, 24, root)
    for offset in (.035, .043, .051):
        tube(bpy, "Connector grip rib", end + offset * axis, end + (offset + .0018) * axis,
             .0192, .0192, gasket, 24, root)


def _hose(bpy, terrain, obstacle):
    root = _empty(bpy, "Garden course hose")
    sheath = _finish(bpy, "Ochre reinforced hose rubber", (.56, .285, .053), .44,
                     (146, .19, .00024))
    tracer = material(bpy, "Pale hose tracer stripe", (.74, .58, .235), .49)
    grip = material(bpy, "Orange connector nylon", (.71, .19, .032), .41,
                    surface_noise=(305, .12, .00021))
    gasket = material(bpy, "Dark ribbed connector elastomer", (.031, .039, .033), .59)
    metal = material(bpy, "Weathered brass connector", (.34, .208, .085), .36, .77,
                     surface_noise=(235, .25, .00031))
    opening = material(bpy, "Shadow in hose fittings", (.009, .010, .008), .85)

    points = np.asarray(obstacle.points)
    span_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    path = np.concatenate([np.linspace(a, b, max(1, int(np.ceil(length / .012))), endpoint=False)
                           for a, b, length in zip(points[:-1], points[1:], span_lengths)]
                          + [points[-1:]])
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    end_distance = np.minimum(distance, distance[-1] - distance)
    section_radius = .0136
    height = sample_height(terrain, path) + section_radius + .0065 * np.exp(-end_distance / .038)
    center = np.column_stack((path, height))
    main = _ring_curve(bpy, "Hose rubber sheath on the ground", [center], section_radius, sheath, root)
    main.data.bevel_resolution = 4
    main.data.use_fill_caps = True
    stripe = center.copy()
    stripe[:, 2] += section_radius * .988
    _ring_curve(bpy, "Longitudinal hose tracer", [stripe], .00065, tracer, root)

    next_at_start = np.searchsorted(distance, .055)
    next_at_end = np.searchsorted(distance, distance[-1] - .055)
    _hose_fitting(bpy, root, center[0], center[next_at_start], grip, gasket, metal, opening)
    _hose_fitting(bpy, root, center[-1], center[next_at_end], grip, gasket, metal, opening)
    return root


def _can_body(bpy, root, paint, inner):
    # The profile runs from the sole over the rolled rim and down into the open neck.
    profile = (
        (.001, .084), (.003, .101), (.013, .108), (.041, .113),
        (.100, .111), (.147, .104), (.171, .092), (.189, .069),
        (.206, .046), (.218, .045), (.220, .051), (.225, .052),
        (.229, .048), (.229, .041), (.224, .037), (.212, .037),
        (.194, .042), (.170, .077), (.148, .093), (.040, .101), (.014, .090),
    )
    sides = 56
    theta = np.linspace(0., 2 * np.pi, sides, endpoint=False)
    vertices = [(radius * np.cos(t) - .050, radius * .79 * np.sin(t), z)
                for z, radius in profile for t in theta]
    faces = [(i * sides + j, i * sides + (j + 1) % sides,
              (i + 1) * sides + (j + 1) % sides, (i + 1) * sides + j)
             for i in range(len(profile) - 1) for j in range(sides)]
    faces += [tuple(range(sides - 1, -1, -1)), tuple(range(len(vertices) - sides, len(vertices)))]
    body = mesh_object(bpy, "Oval enamel watering can with open neck", vertices, faces, paint)
    body.parent = root
    body.data.materials.append(inner)
    for face in body.data.polygons[:-2]:
        face.use_smooth = True
        face.material_index = int(face.index >= 14 * sides)
    body.data.polygons[-1].material_index = 1


def _can_handle(bpy, root, mat):
    # A large carrying loop and the more compact handhold used when pouring.
    def handle_arc(start, control_a, control_b, end):
        a, b, c, d = np.asarray((start, control_a, control_b, end))
        t = np.linspace(0., 1., 33)[:, None]
        u = 1 - t
        xz = u**3 * a + 3 * u**2 * t * b + 3 * u * t**2 * c + t**3 * d
        return np.column_stack((xz[:, 0], np.zeros(len(xz)), xz[:, 1]))

    top = (-.068, .327)
    front = handle_arc((-.002, .217), (.047, .257), (.040, .327), top)
    rear = handle_arc(top, (-.188, .327), (-.228, .246), (-.143, .146))
    carry = np.concatenate((front[:-1], rear))
    pour = handle_arc((-.126, .180), (-.232, .204), (-.244, .071), (-.153, .064))
    _ring_curve(bpy, "Rolled tubular carry handle", [carry], .0102, mat, root)
    _ring_curve(bpy, "Rear pouring grip", [pour], .0092, mat, root)


def _can_rose(bpy, root, stem, end, trim, holes):
    axis = end - stem
    axis /= np.linalg.norm(axis)
    tube(bpy, "Sprinkler rose rim", end - .021 * axis, end - .004 * axis,
         .016, .039, trim, 48, root)
    tube(bpy, "Rounded rim lip", end - .004 * axis, end,
         .039, .037, trim, 48, root)
    across = np.array([0., 1., 0.])
    up = np.cross(axis, across)
    # Dark inset circles share the rose's face plane and give it a perforated read.
    holes_xy = [(0., 0.)]
    for count, radius in ((6, .012), (12, .026)):
        holes_xy.extend((radius * np.cos(t), radius * np.sin(t))
                        for t in np.linspace(0., 2 * np.pi, count, endpoint=False))
    angles = np.linspace(0., 2 * np.pi, 10, endpoint=False)
    ring = .0018 * (np.cos(angles)[:, None] * across + np.sin(angles)[:, None] * up)
    vertices = np.concatenate([end + .00013 * axis + x * across + y * up + ring
                               for x, y in holes_xy])
    faces = [tuple(range(i, i + len(ring))) for i in range(0, len(vertices), len(ring))]
    face = mesh_object(bpy, "Watering rose perforations", vertices.tolist(), faces, holes)
    face.parent = root


def _watering_can(bpy, terrain, obstacle):
    xy = obstacle.points[0]
    ground = float(sample_height(terrain, xy))
    root = _empty(bpy, "Garden course watering can", position=(*xy, ground - .002))
    root.rotation_euler.z = -.92
    blue = _finish(bpy, "Blue satin enamel", (.039, .179, .315), .37,
                   (177, .14, .00029), .19)
    inside = material(bpy, "Dark watermarked enamel", (.023, .085, .129), .61, .30,
                      surface_noise=(73, .29, .00069))
    metal = material(bpy, "Weathered rose nickel", (.29, .315, .305), .41, .78,
                     surface_noise=(230, .27, .00038))
    holes = material(bpy, "Shadowed watering holes", (.009, .016, .017), .79)
    _can_body(bpy, root, blue, inside)
    _can_handle(bpy, root, blue)
    stem = np.array([.053, 0., .100])
    tip = np.array([.211, 0., .261])
    axis = tip - stem
    axis /= np.linalg.norm(axis)
    tube(bpy, "Welded base of the spout", stem - .013 * axis, stem + .020 * axis,
         .038, .032, blue, 48, root)
    tube(bpy, "Tapering watering spout", stem + .011 * axis, tip - .016 * axis,
         .032, .013, blue, 48, root)
    _can_rose(bpy, root, stem, tip, metal, holes)
    return root


def _garden_ball(bpy, terrain, obstacle):
    xy = obstacle.points[0]
    radius = obstacle.radius_m - .004
    ground = float(sample_height(terrain, xy))
    root = _empty(bpy, "Garden course ball", position=(*xy, ground + radius - .004))
    root.rotation_euler = (.71, -.35, -.38)
    panels = (
        ("Old cream ball panel", (.67, .551, .32)),
        ("Golden ball panel", (.68, .314, .041)),
        ("Warm white ball panel", (.72, .61, .414)),
        ("Faded orange ball panel", (.64, .229, .079)),
    )
    mats = [_finish(bpy, label, rgb, .65, (248, .29, .00035), tint=.19)
            for label, rgb in panels]
    seam = material(bpy, "Worn recessed ball seams", (.283, .176, .072), .76)
    panel_count, facets, rows = 8, 10, 36
    width = panel_count * facets
    pitch = 2 * np.pi / panel_count
    panel_angles = np.r_[0., np.linspace(.018, pitch, facets)[:-1]]
    azimuth = (pitch * np.arange(panel_count)[:, None] + panel_angles).ravel()
    section = radius - .0013 * (np.arange(width) % facets == 0)
    vertices = []
    for theta in np.linspace(0., np.pi, rows + 1)[1:-1]:
        angle = azimuth + .41 * np.sin(theta + .46)
        vertices.extend(np.column_stack((section * np.sin(theta) * np.cos(angle),
                                         section * np.sin(theta) * np.sin(angle),
                                         section * np.cos(theta))))
    north, south = len(vertices), len(vertices) + 1
    vertices.extend([(0., 0., radius), (0., 0., -radius)])
    faces = [(i * width + j, (i + 1) * width + j, (i + 1) * width + (j + 1) % width,
              i * width + (j + 1) % width) for i in range(rows - 2) for j in range(width)]
    last = (rows - 2) * width
    faces += [(north, j, (j + 1) % width) for j in range(width)]
    faces += [(south, last + (j + 1) % width, last + j) for j in range(width)]
    ball = mesh_object(bpy, "Weathered rubber playground ball", vertices, faces, mats[0])
    ball.parent = root
    for mat in (*mats[1:], seam):
        ball.data.materials.append(mat)
    for face in ball.data.polygons:
        j = face.index % width
        face.material_index = len(mats) if j % facets == 0 else (j // facets) % len(mats)
        face.use_smooth = True
    return root


def make_course_props(bpy, bundle):
    """Use the same obstacle footprints as the controller and terrain."""
    from ..course_layout import COURSE_OBSTACLES

    builders = {"garden_hose": _hose, "watering_can": _watering_can, "garden_ball": _garden_ball}
    return tuple(builders[obstacle.name](bpy, bundle.terrain, obstacle) for obstacle in COURSE_OBSTACLES)
