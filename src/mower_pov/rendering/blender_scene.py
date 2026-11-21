"""Build the procedural backyard inside Blender, then replay the cached timeline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..cache import load_bundle
from ..terrain import sample_height
from .blender_camera import apply_vehicle_pose, mount_camera
from .blender_grass import make_grass, grass_time_s
from .blender_wind import FoliageReplay


def material(bpy, name, rgb, roughness=.72, metallic=0., surface_noise=None, grain=None, translucency=0.):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1.)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if surface_noise:
        scale, strength, distance = surface_noise
        coord = nodes.new("ShaderNodeTexCoord")
        texture = nodes.new("ShaderNodeTexNoise")
        texture.inputs["Scale"].default_value = scale
        texture.inputs["Detail"].default_value = 3.
        if grain:
            stretch = nodes.new("ShaderNodeVectorMath")
            stretch.operation = "MULTIPLY"
            stretch.inputs[1].default_value = grain
            links.new(coord.outputs["Object"], stretch.inputs[0])
            links.new(stretch.outputs["Vector"], texture.inputs["Vector"])
        else:
            links.new(coord.outputs["Object"], texture.inputs["Vector"])
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = .19
        ramp.color_ramp.elements[1].position = .84
        ramp.color_ramp.elements[0].color = (*(v * .48 for v in rgb), 1.)
        ramp.color_ramp.elements[1].color = (*(v * 1.4 for v in rgb), 1.)
        links.new(texture.outputs["Fac"], ramp.inputs[0])
        links.new(ramp.outputs[0], bsdf.inputs["Base Color"])
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = strength
        bump.inputs["Distance"].default_value = distance
        links.new(texture.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs[0], bsdf.inputs["Normal"])
    if translucency:
        translucent = nodes.new("ShaderNodeBsdfTranslucent")
        translucent.inputs["Color"].default_value = (*rgb, 1.)
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = translucency
        links.new(bsdf.outputs["BSDF"], mix.inputs[1])
        links.new(translucent.outputs["BSDF"], mix.inputs[2])
        links.new(mix.outputs[0], nodes.get("Material Output").inputs["Surface"])
    return mat


def mesh_object(bpy, name, vertices, faces, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def box(bpy, name, center, size, mat, bevel=.0, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new("Rounded manufactured edges", "BEVEL")
        mod.width = bevel
        mod.segments = 3
        obj.modifiers.new("Weighted face normals", "WEIGHTED_NORMAL")
    obj.parent = parent
    obj.location = center
    return obj


def tube(bpy, name, start, end, radius_start, radius_end, mat, sides=12, parent=None):
    from mathutils import Vector

    axis = Vector(end) - Vector(start)
    bpy.ops.mesh.primitive_cone_add(vertices=sides, radius1=radius_start,
                                    radius2=radius_end, depth=axis.length)
    obj = bpy.context.object
    obj.name = name
    obj.parent = parent
    obj.location = (Vector(start) + Vector(end)) * .5
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = axis.to_track_quat("Z", "Y")
    obj.data.materials.append(mat)
    for face in obj.data.polygons:
        face.use_smooth = len(face.vertices) == 4
    return obj


def _ground(bpy, bundle, mat):
    w = bundle.resolved_config.world
    x = np.linspace(-w.width_m / 2, w.width_m / 2, round(w.width_m / .05) + 1)
    y = np.linspace(-w.depth_m / 2, w.depth_m / 2, round(w.depth_m / .05) + 1)
    xy = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    verts = np.column_stack((xy, sample_height(bundle.terrain, xy)))
    i = np.arange(len(xy)).reshape(len(y), len(x))
    faces = np.stack((i[:-1, :-1], i[:-1, 1:], i[1:, 1:], i[1:, :-1]), axis=-1).reshape(-1, 4)
    obj = mesh_object(bpy, "Contoured lawn soil", verts.tolist(), faces.tolist(), mat)
    obj.data.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))


def _vehicle(bpy, scene, cfg):
    chassis = bpy.data.objects.new("Cached chassis pose", None)
    scene.collection.objects.link(chassis)
    m = cfg.mower
    dark = material(bpy, "Textured graphite ABS", (.031, .041, .049), .41,
                    surface_noise=(370, .19, .003))
    shell = material(bpy, "Slate-green molded enclosure", (.047, .096, .076), .43,
                     surface_noise=(490, .10, .0003))
    bumper = material(bpy, "Bumper elastomer", (.017, .021, .022), .64)
    tread = material(bpy, "All-weather tire tread", (.023, .026, .028), .82)
    ring = material(bpy, "Anodized hub", (.155, .177, .166), .39, .6)
    light = material(bpy, "Amber status indicator", (.71, .32, .038), .34)
    box(bpy, "Lower sealed deck", (0, 0, .089), (m.length_m, m.width_m, .087),
        dark, .023, chassis)
    box(bpy, "Sculpted upper enclosure", (-.014, 0, .143),
        (.94 * m.length_m, .91 * m.width_m, .068), shell, .03, chassis)
    box(bpy, "Wrapped front bumper", (.449 * m.length_m, 0, .103),
        (.102 * m.length_m, .96 * m.width_m, .063), bumper, .015, chassis)
    box(bpy, "Rounded front cowl", (.411 * m.length_m, 0, .145),
        (.175 * m.length_m, .79 * m.width_m, .046), shell, .018, chassis)
    box(bpy, "Flush front gasket", (.466 * m.length_m, 0, .159),
        (.040 * m.length_m, .69 * m.width_m, .014), dark, .004, chassis)
    for y in (-.043, .043):
        box(bpy, "Recessed status lens", (.459 * m.length_m, y, .169),
            (.009, .007, .002), light, .001, chassis)
    # The driven axle uses the same body-space contact points as the cached paths.
    radius = .074
    for sign in (-1, 1):
        y = sign * m.track_m / 2
        tube(bpy, "Drive tire", (0, y - .0175, radius),
             (0, y + .0175, radius), radius, radius, tread, 32, chassis)
        tube(bpy, "Recessed axle hub", (0, y, radius),
             (0, y + sign * .019, radius), radius * .59, radius * .59, ring, 20, chassis)
    for x in (-.36 * m.length_m, .36 * m.length_m):
        radius = .037
        tube(bpy, "Center support roller", (x, -.012, radius), (x, .012, radius),
             radius, radius, tread, 20, chassis)
    return chassis


def _fence(bpy, bundle, rng, *, open_north):
    w = bundle.resolved_config.world
    wood = [material(bpy, f"Weathered cedar {i}", (
        .108 + i * .013, .074 + i * .010, .051 + i * .008),
        surface_noise=(12, .17, .004), grain=(2.0, 2.0, .15)) for i in range(5)]
    # Slat geometry is joined into one draw mesh per panel orientation.
    vertices, faces, indices = [], [], []

    def board(x, y, z, sx, sy, sz, visible=True):
        index = int(rng.integers(len(wood)))
        if not visible:
            return
        k = len(vertices)
        vertices.extend((x + a * sx / 2, y + b * sy / 2, z + c * sz / 2)
                        for c in (-1, 1) for b in (-1, 1) for a in (-1, 1))
        faces.extend(tuple(k + p for p in face) for face in (
            (0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
            (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)))
        indices.extend([index] * 6)

    for sign in (-1, 1):
        x = sign * (w.width_m / 2 - .04)
        for y in np.arange(-w.depth_m / 2 + .055, w.depth_m / 2, .116):
            h = rng.uniform(1.61, 1.68)
            board(x, y, h / 2 + .017, .041, .114, h)
        north_visible = sign != 1 or not open_north
        y = sign * (w.depth_m / 2 - .04)
        for x in np.arange(-w.width_m / 2 + .055, w.width_m / 2, .116):
            h = rng.uniform(1.61, 1.68)
            board(x, y, h / 2 + .017, .114, .041, h, north_visible)
        for y in np.arange(-w.depth_m / 2 + .15, w.depth_m / 2, 1.7):
            board(sign * (w.width_m / 2 - .105), y, .85, .13, .11, 1.72)
        for x in np.arange(-w.width_m / 2 + .15, w.width_m / 2, 1.7):
            board(x, sign * (w.depth_m / 2 - .105), .85, .11, .13, 1.72, north_visible)
        for z in (.44, 1.15):
            board(sign * (w.width_m / 2 - .087), 0., z, .048, w.depth_m - .16, .085)
            board(0., sign * (w.depth_m / 2 - .087), z, w.width_m - .16, .048, .085, north_visible)
    obj = mesh_object(bpy, "Weathered horizontal-railed timber fence", vertices, faces, wood[0])
    for mat in wood[1:]:
        obj.data.materials.append(mat)
    obj.data.polygons.foreach_set("material_index", np.asarray(indices, dtype="i"))


def _animated_mesh(bpy, name, vertices, faces, variants, materials, flexibility):
    points = np.asarray(vertices, dtype=np.float64)
    obj = mesh_object(bpy, name, vertices, faces, materials[0])
    for mat in materials[1:]:
        obj.data.materials.append(mat)
    obj.data.polygons.foreach_set("material_index", np.asarray(variants, dtype="i"))
    obj.data.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
    return FoliageReplay(obj.data, points, np.asarray(flexibility))


def _foliage(bpy, name, rng, clusters, materials, motion):
    """Cluster tapered elliptical leaves around branch tips and perennial stalks."""
    vertices, faces, variants, flex = [], [], [], []
    for center, extent, count, sizes in clusters:
        unit = rng.normal(size=(count, 3))
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        unit *= rng.uniform(.15, 1, count)[:, None] ** .39
        positions = center + unit * extent
        for center, size in zip(positions, rng.uniform(*sizes, count)):
            yaw = rng.uniform(0, np.pi * 2)
            a = size * np.array([np.cos(yaw), np.sin(yaw), rng.uniform(-.65, .65)])
            b = size * rng.uniform(.26, .47) * np.array([-np.sin(yaw), np.cos(yaw), rng.uniform(-.5, .5)])
            k = len(vertices)
            vertices.extend((center - .77 * a, center - .50 * a + .61 * b, center + b,
                             center + .48 * a + .70 * b, center + a, center + .48 * a - .70 * b,
                             center - b, center - .50 * a - .61 * b, center + [0, 0, .13 * size]))
            faces.extend((k + j, k + (j + 1) % 8, k + 8) for j in range(8))
            variants.extend([int(rng.integers(len(materials)))] * 8)
            flex.extend([motion * min(max(center[2], .03), 1.)] * 9)
    return _animated_mesh(bpy, name, vertices, faces, variants, materials, flex)


def _flowers(bpy, rng, roots):
    """Wind-driven groups of daisies and violet spikes, rooted within the planting bed."""
    stem = material(bpy, "Flower stems and sepals", (.098, .19, .047), .72, translucency=.15)
    ivory = material(bpy, "Pearl white ray florets", (.74, .69, .48), .64, translucency=.23)
    amber = material(bpy, "Warm pollen", (.68, .307, .033), .69)
    violet = material(bpy, "Violet blue salvia", (.219, .098, .415), .60, translucency=.16)
    mauve = material(bpy, "Lavender salvia tips", (.435, .224, .535), .60, translucency=.16)
    vertices, faces, variants, flex = [], [], [], []

    def add(verts, polygons, material_index, strength):
        k = len(vertices)
        vertices.extend(verts)
        faces.extend(tuple(k + j for j in face) for face in polygons)
        variants.extend([material_index] * len(polygons))
        flex.extend(np.broadcast_to(strength, len(verts)))

    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6)))
    for root, height, flower in roots:
        bend = rng.uniform(-.13, .13, 2)
        stalk = np.array([root + [bend[0] * u**2, bend[1] * u**2, height * u]
                          for u in (0., .35, .68, 1.)])
        stem_radii = rng.uniform(.0021, .0036) * np.array([1., .80, .56, .32])
        stem_flex = height * np.array([0., .11, .48, .99])
        add((stalk[:, None, :] + stem_radii[:, None, None] * ring).reshape(-1, 3),
            [(6*i+j, 6*i+(j+1)%6, 6*(i+1)+(j+1)%6, 6*(i+1)+j)
             for i in range(3) for j in range(6)], 0,
            np.repeat(stem_flex, 6))
        for k, pos in enumerate(stalk[1:-1], 1):
            direction = np.array([.056 * (-1)**k, rng.uniform(-.05, .05), .035])
            normal = np.array([-direction[1], direction[0], 0]) * .25
            add((pos, pos + .45 * direction + normal, pos + direction,
                 pos + .45 * direction - normal, pos + .50 * direction + [0., 0., .01]),
                ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), 0,
                stem_flex[k] + height * np.array([0., .02, .09, .02, .03]))
        tip = stalk[-1]
        if flower == "daisy":
            radius = rng.uniform(.026, .044)
            # The angled disk and thin translucent petals catch the low sun.
            yaw = rng.uniform(0., np.pi * 2)
            axis_a = np.array([np.cos(yaw), np.sin(yaw), .27])
            axis_b = np.array([-np.sin(yaw), np.cos(yaw), .56])
            for angle in np.linspace(0, 2 * np.pi, 10, endpoint=False):
                a = radius * (np.cos(angle) * axis_a + np.sin(angle) * axis_b)
                b = .19 * radius * (-np.sin(angle) * axis_a + np.cos(angle) * axis_b)
                add((tip + .16*a, tip + .65*a + b, tip + a,
                     tip + .65*a - b, tip + .57*a + [0, 0, .004]),
                    ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), 1, .99 * height)
                add((tip + .29*a, tip + .29*a + .32*b, tip + [0, 0, .009],
                     tip + .29*a - .32*b), ((0, 1, 2, 3),), 2, .99 * height)
        else:
            for level in range(6):
                pos = tip + [0, 0, level * .013]
                size = .021 * (1 - .095 * level)
                for angle in angles + .58 * level:
                    a = size * np.array([np.cos(angle), np.sin(angle), .53])
                    b = size * .40 * np.array([-np.sin(angle), np.cos(angle), 0.])
                    add((pos, pos + .61 * a + b, pos + a + [0., 0., .007],
                         pos + .69 * a - b), ((0, 1, 2, 3),), 3 + (level > 2), .99 * height)
    return _animated_mesh(bpy, "Daisies and salvia in the breeze", vertices, faces, variants,
                          [stem, ivory, amber, violet, mauve], flex)


def _garden(bpy, bundle, *, open_north):
    rng = np.random.default_rng([bundle.resolved_config.seed, 64])
    _fence(bpy, bundle, rng, open_north=open_north)
    w = bundle.resolved_config.world
    x0, y0, x1, y1 = bundle.terrain.obstacle_xyxy_m
    stone = [material(bpy, f"Weathered fieldstone {i}", color, surface_noise=(36, .33, .023))
             for i, color in enumerate(((.25, .21, .16), (.31, .285, .222),
                                        (.218, .236, .181), (.37, .321, .253)))]
    earth = material(bpy, "Dark flower-bed mulch", (.064, .034, .018),
                     surface_noise=(96, .54, .026))
    box(bpy, "Raised mulch bed", ((x0 + x1) / 2, (y0 + y1) / 2, .006),
        (x1 - x0, y1 - y0, .092), earth, .031)
    width, length = x1 - x0, y1 - y0
    if width > .16 and length > .2:
        for i, y in enumerate(np.arange(y0 + .117, y1 - .080, .232)):
            x = x0 + rng.uniform(.077, .088)
            z = float(sample_height(bundle.terrain, [x, y]))
            obj = box(bpy, "Rounded fieldstone", (x, y, z + rng.uniform(.023, .049)),
                      (.116, .216, rng.uniform(.107, .161)), stone[i % len(stone)], .030)
            obj.rotation_euler[2] = rng.uniform(-.12, .12)
        for y in (y0 + .067, y1 - .067):
            for x in np.arange(x0 + .188, x1 - .070, .222):
                z = float(sample_height(bundle.terrain, [x, y]))
                box(bpy, "Fieldstone return", (x, y, z + .030), (.207, .114, .137),
                    stone[int(rng.integers(len(stone)))], .029)

    leaves = [material(bpy, name, color, .61, translucency=.28) for name, color in (
        ("Young sunlit leaves", (.181, .277, .055)), ("Garden leaf green", (.087, .183, .037)),
        ("Olive midtone", (.153, .217, .053)), ("Dark undersides", (.056, .126, .032)))]
    bark = material(bpy, "Deeply ridged bark", (.103, .074, .047), surface_noise=(17, .62, .08),
                    grain=(1.4, 1.4, .26))
    tree_clusters = []
    trees = [
        (-1.07, 2.02, .96), (w.width_m * .5 + .78, w.depth_m * .39, 1.10),
        (-w.width_m * .5 - .72, -w.depth_m * .34, 1.22),
        (-w.width_m * .5 - .12, w.depth_m * .41, .77),
    ]
    for tree_x, tree_y, size in trees:
        base = np.array([tree_x, tree_y, float(sample_height(bundle.terrain, [tree_x, tree_y]))])
        trunk_top = base + np.array([.10, .08, 2.25]) * size
        tube(bpy, "Tapered orchard trunk", base - [0, 0, .1], trunk_top, .118*size, .052*size, bark)
        for i in range(9):
            angle = i * 2.39996 + rng.uniform(-.25, .25)
            outward = np.array([np.cos(angle), np.sin(angle), 0.])
            start = base + (trunk_top - base) * rng.uniform(.56, .96)
            end = trunk_top + size * (outward * rng.uniform(.67, 1.44) + [0., 0., rng.uniform(.16, 1.02)])
            tube(bpy, "Spreading orchard branch", start, end, .045*size, .009*size, bark, 8)
            for j in range(2):
                crown = end + size * (rng.normal(0, .20, 3) + outward * .15 * j)
                tree_clusters.append((crown, size * np.array([.62, .57, .43]), 240, (.034, .105)))
    # Thin branches tie the distant crowns to the orchard beyond the fence.
    def distant_crown(x, y, z, extent, count, sizes):
        sway = rng.uniform(-.28, .28, 2)
        base = np.array([x + sway[0], y + sway[1], -.1])
        tip = np.array([x, y, z])
        mid = base + .72 * (tip - base)
        tube(bpy, "Distant slender trunk", base, tip, .061, .014, bark, 7)
        for sign in (-1, 1):
            branch_tip = tip + [sign * extent[0] * .55, .12 * sign, -.10]
            tube(bpy, "Distant canopy twig", mid, branch_tip, .020, .005, bark, 6)
        tree_clusters.append((tip, extent, count, sizes))

    for x in np.linspace(-.55 * w.width_m, .55 * w.width_m, 9):
        for y in (-.64 * w.depth_m, .62 * w.depth_m):
            # Keep the north tree row behind the addition instead of through the facade.
            row_y = w.depth_m * .5 + 5.2 if open_north and y > 0 else y
            distant_crown(x, row_y, rng.uniform(1.6, 2.55), (.94, .73, .69), 340, (.045, .119))
    for y in np.linspace(-.47 * w.depth_m, .47 * w.depth_m, 8):
        for x in (-.63 * w.width_m, .65 * w.width_m):
            distant_crown(x, y, rng.uniform(1.9, 3.4), (1.08, 1.16, .90), 390, (.058, .142))
    motion = [_foliage(bpy, "Orchard and background canopy", rng, tree_clusters, leaves, .28)]

    plants = []
    for x in np.linspace(-w.width_m / 2 + .29, w.width_m / 2 - .29, 22):
        for sign in (-1, 1):
            y = sign * (w.depth_m / 2 - rng.uniform(.25, .56))
            plants.append(([x, y, rng.uniform(.27, .91)], [rng.uniform(.28, .51), .34, .35],
                           130, (.025, .095)))
    for y in np.linspace(-w.depth_m / 2 + .25, w.depth_m / 2 - .25, 17):
        if y0 - .33 < y < y1 + .33:
            signs = (-1,)
        else:
            signs = (-1, 1)
        for sign in signs:
            x = sign * (w.width_m / 2 - rng.uniform(.24, .55))
            plants.append(([x, y, rng.uniform(.28, .78)], [.32, .41, .36], 125, (.027, .094)))
    flowers = []
    if width > .30 and length > .35:
        for j, y in enumerate(np.linspace(y0 + .18, y1 - .18, 15)):
            x = x0 + width * rng.uniform(.31, .65)
            plants.append(([x, y, rng.uniform(.16, .30)], [.143, .132, .147], 83, (.026, .074)))
            if j % 3 != 1:
                for _ in range(5):
                    root = np.array([x0 + width * rng.uniform(.20, .88), y + rng.uniform(-.08, .08), .048])
                    species = "daisy" if j % 3 == 0 else "salvia"
                    height = rng.uniform(.22, .40) if species == "daisy" else rng.uniform(.29, .59)
                    flowers.append((root, height, species))
    motion.append(_foliage(bpy, "Perennials and shaded boundary hedges", rng, plants, leaves, .36))
    if flowers:
        motion.append(_flowers(bpy, rng, flowers))
    return motion


def _light(bpy, scene):
    from mathutils import Vector

    world = bpy.data.worlds.new("Outdoor skylight")
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    sky = nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = False
    sky.sun_elevation = .55
    sky.sun_rotation = 2.50
    background = nodes.get("Background")
    background.inputs["Strength"].default_value = .22
    links.new(sky.outputs["Color"], background.inputs["Color"])
    scene.world = world
    sun_data = bpy.data.lights.new("Late-afternoon sun", "SUN")
    sun_data.energy = 3.90
    sun_data.angle = np.deg2rad(2.6)
    sun_data.color = (1., .864, .684)
    sun = bpy.data.objects.new(sun_data.name, sun_data)
    sun.rotation_euler = Vector((.34, -.67, -.69)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(sun)


PRESETS = {
    "smoke": (640, 360, 8), "preview": (960, 540, 32), "beauty": (1280, 720, 48),
    "final": (1920, 1080, 128),
}


@dataclass
class SceneReplay:
    bundle: object
    scene: object
    camera: object
    chassis: object
    grass: object
    foliage: list[FoliageReplay]
    yard_life: object | None = None

    def apply(self, frame):
        fps = self.bundle.resolved_config.render.fps
        time = grass_time_s(frame, fps)
        whole = int(np.floor(frame))
        self.scene.frame_set(whole, subframe=frame - whole)
        apply_vehicle_pose(self.bundle, self.camera, self.chassis, time)
        self.grass.apply(time)
        for plant in self.foliage:
            plant.apply(time)
        if self.yard_life:
            self.yard_life.apply(time)


def build_scene(run_dir: str, preset: str, *, yard="legacy", asset_digest=None) -> SceneReplay:
    import bpy

    bundle = load_bundle(Path(run_dir))
    cfg = bundle.resolved_config
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y, scene.cycles.samples = PRESETS[preset]
    if preset == "final":
        scene.render.resolution_x, scene.render.resolution_y = cfg.render.width, cfg.render.height
        scene.cycles.samples = cfg.render.samples
    scene.render.resolution_percentage = 100
    scene.render.fps = cfg.render.fps
    scene.frame_start = 0
    scene.frame_end = cfg.frame_count - 1
    scene.render.use_motion_blur = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 5
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scene.cycles.adaptive_threshold = .035
    scene.cycles.max_bounces = 5
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.view_layers[0].cycles.denoising_store_passes = True
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = .14
    _light(bpy, scene)
    soil = material(bpy, "Patchy loam and basal grass", (.080, .104, .035),
                    surface_noise=(11, .23, .029))
    _ground(bpy, bundle, soil)
    from .blender_life import make_backyard_life

    foliage = _garden(bpy, bundle, open_north=yard == "house")
    if yard == "house":
        from .blender_house import make_house

        make_house(bpy, bundle, expected_digest=asset_digest)
    yard_life, shelter = make_backyard_life(bpy, bundle)
    if shelter is not None:
        foliage.insert(0, shelter)
    chassis = _vehicle(bpy, scene, cfg)
    camera = mount_camera(scene, cfg)
    if preset in {"beauty", "final"}:
        camera.data.dof.use_dof = True
        camera.data.dof.focus_distance = .85
        camera.data.dof.aperture_fstop = 9.5
    grass = make_grass(scene, bundle)
    replay = SceneReplay(bundle, scene, camera, chassis, grass, foliage, yard_life)
    replay.apply(0)
    return replay
