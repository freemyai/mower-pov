"""Build the procedural backyard inside Blender, then replay the cached timeline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..cache import load_bundle
from ..terrain import sample_height
from .blender_camera import apply_vehicle_pose, mount_camera
from .blender_grass import make_grass, grass_time_s


def material(bpy, name, rgb, roughness=.72, metallic=0., surface_noise=None):
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
    shell = material(bpy, "Slate-green molded enclosure", (.068, .14, .106), .33,
                     surface_noise=(490, .12, .001))
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
            (.013, .012, .002), light, .001, chassis)
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


def _fence(bpy, bundle, rng):
    w = bundle.resolved_config.world
    wood = [material(bpy, f"Weathered cedar {i}", (
        .251 + i * .027, .158 + i * .018, .088 + i * .012),
        surface_noise=(22, .15, .009)) for i in range(5)]
    # Slat geometry is joined into one draw mesh per panel orientation.
    vertices, faces, indices = [], [], []

    def board(x, y, z, sx, sy, sz):
        k = len(vertices)
        vertices.extend((x + a * sx / 2, y + b * sy / 2, z + c * sz / 2)
                        for c in (-1, 1) for b in (-1, 1) for a in (-1, 1))
        faces.extend(tuple(k + p for p in face) for face in (
            (0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
            (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)))
        indices.extend([int(rng.integers(len(wood)))] * 6)

    for sign in (-1, 1):
        x = sign * (w.width_m / 2 - .04)
        for y in np.arange(-w.depth_m / 2 + .055, w.depth_m / 2, .116):
            h = rng.uniform(1.61, 1.68)
            board(x, y, h / 2 + .017, .041, .114, h)
        y = sign * (w.depth_m / 2 - .04)
        for x in np.arange(-w.width_m / 2 + .055, w.width_m / 2, .116):
            h = rng.uniform(1.61, 1.68)
            board(x, y, h / 2 + .017, .114, .041, h)
        for y in np.arange(-w.depth_m / 2 + .15, w.depth_m / 2, 1.7):
            board(sign * (w.width_m / 2 - .105), y, .85, .13, .11, 1.72)
        for x in np.arange(-w.width_m / 2 + .15, w.width_m / 2, 1.7):
            board(x, sign * (w.depth_m / 2 - .105), .85, .11, .13, 1.72)
        for z in (.44, 1.15):
            board(sign * (w.width_m / 2 - .087), 0., z, .048, w.depth_m - .16, .085)
            board(0., sign * (w.depth_m / 2 - .087), z, w.width_m - .16, .048, .085)
    obj = mesh_object(bpy, "Weathered horizontal-railed timber fence", vertices, faces, wood[0])
    for mat in wood[1:]:
        obj.data.materials.append(mat)
    obj.data.polygons.foreach_set("material_index", np.asarray(indices, dtype="i"))


def _foliage(bpy, name, rng, clusters, foliage):
    """Spread individually oriented folded leaves within ellipsoidal crowns."""
    vertices, faces, variants = [], [], []
    for center, extent, count, sizes in clusters:
        unit = rng.normal(size=(count, 3))
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        unit *= rng.uniform(.23, 1, count)[:, None] ** .37
        positions = center + unit * extent
        for center, size in zip(positions, rng.uniform(*sizes, count)):
            yaw = rng.uniform(0, np.pi * 2)
            slope = rng.uniform(-.65, .65)
            a = size * np.array([np.cos(yaw), np.sin(yaw), slope])
            b = .37 * size * np.array([-np.sin(yaw), np.cos(yaw), rng.uniform(-.6, .6)])
            verts = (center - a * .65, center + b, center + a,
                     center - b, center + [0, 0, size * .13])
            k = len(vertices)
            vertices.extend(verts)
            faces.extend((k + j, k + ((j + 1) % 4), k + 4) for j in range(4))
            variants.extend([int(rng.integers(len(foliage)))] * 4)
    obj = mesh_object(bpy, name, vertices, faces, foliage[0])
    for mat in foliage[1:]:
        obj.data.materials.append(mat)
    obj.data.polygons.foreach_set("material_index", np.asarray(variants, dtype="i"))
    return obj


def _garden(bpy, bundle):
    rng = np.random.default_rng([bundle.resolved_config.seed, 64])
    _fence(bpy, bundle, rng)
    x0, y0, x1, y1 = bundle.terrain.obstacle_xyxy_m
    stone = material(bpy, "Weathered limestone", (.328, .283, .213),
                     surface_noise=(49, .38, .033))
    earth = material(bpy, "Flower-bed mulch", (.07, .035, .02),
                     surface_noise=(83, .62, .044))
    box(bpy, "Raised mulch bed", ((x0 + x1) / 2, (y0 + y1) / 2, .013),
        (x1 - x0, y1 - y0, .09), earth, .033)
    for y in np.arange(y0 + .11, y1, .227):
        z = float(sample_height(bundle.terrain, [x0 + .07, y]))
        box(bpy, "Irregular stone edge", (x0 + .062, y, z + .035),
            (.123, .218, rng.uniform(.13, .17)), stone, .022)
    for y in (y0 + .061, y1 - .061):
        for x in np.arange(x0 + .194, x1, .218):
            z = float(sample_height(bundle.terrain, [x, y]))
            box(bpy, "Stone return edge", (x, y, z + .032), (.21, .119, .14), stone, .019)
    green = [material(bpy, name, color, .76) for name, color in (
        ("Young leaves", (.117, .215, .046)), ("Green leaves", (.072, .135, .028)),
        ("Olive leaves", (.149, .189, .049)), ("Deep green leaves", (.043, .101, .027)))]
    bark = material(bpy, "Fissured tree bark", (.112, .078, .053),
                    surface_noise=(11, .62, .14))
    tree_clusters = []
    for tree_x, tree_y, size in ((-1.07, 1.84, 1.), (6.76, 3.69, 1.17),
                                (-6.4, -3.47, 1.2), (-5.34, 4.62, .7)):
        base = np.array([tree_x, tree_y, float(sample_height(bundle.terrain, [tree_x, tree_y]))])
        trunk_top = base + np.array([.14, .08, 2.31]) * size
        tube(bpy, "Tapered garden-tree trunk", base - [0, 0, .09], trunk_top,
             .132 * size, .063 * size, bark)
        for i in range(7):
            angle = i * 2.39996 + rng.uniform(-.28, .28)
            branch_start = base + (trunk_top - base) * rng.uniform(.57, .96)
            branch_x = np.cos(angle) * rng.uniform(.55, 1.4)
            branch_y = np.sin(angle) * rng.uniform(.55, 1.4)
            tip = trunk_top + size * np.array([branch_x, branch_y, rng.uniform(.28, 1.09)])
            tube(bpy, "Forked branch", branch_start, tip, .049 * size, .014 * size, bark, 8)
            for j in range(2):
                crown = tip + rng.normal(0, .25, 3) * size
                tree_clusters.append((crown, size * np.array([.67, .59, .44]), 360, (.037, .102)))
    _foliage(bpy, "Sunlit woodland canopy leaves", rng, tree_clusters, green)
    bed = []
    if x1 - x0 > .25 and y1 - y0 > .30:
        stem = material(bpy, "Perennial stems", (.076, .134, .033), .79)
        for y in np.linspace(y0 + .2, y1 - .2, 13):
            x = rng.uniform(x0 + .22, x1 - min(.13, .18 * (x1 - x0)))
            height = rng.uniform(.16, .30)
            bed.append((np.array([x, y, height]), [.177, .18, height * .63], 156, (.022, .066)))
            for dx, dy in ((-.055, 0), (.03, -.073), (.093, .058)):
                tube(bpy, "Perennial foliage stems", (x, y, .044),
                     (x + dx, y + dy, 1.22 * height), .008, .0025, stem, 6)
    for x in np.linspace(-5.44, 5.4, 27):
        bed.append(([x, -4.69, .44], [.37, .29, .42], 100, (.029, .078)))
        bed.append(([x, 4.68, .51], [.39, .27, .47], 100, (.028, .089)))
    _foliage(bpy, "Bed perennials and low shrubs", rng, bed, green)
    violet = [material(bpy, "Purple salvia flowers", (.209, .076, .252), .77),
              material(bpy, "Salvia bud highlights", (.334, .151, .363), .69)]
    flowers = [((x + .022, y, h + .16), [.055, .048, .102], 38, (.008, .018))
               for (x, y, h), _, _, _ in bed[:13:2]]
    if flowers:
        _foliage(bpy, "Purple inflorescences", rng, flowers, violet)


def _light(bpy, scene):
    from mathutils import Vector

    world = bpy.data.worlds.new("Outdoor skylight")
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    sky = nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = False
    sky.sun_elevation = .73
    sky.sun_rotation = 2.37
    background = nodes.get("Background")
    background.inputs["Strength"].default_value = .17
    links.new(sky.outputs["Color"], background.inputs["Color"])
    scene.world = world
    sun_data = bpy.data.lights.new("Late-afternoon sun", "SUN")
    sun_data.energy = 3.35
    sun_data.angle = np.deg2rad(1.7)
    sun_data.color = (1., .948, .846)
    sun = bpy.data.objects.new(sun_data.name, sun_data)
    sun.rotation_euler = Vector((.39, -.68, -.84)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(sun)


PRESETS = {"smoke": (640, 360, 8), "preview": (960, 540, 32)}


@dataclass
class SceneReplay:
    bundle: object
    scene: object
    camera: object
    chassis: object
    grass: object

    def apply(self, frame):
        fps = self.bundle.resolved_config.render.fps
        time = grass_time_s(frame, fps)
        whole = int(np.floor(frame))
        self.scene.frame_set(whole, subframe=frame - whole)
        apply_vehicle_pose(self.bundle, self.camera, self.chassis, time)
        self.grass.apply(time)


def build_scene(run_dir: str, preset: str) -> SceneReplay:
    import bpy

    bundle = load_bundle(Path(run_dir))
    cfg = bundle.resolved_config
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y, scene.cycles.samples = PRESETS[preset]
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
    soil = material(bpy, "Patchy loam and basal grass", (.069, .093, .032),
                    surface_noise=(14, .34, .065))
    _ground(bpy, bundle, soil)
    _garden(bpy, bundle)
    chassis = _vehicle(bpy, scene, cfg)
    camera = mount_camera(scene, cfg)
    grass = make_grass(scene, bundle)
    replay = SceneReplay(bundle, scene, camera, chassis, grass)
    replay.apply(0)
    return replay
