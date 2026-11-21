"""Original procedural animals and garden play-area geometry."""

from dataclasses import dataclass

import numpy as np

from ..actors.behavior import actor_pose, story_track
from ..actors.gait import DOG_LEGS, STRIDE_M, evaluate_dog
from ..obstacles import play_rails, toy_center, trampoline_uprights
from ..terrain import sample_height
from .blender_dog import SkinnedDog, make_dog
from .life_motion import (BALL_RADIUS, FENCE_Y, TRAMPOLINE_RADIUS, TRAMPOLINE_XY,
                          require_life_layout, shelter_xy)


def _empty(bpy, name, parent=None, position=(0., 0., 0.)):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    obj.location = position
    return obj


def _sphere(bpy, name, parent, location, scale, mat, segments=24):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=max(8, segments // 2))
    obj = bpy.context.object
    obj.name = name
    obj.parent = parent
    obj.location = location
    obj.scale = scale
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def _oriented_oval(bpy, name, parent, start, end, width, thickness, mat):
    from mathutils import Vector

    a, b = Vector(start), Vector(end)
    offset = b - a
    obj = _sphere(bpy, name, parent, (a + b) * .5, (width, thickness, offset.length * .62), mat)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = offset.to_track_quat("Z", "Y")
    return obj


def _ring_curve(bpy, name, paths, radius, mat, parent=None, closed=False):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = radius
    curve.bevel_resolution = 2
    for path in paths:
        spline = curve.splines.new("POLY")
        spline.points.add(len(path) - 1)
        spline.points.foreach_set("co", np.column_stack((path, np.ones(len(path)))).ravel())
        spline.use_cyclic_u = closed
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    curve.materials.append(mat)
    return obj


def _quills(bpy, parent, seed, palette):
    from .blender_scene import mesh_object

    rng = np.random.default_rng([seed, 713])
    n = 10500
    z = rng.uniform(-.34, 1, n)
    yaw = rng.uniform(0, np.pi * 2, n)
    r = np.sqrt(1. - z * z)
    unit = np.column_stack((r * np.cos(yaw), r * np.sin(yaw), z))
    unit = unit[unit[:, 0] < .59 + .23 * unit[:, 2]]
    n = len(unit)
    center = np.array([-.033, 0., .097])
    radii = np.array([.142, .091, .087])
    base = center + unit * radii
    normal = unit / radii
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    direction = normal * .59 + np.array([-.69, 0., .16]) + rng.normal(0., .09, (n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    across = np.cross(direction, [0., 1., 0.])
    across /= np.linalg.norm(across, axis=1, keepdims=True)
    tangent = np.cross(across, direction)
    widths = rng.uniform(.00055, .00120, n)
    lengths = rng.uniform(.015, .0305, n)
    positions = [0., .42, .68, .86, 1.]
    radii = [1., .92, .72, .44, .04]
    ring_phi = np.linspace(0, np.pi * 2, 3, endpoint=False)
    section = np.cos(ring_phi)[None, :, None] * across[:, None, :] + (
        np.sin(ring_phi)[None, :, None] * tangent[:, None, :])
    stem = base[:, None, :] + lengths[:, None, None] * np.asarray(positions)[None, :, None] * direction[:, None, :]
    vertices = stem[:, :, None, :] + widths[:, None, None, None] * np.asarray(radii)[None, :, None, None] * section[:, None, :, :]
    faces, indices = [], []
    warm = rng.random(n) > .41
    for i in range(4):
        for j in range(3):
            next_j = (j + 1) % 3
            faces.append(np.arange(n)[:, None] * 15
                         + [3 * i + j, 3 * i + next_j, 3 * (i + 1) + next_j, 3 * (i + 1) + j])
            index = (np.where(warm, 1, 0) if i == 0 else
                     np.where(warm, 0, 1) if i == 1 else
                     np.where(warm, 2, 3) if i == 2 else
                     np.where(warm, 3, 2))
            indices.append(index)
    obj = mesh_object(bpy, "Banded individual hedgehog spines", vertices.reshape(-1, 3).tolist(),
                      np.concatenate(faces).tolist(), palette[0])
    obj.parent = parent
    for mat in palette[1:]:
        obj.data.materials.append(mat)
    obj.data.polygons.foreach_set("material_index", np.concatenate(indices).astype("i"))


def _hedgehog(bpy, seed):
    from .blender_scene import material

    body = _empty(bpy, "European hedgehog")
    fur = material(bpy, "Hedgehog salt-and-pepper fur", (.167, .105, .064), .88,
                   surface_noise=(215, .25, .005), grain=(.7, 1.7, 1.8))
    face = material(bpy, "Fawn face and throat", (.281, .190, .114), .86,
                    surface_noise=(184, .36, .007), grain=(.55, 1.4, 1.6))
    ear_mat = material(bpy, "Hedgehog ear cartilage", (.126, .073, .050), .72)
    skin = material(bpy, "Hedgehog nose and toes", (.031, .019, .016), .42)
    eye = material(bpy, "Reflective black hedgehog eyes", (.011, .008, .006), .17)
    _sphere(bpy, "Arched back", body, (-.033, 0., .097), (.142, .091, .087), fur, 40)
    _sphere(bpy, "Soft belly", body, (-.011, 0., .064), (.128, .074, .059), face, 32)
    head = _empty(bpy, "Hedgehog sniff pivot", body, (.092, 0., .085))
    muzzle = _loft(bpy, "Tapered hedgehog face", head, [
        (-.050, .008, .008, .009), (-.035, .010, .042, .040), (-.002, .013, .054, .046),
        (.031, .010, .049, .036), (.056, .003, .035, .029), (.087, -.008, .024, .020),
        (.110, -.014, .016, .015), (.118, -.016, .006, .006),
    ], face)
    _fur_fibers(bpy, muzzle, head, seed, face, 8650)
    _sphere(bpy, "Wet nose", head, (.112, 0., -.017), (.015, .023, .016), skin)
    for sign in (-1, 1):
        _sphere(bpy, "Alert eye", head, (.045, sign * .041, .026), (.009, .008, .009), eye)
        ear = _sphere(bpy, "Small rounded ear", head, (-.018, sign * .044, .046),
                      (.019, .009, .023), ear_mat)
        ear.rotation_euler[0] = -.26 * sign
    whiskers = []
    for sign in (-1, 1):
        for i in range(6):
            whiskers.append(np.array([[.102, sign * .016, -.018 + .0012 * i],
                                      [.103 - .004 * i, sign * .034, -.025 + .004 * i],
                                      [.095 - .006 * i, sign * (.050 + .001 * i), -.028 + .008 * i]]))
    _ring_curve(bpy, "Fine snout whiskers", whiskers, .00042, face, parent=head)
    palette = [material(bpy, f"Keratin spine band {i}", color, .58, translucency=.08)
               for i, color in enumerate(((.070, .043, .027), (.167, .101, .056),
                                           (.46, .320, .164), (.71, .530, .293)))]
    _quills(bpy, body, seed, palette)
    feet = [_sphere(bpy, "Small hedgehog paw", body, (x, y, .018),
                    (.029, .019, .017), skin)
            for x, y in ((.065, -.054), (-.094, .054), (.065, .054), (-.094, -.054))]
    return body, head, feet


def _limb(bpy, name, parent, mat, r0, r1):
    from .blender_scene import tube

    return tube(bpy, name, (0., 0., -.5), (0., 0., .5), r0, r1, mat, 12, parent)


def _apply_segment(obj, start, end):
    from mathutils import Vector

    a, b = Vector(start), Vector(end)
    delta = b - a
    obj.location = (a + b) * .5
    obj.rotation_quaternion = delta.to_track_quat("Z", "Y")
    obj.scale.z = delta.length


def _loft(bpy, name, parent, sections, mat, coat_colors=None):
    """Sample a smooth elliptical skin along the animal's forward axis."""
    from .blender_scene import mesh_object

    sec = np.asarray(sections)
    x = sec[:, 0]
    data = sec[:, 1:]
    tangent = np.gradient(data, x, axis=0)
    xx, profile = [], []
    for j in range(len(x) - 1):
        u = np.linspace(0., 1., 5, endpoint=False)
        h = x[j + 1] - x[j]
        t = u[:, None]
        values = ((2*t**3 - 3*t**2 + 1) * data[j] + (t**3 - 2*t**2 + t) * h * tangent[j]
                  + (-2*t**3 + 3*t**2) * data[j + 1] + (t**3 - t**2) * h * tangent[j + 1])
        xx.extend(x[j] + h * u)
        profile.extend(values.tolist())
    xx.append(x[-1])
    profile.append(data[-1])
    center, width, height = np.asarray(profile).T
    theta = np.linspace(0., 2 * np.pi, 40, endpoint=False)
    yy = np.maximum(width[:, None], .002) * np.cos(theta)
    zz = center[:, None] + np.maximum(height[:, None], .002) * np.sin(theta)
    verts = np.stack((np.broadcast_to(np.asarray(xx)[:, None], yy.shape), yy, zz), axis=-1).reshape(-1, 3)
    mesh_faces = [(k * 40 + j, k * 40 + (j+1) % 40, (k+1) * 40 + (j+1) % 40, (k+1) * 40 + j)
                  for k in range(len(xx) - 1) for j in range(40)]
    mesh_faces.extend((tuple(range(39, -1, -1)), tuple(range((len(xx)-1)*40, len(xx)*40))))
    obj = mesh_object(bpy, name, verts.tolist(), mesh_faces, mat)
    obj.parent = parent
    obj.data.polygons.foreach_set("use_smooth", np.ones(len(mesh_faces), dtype=bool))
    if coat_colors:
        vr, vy, vz = verts.T
        if coat_colors == "body":
            saddle = np.clip((vz - .355) / .093, 0, 1) * np.clip((.193 - vr) / .16, 0, 1)
            pale = np.clip((vr - .13) / .13, 0, 1) * np.clip((.38 - vz) / .15, 0, 1)
        else:
            saddle = np.clip((vz - .054) / .090, 0, 1) * np.clip((.135 - vr) / .1, 0, 1) * .37
            pale = np.clip((vr - .072) / .076, 0, 1) * np.clip((.059 - vz) / .065, 0, 1)
        f = .96 + .04 * np.cos(136 * vr + 35 * vy) * np.sin(193 * vz - 24 * vy)
        base = np.array([.47, .252, .100])
        rgb = f[:, None] * (base + saddle[:, None] * (np.array([.233, .097, .037]) - base)
                            + pale[:, None] * (np.array([.70, .53, .35]) - base))
        attr = obj.data.color_attributes.new(name="Coat", type="FLOAT_COLOR", domain="POINT")
        attr.data.foreach_set("color", np.column_stack((rgb, np.ones(len(rgb)))).astype("f").ravel())
    return obj



def _fur_fibers(bpy, surface, parent, seed, mat, count):
    """Scatter tapered fibers on the modeled face while leaving the nose bare."""
    from .blender_scene import mesh_object

    rng = np.random.default_rng([seed, 714])
    surface.data.calc_loop_triangles()
    triangles = np.asarray([tuple(t.vertices) for t in surface.data.loop_triangles])
    points = np.asarray([tuple(v.co) for v in surface.data.vertices])[triangles]
    points = points[points[:, :, 0].mean(axis=1) < .112]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    area = np.linalg.norm(cross, axis=1)
    ids = rng.choice(len(points), count, p=area / area.sum())
    normal = cross[ids] / area[ids, None]
    uv = rng.random((count, 2))
    uv = np.where(uv.sum(axis=1, keepdims=True) > 1, 1 - uv, uv)
    chosen = points[ids]
    root = chosen[:, 0] + uv[:, :1] * (chosen[:, 1] - chosen[:, 0]) + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])
    direction = normal + [-.44, 0., .10] + rng.normal(0, .30, (count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    width = np.cross(direction, [0, 0, 1.0])
    width *= rng.uniform(.00007, .00023, (count, 1))
    tip = root + direction * rng.uniform(.0020, .0085, (count, 1))
    vertices = np.stack((root - width, root + width, tip), axis=1).reshape(-1, 3)
    fibers = mesh_object(bpy, "Fine hedgehog facial hairs", vertices.tolist(),
                         np.arange(count * 3).reshape(-1, 3).tolist(), mat)
    fibers.parent = parent



def _dog(bpy):
    from .blender_scene import material

    obj = _empty(bpy, "Debug proxy · garden dog")
    coat = material(bpy, "Golden terrier coat", (.49, .252, .092), .84,
                    surface_noise=(67, .28, .009), grain=(.50, 1.5, 1.25))
    dark = material(bpy, "Caramel ears and saddle", (.234, .102, .042), .81,
                    surface_noise=(64, .26, .009))
    cream = material(bpy, "Pale chest and feet", (.63, .48, .29), .81,
                     surface_noise=(116, .21, .007))
    eye = material(bpy, "Dark eyes and nose", (.020, .011, .007), .31)
    collar = material(bpy, "Woven turquoise dog collar", (.025, .19, .197), .65)
    skin = material(bpy, "Continuous terrier coat", (.47, .252, .10), .84,
                    surface_noise=(225, .36, .004), grain=(.54, 1.7, 1.4))
    nodes, links = skin.node_tree.nodes, skin.node_tree.links
    color = nodes.new("ShaderNodeVertexColor")
    color.layer_name = "Coat"
    links.new(color.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
    _loft(bpy, "Continuous rib cage and neck", obj, [
        (-.322, .350, .003, .005), (-.296, .343, .064, .066), (-.236, .326, .100, .111),
        (-.144, .325, .102, .113), (-.054, .331, .092, .100), (.049, .335, .108, .120),
        (.130, .348, .118, .132), (.199, .361, .103, .123), (.248, .410, .083, .090),
        (.298, .455, .068, .065), (.329, .460, .022, .033),
    ], skin, "body")
    head = _empty(bpy, "Dog head and gaze", obj, (.283, 0., .452))
    _loft(bpy, "Terrier skull and tapered muzzle", head, [
        (-.071, .028, .007, .013), (-.049, .034, .059, .067), (-.007, .035, .077, .083),
        (.046, .029, .082, .086), (.090, .005, .069, .061), (.122, -.009, .057, .041),
        (.166, -.008, .044, .032), (.192, -.001, .033, .026), (.200, .001, .011, .012),
    ], skin, "head")
    _sphere(bpy, "Dog nose", head, (.190, 0., .006), (.018, .036, .018), eye)
    ears = []
    for sign in (-1, 1):
        _sphere(bpy, "Dog eye", head, (.076, sign * .071, .043), (.014, .013, .014), eye)
        ear = _empty(bpy, "Floppy ear pivot", head, (.006, sign * .065, .055))
        _oriented_oval(bpy, "Dropped ear", ear, (0., 0., .011), (-.024, sign * .037, -.124),
                       .048, .022, dark)
        ears.append(ear)
    theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    ring = np.column_stack((.213 + .049 * np.cos(theta), .090 * np.sin(theta),
                            .400 - .062 * np.cos(theta)))
    _ring_curve(bpy, "Turquoise collar", [ring], .012, collar, parent=obj, closed=True)
    tail = _empty(bpy, "Dog tail pivot", obj, (-.249, 0., .365))
    _oriented_oval(bpy, "Feathered tail base", tail, (0, 0, 0), (-.16, .012, .083),
                   .044, .039, coat)
    _oriented_oval(bpy, "Tail tip", tail, (-.129, .012, .071), (-.247, .019, .158),
                   .027, .024, cream)
    legs = {}
    for spec in DOG_LEGS:
        top = _limb(bpy, "Dog upper leg", obj, coat, .043 if spec.x > 0 else .052, .035)
        bottom = _limb(bpy, "Dog shin", obj, cream, .036, .025)
        paw = _sphere(bpy, "Rounded dog paw", obj, (spec.x, spec.y, .029),
                      (.056, .034, .029), cream)
        legs[spec.name] = top, bottom, paw
    return obj, head, ears, tail, legs


def _apply_proxy_dog(dog, gait, time):
    obj, head, ears, tail, legs = dog
    pose = gait.root
    obj.rotation_euler[2] = pose.heading
    obj.location = (*pose.xy, gait.body_z)
    phase = 2 * np.pi * pose.distance / STRIDE_M
    head.rotation_euler = (0., .07 * np.sin(2 * phase) * pose.moving
                           + .20 * float(np.clip((time - 21.8) / 1.1, 0., 1.)), .115)
    tail.rotation_euler[2] = pose.moving * .24 * np.sin(phase - .5)
    for i, ear in enumerate(ears):
        ear.rotation_euler[1] = pose.moving * .15 * np.sin(2 * phase + .35 * i)
    for name, (top, bottom, paw) in legs.items():
        leg = gait.legs[name]
        _apply_segment(top, leg.hip, leg.knee)
        _apply_segment(bottom, leg.knee, leg.ankle)
        paw.location = leg.ankle


def _play_area(bpy, terrain, scenario="legacy"):
    from .blender_scene import box, material, tube

    beam = material(bpy, "Honey-toned play fence", (.303, .170, .079), .80,
                    surface_noise=(14, .28, .012), grain=(.28, 2, 2))
    resilient = material(bpy, "Trampoline turquoise padding", (.034, .245, .213), .69,
                         surface_noise=(205, .18, .002))
    textile = material(bpy, "Trampoline woven black mat", (.035, .045, .043), .92,
                       surface_noise=(175, .29, .004))
    metal = material(bpy, "Painted play frame", (.093, .113, .112), .45, .48)
    orange = material(bpy, "Terracotta toy shell", (.62, .127, .032), .50)
    blue = material(bpy, "Dusty toy blue", (.058, .179, .339), .52)
    yellow = material(bpy, "Golden yellow sand molds", (.68, .42, .081), .54)
    offwhite = material(bpy, "Ivory ball panels", (.70, .63, .44), .68)
    red = material(bpy, "Warm red ball panels", (.68, .102, .045), .72)
    for x0, x1 in play_rails(scenario):
        count = max(2, int(np.ceil((x1 - x0) / .75)) + 1)
        for x in np.linspace(x0, x1, count):
            z = float(sample_height(terrain, [x, FENCE_Y]))
            box(bpy, "Low yard divider post", (x, FENCE_Y, z + .145), (.075, .075, .29), beam, .012)
        box(bpy, "Low wooden divider rail", ((x0 + x1) / 2, FENCE_Y, .165),
            (x1 - x0 + .08, .049, .066), beam, .008)
    tx, ty = TRAMPOLINE_XY
    z0 = float(sample_height(terrain, [tx, ty]))
    mat = _empty(bpy, "Mini garden trampoline", position=(tx, ty, z0))
    angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    xy = TRAMPOLINE_RADIUS * np.column_stack((np.cos(angles), np.sin(angles)))
    _ring_curve(bpy, "Trampoline padded rim", [np.column_stack((xy, np.full(80, .39)))],
                .054, resilient, parent=mat, closed=True)
    _sphere(bpy, "Woven rebound mat", mat, (0, 0, .376),
            (TRAMPOLINE_RADIUS - .039, TRAMPOLINE_RADIUS - .039, .020), textile, 48)
    uprights = trampoline_uprights()
    for x, y in uprights:
        tube(bpy, "Padded net upright", (x, y, .36), (.93*x, .93*y, 1.35),
             .027, .022, resilient, 12, mat)
    for a, b in uprights.reshape(3, 2, 2):
        start, end = a * 1.050, b * 1.050
        ground_a, ground_b = sample_height(terrain, np.array([start, end]) + [tx, ty]) - z0
        base_a = np.array([*start, ground_a + .024])
        base_b = np.array([*end, ground_b + .024])
        radius = .041
        length = np.linalg.norm(end - start)
        u = radius / length
        along = (base_b - base_a) / length
        # Each bent U frame has two endpoint contacts fitted to the actual lawn.
        path = [[*a, .375], base_a + [0, 0, radius * 1.5]]
        path.extend(base_a + radius * (along * (1 - np.cos(t)) + [0, 0, 1 - np.sin(t)])
                    for t in np.linspace(0, np.pi/2, 7))
        middle = [base_a + fraction * (base_b - base_a)
                  for fraction in np.linspace(u, 1 - u, 5)[1:-1]]
        path.extend(middle)
        path.extend(base_b + radius * (-along * (1 - np.cos(t)) + [0, 0, 1 - np.sin(t)])
                    for t in np.linspace(np.pi/2, 0, 7))
        path.extend((base_b + [0, 0, radius * 1.5], [*b, .375]))
        _ring_curve(bpy, "Bent U-leg support", [np.asarray(path)], .026, metal, parent=mat)
    _ring_curve(bpy, "Enclosure top band", [np.column_stack((xy * .87, np.full(80, 1.35)))],
                .015, metal, parent=mat, closed=True)
    net_angles = np.linspace(0, 2 * np.pi, 52, endpoint=False)
    net_xy = .94 * TRAMPOLINE_RADIUS * np.column_stack((np.cos(net_angles), np.sin(net_angles)))
    vertical = [np.array([[x, y, .416], [.93*x, .93*y, .88], [.926*x, .926*y, 1.35]])
                for x, y in net_xy]
    horizontal = [np.column_stack((xy * factor, np.full(80, height))) for factor, height in
                  zip(np.linspace(.94, .87, 14), np.linspace(.43, 1.33, 14))]
    _ring_curve(bpy, "Woven enclosure vertical cords", vertical, .0017, textile, parent=mat)
    _ring_curve(bpy, "Woven enclosure horizontal cords", horizontal, .0016, textile, parent=mat, closed=True)
    x, y = toy_center(scenario)
    z = float(sample_height(terrain, [x, y]))
    box(bpy, "Toy truck bed", (x, y, z + .092), (.245, .151, .073), blue, .012)
    box(bpy, "Toy truck cab", (x - .075, y, z + .147), (.091, .144, .09), orange, .012)
    for xx in (x - .073, x + .074):
        for sign in (-1, 1):
            tube(bpy, "Toy truck wheel", (xx, y + sign * .067, z + .04),
                 (xx, y + sign * .091, z + .04), .041, .041, textile, 14)
    box(bpy, "Small yellow sand mold", (x + .19, y + .108, z + .048),
        (.08, .07, .086), yellow, .015)
    ball = _sphere(bpy, "Panelled rolling play ball", None, (0, 0, 0),
                   (BALL_RADIUS,) * 3, offwhite, 32)
    ball.data.materials.append(red)
    for face in ball.data.polygons:
        normal = face.center
        longitude = np.arctan2(normal.y, normal.x)
        band = int(np.floor((longitude + np.pi) * 8 / (2 * np.pi)))
        face.material_index = int(band % 2 == 0 and abs(normal.z) < .83)
    return ball


@dataclass
class BackyardLife:
    bundle: object
    hedgehog: tuple
    dog: SkinnedDog | tuple
    ball: object

    def apply(self, time):
        from mathutils import Quaternion

        obj, head, feet = self.hedgehog
        terrain = self.bundle.terrain
        pose = actor_pose(self.bundle, "hedgehog", time)
        obj.location = (*pose.xy, float(sample_height(terrain, pose.xy)))
        obj.rotation_euler[2] = pose.heading
        head.rotation_euler = (.025 * np.sin(4.3 * time), .11 * np.sin(7.2 * time) + .08, .08 * np.sin(3.7 * time))
        for foot_idx, (foot, x) in enumerate(zip(feet, (.065, -.094, .065, -.094))):
            phase = pose.distance * 49.0 + np.pi * (foot_idx // 2)
            foot.location.x = x + pose.moving * .023 * np.cos(phase)
            foot.location.z = .016 + pose.moving * .011 * max(0., np.sin(phase))

        gait = evaluate_dog(time, terrain, story_track(self.bundle, "dog"))
        if isinstance(self.dog, SkinnedDog):
            self.dog.apply(gait)
        else:
            _apply_proxy_dog(self.dog, gait, time)

        pose = actor_pose(self.bundle, "ball", time)
        self.ball.location = (*pose.xy, float(sample_height(terrain, pose.xy)) + BALL_RADIUS)
        self.ball.rotation_mode = "QUATERNION"
        self.ball.rotation_quaternion = (Quaternion((0., 1., 0.), -pose.distance / BALL_RADIUS)
                                         @ Quaternion((0., 0., 1.), .23))


def make_backyard_life(bpy, bundle, *, debug_dog=None):
    from .blender_scene import _foliage, material

    require_life_layout(bundle)
    seed = bundle.resolved_config.seed
    rng = np.random.default_rng([seed, 561])
    leaf = [material(bpy, f"Hedgehog shelter leaf {i}", color, .71, translucency=.24)
            for i, color in enumerate(((.133, .215, .049), (.101, .163, .040), (.183, .234, .051)))]
    shelter = _foliage(bpy, "Low hedgehog shelter", rng,
                       [((*shelter_xy(bundle), .125), (.37, .29, .139), 1340, (.024, .066))],
                       leaf, .32)
    if debug_dog is None:
        debug_dog = bundle.story is None
    dog = _dog(bpy) if debug_dog else make_dog(bpy, seed)
    return BackyardLife(bundle, _hedgehog(bpy, seed), dog,
                        _play_area(bpy, bundle.terrain, bundle.resolved_config.scenario)), shelter
