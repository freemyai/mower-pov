"""Original short-haired terrier, released under the project's GPL-3.0-or-later.

The rest mesh is a voxel union of anatomical profiles in metres. The +X axis
points towards the nose. Explicit weight ramps blend neighbouring bones at
each joint. The soles and facial details have rigid weights so they retain
their shape. Groom vertices inherit the skin weights at their roots.
"""

from dataclasses import dataclass

import numpy as np

from ..actors.gait import DOG_LEGS, SOLE_TO_ANKLE_M, STRIDE_M, solve_knee


def _ease(start, end, values):
    u = np.clip((values - start) / (end - start), 0., 1.)
    return u * u * (3. - 2. * u)


def _sweep(sections, segments=24, side=(0., 1., 0.)):
    """Closed elliptical profiles with the first radius across the animal."""
    sections = np.asarray(sections)
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(sections[:, :3], axis=0), axis=1))]
    slopes = np.gradient(sections, distances, axis=0)
    t = np.arange(5)[:, None] / 5.
    curved = []
    for i, span in enumerate(np.diff(distances)):
        curved.append(((2 * t**3 - 3 * t**2 + 1) * sections[i]
                       + (t**3 - 2 * t**2 + t) * span * slopes[i]
                       + (-2 * t**3 + 3 * t**2) * sections[i + 1]
                       + (t**3 - t**2) * span * slopes[i + 1]))
    sections = np.vstack((*curved, sections[-1:]))
    centers, radii = sections[:, :3], sections[:, 3:]
    radii = radii.clip(.001, None)
    tangent = np.gradient(centers, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    across = np.broadcast_to(side, tangent.shape).copy()
    across -= (across * tangent).sum(axis=1, keepdims=True) * tangent
    across /= np.linalg.norm(across, axis=1, keepdims=True)
    depth = np.cross(tangent, across)
    angles = np.arange(segments) * (2 * np.pi / segments)
    vertices = (centers[:, None, :] + (radii[:, :1] * np.cos(angles))[:, :, None] * across[:, None, :]
                + (radii[:, 1:] * np.sin(angles))[:, :, None] * depth[:, None, :])
    faces = [(i * segments + j, i * segments + (j + 1) % segments,
              (i + 1) * segments + (j + 1) % segments, (i + 1) * segments + j)
             for i in range(len(sections) - 1) for j in range(segments)]
    faces += [tuple(reversed(range(segments))), tuple(range((len(sections) - 1) * segments,
                                                           len(sections) * segments))]
    return vertices.reshape(-1, 3), faces


def _bone_specs():
    bones = {
        "pelvis": ((-.206, 0., .385), (-.092, 0., .396), None),
        "spine": ((-.092, 0., .396), (.082, 0., .400), "pelvis"),
        "chest": ((.082, 0., .400), (.206, 0., .406), "spine"),
        "neck": ((.206, 0., .406), (.284, 0., .491), "chest"),
        "head": ((.284, 0., .491), (.420, 0., .493), "neck"),
        "tail": ((-.252, 0., .410), (-.371, 0., .468), "pelvis"),
        "tail_tip": ((-.371, 0., .468), (-.509, 0., .511), "tail"),
    }
    for sign in (-1, 1):
        bones[f"ear{sign:+}"] = ((.274, sign * .057, .545), (.242, sign * .110, .466), "head")
    for spec in DOG_LEGS:
        hip = np.array([spec.x, spec.y * .79, .341])
        ankle = np.array([spec.x, spec.y, SOLE_TO_ANKLE_M])
        knee = solve_knee(hip, ankle, spec)
        bones[spec.name + ".upper"] = (hip, knee, "chest" if spec.x > 0 else "pelvis")
        bones[spec.name + ".lower"] = (knee, ankle, spec.name + ".upper")
        bones[spec.name + ".paw"] = (ankle, ankle + [.052, 0., 0.], spec.name + ".lower")
    return bones


def _anatomy(bones):
    trunk = [
        (-.301, 0., .376, .009, .015), (-.280, 0., .366, .067, .063),
        (-.248, 0., .362, .098, .089), (-.202, 0., .360, .108, .097),
        (-.143, 0., .367, .100, .089), (-.085, 0., .376, .082, .077),
        (-.023, 0., .371, .083, .077), (.035, 0., .362, .099, .095),
        (.096, 0., .360, .109, .101), (.149, 0., .363, .105, .108),
        (.196, 0., .384, .091, .096), (.234, 0., .396, .074, .077),
        (.267, 0., .402, .047, .045), (.278, 0., .407, .011, .014),
    ]
    neck = [
        (.136, 0., .345, .046, .047), (.173, 0., .378, .086, .077),
        (.211, 0., .422, .077, .081), (.242, 0., .452, .065, .071),
        (.264, 0., .479, .059, .067), (.284, 0., .503, .054, .056),
        (.303, 0., .517, .024, .026),
    ]
    head = [
        (.215, 0., .494, .009, .015), (.237, 0., .498, .049, .052),
        (.264, 0., .508, .068, .061), (.292, 0., .513, .073, .060),
        (.317, 0., .509, .072, .054), (.338, 0., .502, .068, .045),
        (.359, 0., .486, .057, .034), (.382, 0., .478, .054, .031),
        (.410, 0., .476, .047, .028), (.432, 0., .475, .036, .023),
        (.449, 0., .475, .031, .019),
        (.460, 0., .475, .016, .014),
    ]
    tail = [
        (-.247, 0., .409, .035, .033), (-.288, 0., .429, .034, .033),
        (-.330, 0., .452, .028, .028), (-.372, 0., .471, .023, .023),
        (-.414, 0., .488, .019, .019), (-.457, 0., .506, .014, .014),
        (-.488, 0., .512, .010, .011), (-.511, 0., .510, .004, .004),
        (-.516, 0., .509, .002, .002),
    ]
    profiles = [(trunk, (0., 1., 0.)), (neck, (0., 1., 0.)),
                (head, (0., 1., 0.)), (tail, (0., 1., 0.))]
    for sign in (-1, 1):
        ear = [
            (.271, sign * .046, .548, .024, .016),
            (.269, sign * .067, .552, .037, .016),
            (.263, sign * .088, .546, .043, .014),
            (.253, sign * .107, .526, .038, .012),
            (.247, sign * .112, .498, .031, .010),
            (.242, sign * .111, .472, .023, .009),
            (.246, sign * .104, .454, .014, .006),
            (.250, sign * .098, .448, .005, .003),
        ]
        profiles.append((ear, (1., 0., 0.)))
    for spec in DOG_LEGS:
        hip, knee, _ = bones[spec.name + ".upper"]
        _, ankle, _ = bones[spec.name + ".lower"]
        # Upper volume extends into the shoulder or haunch. Forelegs are more
        # columnar than the broad haunch and swept rear hock.
        front = spec.x > 0
        upper = knee - hip
        lower = ankle - knee
        shoulder = hip - .27 * upper
        side = (0., 1., 0.)
        profiles.append(([
            (*shoulder, .036, .053), (*hip, .048 if front else .057, .071 if front else .089),
            (*(hip + .28 * upper), .043 if front else .053, .060 if front else .076),
            (*(hip + .64 * upper), .032 if front else .035, .041 if front else .047),
            (*knee, .027, .030), (*(knee + .18 * lower), .025, .026),
            (*(knee + .44 * lower), .022, .023),
            (*(knee + .73 * lower), .0185, .0200), (*ankle, .020, .021),
            (*(ankle + .060 * lower), .018, .018),
        ], side))
        x, y, _ = ankle
        profiles.append(([
            (x - .035, y, .023, .010, .010), (x - .025, y, .026, .026, .0255),
            (x - .006, y, .025, .030, .025), (x + .018, y, .021, .032, .025),
            (x + .038, y, .018, .030, .020), (x + .052, y, .016, .021, .016),
            (x + .058, y, .016, .005, .007),
        ], side))
    return profiles


def _build_surface(bpy, bones, coat):
    import bmesh

    from .blender_scene import mesh_object

    vertices, faces = [], []
    for sections, side in _anatomy(bones):
        points, polygons = _sweep(sections, side=side)
        faces.extend([[len(vertices) + i for i in face] for face in polygons])
        vertices.extend(points.tolist())
    source = mesh_object(bpy, "Terrier anatomical volumes", vertices, faces, coat)
    mesh = bmesh.new()
    mesh.from_mesh(source.data)
    bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))
    mesh.to_mesh(source.data)
    mesh.free()
    bpy.context.view_layer.objects.active = source
    source.data.remesh_voxel_size = .0047
    bpy.ops.object.voxel_remesh()
    smooth = source.modifiers.new("Smooth the anatomical union", 'SMOOTH')
    smooth.factor, smooth.iterations = .95, 6
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    source.name = "Terrier continuous skin"
    source.data.name = "Welded terrier skin"
    points = np.empty(len(source.data.vertices) * 3, dtype=np.float32)
    source.data.vertices.foreach_get("co", points)
    points = points.reshape(-1, 3).astype(np.float64)
    points[:, 2] = np.maximum(points[:, 2] - .002 * (1. - _ease(.003, .023, points[:, 2])), 0.)
    source.data.vertices.foreach_set("co", points.astype(np.float32).ravel())
    source.data.update()
    source.data.polygons.foreach_set("use_smooth", np.ones(len(source.data.polygons), dtype=bool))
    return source, points


def _skin_weights(points, bones):
    x, y, z = points.T
    weights = {name: np.zeros(len(points)) for name in bones}
    pelvis = 1. - _ease(-.190, -.049, x)
    chest = _ease(-.040, .166, x)
    weights["pelvis"] = pelvis
    weights["spine"] = (1. - pelvis) * (1. - chest)
    weights["chest"] = (1. - pelvis) * chest
    neck = _ease(.152, .280, x) * _ease(.335, .458, z)
    head = np.maximum(_ease(.255, .347, x), _ease(.473, .536, z) * _ease(.202, .269, x))
    tail = (1. - _ease(-.345, -.272, x)) * _ease(.350, .428, z)
    tip = 1. - _ease(-.452, -.341, x)
    for name in ("pelvis", "spine", "chest"):
        weights[name] *= (1. - neck) * (1. - head) * (1. - tail)
    weights["neck"] = neck * (1. - head)
    weights["head"] = head
    weights["tail"] = tail * (1. - tip)
    weights["tail_tip"] = tail * tip

    for sign in (-1, 1):
        ear = (_ease(.061, .096, y * sign) * (1. - _ease(.286, .313, x))
               * _ease(.438, .475, z))
        for name in ("head", "neck", "chest"):
            weights[name] *= 1. - ear
        weights[f"ear{sign:+}"] = ear

    # Estimate ownership using distance to each limb in the rest surface.
    # Falloff keeps the opposite knee from pulling on the stifle.
    def segment_distance(start, end):
        axis = end - start
        t = np.clip(((points - start) @ axis) / (axis @ axis), 0., 1.)
        return ((points - start - t[:, None] * axis) ** 2).sum(axis=1)

    distances = []
    for spec in DOG_LEGS:
        hip, knee, _ = bones[spec.name + ".upper"]
        _, ankle, _ = bones[spec.name + ".lower"]
        distances.append(np.minimum(segment_distance(hip, knee), segment_distance(knee, ankle)))
    distances = np.column_stack(distances)
    limb_mix = np.exp(-(distances - distances.min(axis=1, keepdims=True)) / .00065)
    limb_mix[limb_mix < .001] = 0.
    limb_mix /= limb_mix.sum(axis=1, keepdims=True)
    region = _ease(.023, .083, np.abs(y)) * np.maximum(1 - _ease(-.135, -.052, x), _ease(-.044, .069, x))
    leg_share = np.maximum(region * (1 - _ease(.218, .408, z)), 1 - _ease(.212, .263, z))
    for i, spec in enumerate(DOG_LEGS):
        name = spec.name
        knee = bones[name + ".upper"][1]
        share = leg_share * limb_mix[:, i]
        upper = _ease(knee[2] - .032, knee[2] + .046, z)
        # Keep the sole and digits rigid all swing long.
        paw = 1. - _ease(.050, .104, z)
        weights[name + ".upper"] = share * upper
        weights[name + ".lower"] = share * (1. - upper) * (1. - paw)
        weights[name + ".paw"] = share * (1. - upper) * paw
    for name in ("pelvis", "spine", "chest", "neck", "head", "tail", "tail_tip", "ear-1", "ear+1"):
        weights[name] *= 1. - leg_share
    values = np.column_stack(list(weights.values()))
    values /= values.sum(axis=1, keepdims=True)
    return values


def _bind(mesh, rig, weights):
    """Assign normalized weights with 1/1024 precision and bounded Blender calls."""
    names = tuple(weights)
    values = np.column_stack(tuple(weights.values()))
    quantized = np.rint(values * 1024).astype(np.int16)
    quantized[np.arange(len(values)), values.argmax(axis=1)] += 1024 - quantized.sum(axis=1)
    for name, column in zip(names, quantized.T):
        group = mesh.vertex_groups.new(name=name)
        indices = np.flatnonzero(column)
        indices = indices[np.argsort(column[indices], kind="stable")]
        breaks = np.flatnonzero(np.diff(column[indices])) + 1
        for equal in np.split(indices, breaks):
            if len(equal):
                group.add(equal.tolist(), float(column[equal[0]]) / 1024., "REPLACE")
    mesh.parent = rig
    deform = mesh.modifiers.new("Terrier armature skinning", 'ARMATURE')
    deform.object = rig
    deform.use_deform_preserve_volume = True


def _coat_colors(vertices):
    x, y, z = vertices.T
    marbling = .55 * np.sin(43 * x - 18 * y + 27 * z) + .23 * np.sin(87 * z + 42 * y)
    saddle = 1. - _ease(.79, 1.12, ((x + .173) / .153) ** 2 + ((z - .433) / .126) ** 2 + .17 * marbling)
    bib = _ease(.118, .250, x) * (1. - _ease(.319, .474, z))
    cap = _ease(.219, .275, x) * _ease(.471, .529, z)
    blaze = (1. - _ease(.012, .027, np.abs(y))) * _ease(.314, .371, x)
    ear = _ease(.062, .096, np.abs(y)) * _ease(.423, .469, z) * (1. - _ease(.284, .318, x))
    muzzle = _ease(.354, .448, x) * (1. - _ease(.484, .519, z))
    paws = 1. - _ease(.061, .181, z + .009 * marbling)
    patch = np.maximum(saddle, cap * (1. - blaze))
    base = np.asarray([.514, .308, .146])
    chest = np.asarray([.738, .641, .489])
    russet = np.asarray([.250, .091, .033])
    dark_ear = np.asarray([.146, .049, .023])
    mixed = base + patch[:, None] * (russet - base)
    mixed += np.maximum.reduce([bib, paws, muzzle, blaze * cap])[:, None] * (chest - mixed)
    mixed += ear[:, None] * (dark_ear - mixed)
    lip_height = .451 + .021 * _ease(.386, .459, x)
    lip = ((1. - _ease(.0012, .0047, np.abs(z - lip_height)))
           * _ease(.371, .403, x) * (1. - _ease(.451, .469, x))
           * _ease(.018, .031, np.abs(y)))
    mixed += .82 * lip[:, None] * (np.asarray([.041, .020, .014]) - mixed)
    mixed *= (.962 + .066 * marbling + .047 * np.sin(113 * x + 134 * z - 67 * y))[:, None]
    return mixed


def _color_attribute(obj, colors):
    attribute = obj.data.color_attributes.new(name="Coat", type="FLOAT_COLOR", domain="POINT")
    attribute.data.foreach_set("color", np.column_stack((colors, np.ones(len(colors)))).astype("f").ravel())


def _groom(bpy, surface, rig, coat, points, weights, bone_names, colors, seed):
    """Short tapered fibres swept with the coat and deformed with their roots."""
    from .blender_scene import mesh_object

    rng = np.random.default_rng([seed, 613])
    surface.data.calc_loop_triangles()
    indices = np.asarray([tuple(t.vertices) for t in surface.data.loop_triangles])
    triangles = points[indices]
    centers = triangles.mean(axis=1)
    # Leave the sole, muzzle margins and eyelids free from silhouette hairs.
    mask = (centers[:, 2] > .022) & (centers[:, 0] < .432)
    eyes = (centers[:, 0] > .322) & (centers[:, 2] > .488) & (np.abs(centers[:, 1]) > .049)
    indices, triangles = indices[mask & ~eyes], triangles[mask & ~eyes]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    choice = rng.choice(len(indices), 22000, p=areas / areas.sum())
    normals = cross[choice] / areas[choice, None]
    uv = rng.random((len(choice), 2))
    uv = np.where(uv.sum(axis=1, keepdims=True) > 1, 1 - uv, uv)
    factors = np.column_stack((1. - uv.sum(axis=1), uv))
    triangle_ids = indices[choice]
    roots = np.einsum("ij,ijk->ik", factors, points[triangle_ids])
    fur_weights = np.einsum("ij,ijk->ik", factors, weights[triangle_ids])
    fur_colors = np.einsum("ij,ijk->ik", factors, colors[triangle_ids])
    flow = np.broadcast_to([-.88, 0., -.37], normals.shape).copy()
    flow[roots[:, 0] > .340] = (.87, 0., -.19)
    flow[roots[:, 2] < .215] = (-.07, 0., -.98)
    flow -= (flow * normals).sum(axis=1, keepdims=True) * normals
    direction = flow + .40 * normals + rng.normal(0., .13, flow.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    length = rng.uniform(.0017, .0054, (len(roots), 1))
    length *= (1. + .25 * (roots[:, 0] < -.289))[:, None]
    width = np.cross(direction, normals)
    width *= rng.uniform(.000085, .000185, (len(roots), 1))
    tip = roots + length * direction
    mid = roots + .47 * length * direction + .07 * length * normals
    vertices = np.stack((roots - width, roots + width, mid + .62 * width, mid - .62 * width, tip), axis=1)
    starts = np.arange(len(roots))[:, None] * 5
    faces = (starts + [0, 1, 2, 3]).tolist() + (starts + [3, 2, 4]).tolist()
    fibers = mesh_object(bpy, "Terrier short coat fibres", vertices.reshape(-1, 3).tolist(),
                         faces, coat)
    fur_colors *= rng.uniform(.71, 1.15, (len(roots), 1))
    _color_attribute(fibers, np.repeat(fur_colors, 5, axis=0))
    _bind(fibers, rig, dict(zip(bone_names, np.repeat(fur_weights, 5, axis=0).T)))


def _rig(bpy, bones):
    from mathutils import Vector

    data = bpy.data.armatures.new("Terrier skeleton")
    rig = bpy.data.objects.new("Terrier · analytic IK armature", data)
    bpy.context.scene.collection.objects.link(rig)
    rig.show_in_front = True
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    for name, (start, end, parent) in bones.items():
        bone = data.edit_bones.new(name)
        bone.head, bone.tail = start, end
        if parent:
            bone.parent = data.edit_bones[parent]
        bone.align_roll(Vector((0., 1., 0.)))
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig


def _details(bpy, rig):
    from mathutils import Matrix

    from .blender_life import _sphere
    from .blender_scene import material

    nose = material(bpy, "Terrier wet nose", (.031, .018, .014), .38,
                    surface_noise=(265., .24, .002))
    nostril = material(bpy, "Terrier nostrils", (.018, .010, .008), .58)
    socket = material(bpy, "Terrier thin dark eyelids", (.081, .033, .019), .67)
    iris = material(bpy, "Terrier brown iris", (.116, .055, .022), .19)
    pupil = material(bpy, "Terrier dark cornea", (.018, .010, .007), .13)
    spheres = [
        ("Leathery nose", (.456, 0., .478), (.019, .032, .016), nose),
    ]
    for sign in (-1, 1):
        spheres.extend([
            ("Nostril", (.469, .015 * sign, .481), (.0045, .0075, .003), nostril),
            ("Eye rim", (.328, .065 * sign, .510), (.0146, .0073, .0112), socket),
            ("Amber iris", (.332, .068 * sign, .510), (.0117, .0066, .0090), iris),
            ("Reflective pupil", (.335, .072 * sign, .5105), (.0077, .0047, .0075), pupil),
        ])
    for name, location, scale, mat in spheres:
        detail = _sphere(bpy, name, None, location, scale, mat)
        detail.data.transform(Matrix.Translation(location) @ Matrix.Diagonal((*scale, 1.)))
        detail.matrix_basis = Matrix.Identity(4)
        _bind(detail, rig, {"head": np.ones(len(detail.data.vertices))})


@dataclass
class SkinnedDog:
    rig: object
    surface: object
    bones: dict

    def apply(self, gait):
        from mathutils import Matrix, Vector

        def turn(origin, axis, angle):
            return (Matrix.Translation(origin) @ Matrix.Rotation(angle, 4, axis)
                    @ Matrix.Translation(-Vector(origin)))

        root = gait.root
        self.rig.location = (*root.xy, gait.body_z)
        self.rig.rotation_euler[2] = root.heading
        phase = root.distance * (2 * np.pi / STRIDE_M)
        moving = root.moving
        rest = self.rig.data.bones
        transforms = {
            "pelvis": turn(self.bones["pelvis"][0], 'X', .030 * moving * np.sin(phase)),
            "spine": turn(self.bones["spine"][0], 'Z', .023 * moving * np.sin(phase - .42)),
            "chest": turn(self.bones["chest"][0], 'Y', -.020 * moving * np.cos(2 * phase)),
            "neck": turn(self.bones["neck"][0], 'Y', .025 * moving * np.sin(2 * phase - .35)),
            "head": (turn(self.bones["head"][0], 'Z', .110 + .025 * moving * np.sin(phase))
                     @ turn(self.bones["head"][0], 'Y', .022 + .038 * moving * np.sin(2 * phase))),
            "tail": turn(self.bones["tail"][0], 'Z', (.09 + .16 * moving) * np.sin(phase - .50)),
            "tail_tip": turn(self.bones["tail_tip"][0], 'Z', .12 * moving * np.sin(phase - 1.1)),
            "ear-1": turn(self.bones["ear-1"][0], 'Y', .11 * moving * np.sin(2 * phase)),
            "ear+1": turn(self.bones["ear+1"][0], 'Y', .11 * moving * np.sin(2 * phase + .40)),
        }
        targets, motion = {}, {}
        for name, transform in transforms.items():
            _, _, parent = self.bones[name]
            motion[name] = motion.get(parent, Matrix.Identity(4)) @ transform
            targets[name] = motion[name] @ rest[name].matrix_local
        for name, leg in gait.legs.items():
            joints = (leg.hip, leg.knee, leg.ankle)
            for suffix, start, end in zip((".upper", ".lower"), joints, joints[1:]):
                full_name = name + suffix
                direction = (Vector(end) - Vector(start)).normalized()
                z_axis = Vector((0., 1., 0.))
                z_axis = (z_axis - z_axis.dot(direction) * direction).normalized()
                x_axis = direction.cross(z_axis)
                matrix = Matrix((x_axis, direction, z_axis)).transposed().to_4x4()
                matrix.translation = start
                targets[full_name] = matrix
            paw = name + ".paw"
            targets[paw] = rest[paw].matrix_local.copy()
            targets[paw].translation = leg.ankle

        # Convert the solved global pose into bone-local channels. This avoids
        # a pose update between each parent and child and keeps bones unscaled.
        for name, target in targets.items():
            _, _, parent = self.bones[name]
            bone = self.rig.pose.bones[name]
            if parent:
                bone.matrix_basis = rest[name].convert_local_to_pose(
                    target, rest[name].matrix_local, parent_matrix=targets[parent],
                    parent_matrix_local=rest[parent].matrix_local, invert=True)
            else:
                bone.matrix_basis = rest[name].matrix_local.inverted() @ target


def make_dog(bpy, seed):
    from .blender_scene import material

    bones = _bone_specs()
    rig = _rig(bpy, bones)
    coat = material(bpy, "Terrier fawn coat", (.514, .308, .146), .72,
                    surface_noise=(460., .25, .0018), grain=(.56, 1.4, 1.5))
    nodes, links = coat.node_tree.nodes, coat.node_tree.links
    attribute = nodes.new("ShaderNodeVertexColor")
    attribute.layer_name = "Coat"
    principled = nodes.get("Principled BSDF")
    links.new(attribute.outputs["Color"], principled.inputs["Base Color"])
    principled.inputs["Sheen Weight"].default_value = .18
    surface, points = _build_surface(bpy, bones, coat)
    weights = _skin_weights(points, bones)
    colors = _coat_colors(points)
    _color_attribute(surface, colors)
    _bind(surface, rig, dict(zip(bones, weights.T)))
    _groom(bpy, surface, rig, coat, points, weights, tuple(bones), colors, seed)
    _details(bpy, rig)
    return SkinnedDog(rig, surface, bones)
