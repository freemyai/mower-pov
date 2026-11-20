"""Independent garden cameras and the animated child in the play-area montage."""

from dataclasses import dataclass

import numpy as np

from ..terrain import sample_height
from .blender_life import _apply_segment, _empty, _limb, _ring_curve, _sphere
from .life_motion import TRAMPOLINE_XY
from .showcase_motion import jump, look


@dataclass
class Child:
    root: object
    upper_body: object
    head: object
    legs: list
    arms: list
    surface: float

    def apply(self, time):
        rise, compression, reach = jump(time)
        z = self.surface + rise
        self.root.location = (*TRAMPOLINE_XY, z)
        self.upper_body.location.z = -compression
        self.upper_body.rotation_euler[1] = .13 * compression / .10
        self.head.rotation_euler[2] = .074 * np.sin(2.4 * time)
        self.head.rotation_euler[1] = .055 * (1 - reach) + .03 * np.sin(3.8 * time)
        for side, upper, shin, shoe in self.legs:
            hip = (0., side * .092, .56 - compression)
            knee = (.061 + compression * .60, side * .108, .323 + reach * .023 - compression * .37)
            foot = (.023 + reach * .031, side * .105, .068 + reach * .047)
            _apply_segment(upper, hip, knee)
            _apply_segment(shin, knee, foot)
            shoe.location = foot
            shoe.rotation_euler[1] = -.25 * reach
        for side, bicep, forearm, palm in self.arms:
            shoulder = (0., side * .151, .824 - compression)
            elbow = (.038 + .036 * reach, side * (.202 + .082 * reach), .652 + .051 * reach - compression)
            wrist = (.107 + .075 * reach, side * (.206 + .112 * reach), .538 + .144 * reach - compression)
            _apply_segment(bicep, shoulder, elbow)
            _apply_segment(forearm, elbow, wrist)
            palm.location = wrist


def _hair(bpy, head, mat):
    from .blender_scene import mesh_object

    phi = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    vertices = []
    for u in np.linspace(.04, 1, 16):
        front = np.maximum(0, np.cos(phi))
        theta = u * (1.60 - .50 * front + .075 * np.sin(phi * 3 + .4))
        scalp = np.column_stack((.113 * np.sin(theta) * np.cos(phi) - .013,
                                 .096 * np.sin(theta) * np.sin(phi),
                                 .134 * np.cos(theta) + .015))
        vertices.extend(scalp.tolist())
    facets = [(k * 48 + j, (k + 1) * 48 + j, (k + 1) * 48 + (j + 1) % 48, k * 48 + (j + 1) % 48)
              for k in range(15) for j in range(48)]
    facets.append(tuple(range(47, -1, -1)))
    obj = mesh_object(bpy, "Short swept chestnut hair", vertices, facets, mat)
    obj.parent = head
    obj.data.polygons.foreach_set("use_smooth", np.ones(len(facets), dtype=bool))


def _shirt(bpy, parent, cotton, trim):
    from .blender_scene import mesh_object

    profile = np.array([
        [.600, .075, .115], [.627, .091, .130], [.658, .099, .132],
        [.713, .103, .137], [.776, .103, .147], [.821, .095, .154],
        [.854, .087, .135], [.884, .066, .090], [.900, .047, .051],
    ])
    theta = np.linspace(0., 2*np.pi, 40, endpoint=False)
    rings = [np.column_stack((depth * np.cos(theta), width * np.sin(theta),
                              z + .003 * np.sin(3 * theta + .9)))
             for z, depth, width in profile]
    vertices = np.concatenate(rings)
    faces = [(i*40+j, i*40+(j+1)%40, (i+1)*40+(j+1)%40, (i+1)*40+j)
             for i in range(len(rings)-1) for j in range(40)]
    obj = mesh_object(bpy, "Tailored cotton play shirt", vertices.tolist(), faces, cotton)
    obj.parent = parent
    obj.data.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
    _ring_curve(bpy, "Ribbed crewneck collar", [rings[-1]], .0065, trim, parent=parent, closed=True)



def _child(bpy, terrain):
    from .blender_scene import material

    skin = material(bpy, "Warm skin", (.46, .221, .133), .65,
                    surface_noise=(230, .09, .0011), translucency=.06)
    cheek = material(bpy, "Nose and lip colour", (.43, .176, .122), .63)
    hair = material(bpy, "Chestnut hair", (.084, .043, .023), .71,
                    surface_noise=(238, .28, .003), grain=(1., .65, 1.65))
    cotton = material(bpy, "Ochre cotton shirt", (.54, .247, .080), .86,
                      surface_noise=(272, .20, .0015))
    hem = material(bpy, "Warm ribbed cotton seams", (.439, .181, .059), .83)
    denim = material(bpy, "Deep blue drawstring shorts", (.048, .125, .187), .85,
                     surface_noise=(165, .26, .0025))
    canvas = material(bpy, "Pale shoe canvas and eye whites", (.68, .642, .51), .85)
    sole = material(bpy, "White rubber trim", (.81, .775, .64), .73)
    iris = material(bpy, "Dark brown iris", (.035, .019, .012), .31)
    z = float(sample_height(terrain, TRAMPOLINE_XY)) + .396
    root = _empty(bpy, "Child in the garden montage")
    root.rotation_euler[2] = -1.94
    torso = _empty(bpy, "Jumping child torso", root)
    _shirt(bpy, torso, cotton, hem)
    _sphere(bpy, "Waist and seat of shorts", torso, (-.025, 0, .563), (.096, .126, .092), denim, 36)
    _sphere(bpy, "Neck", torso, (.011, 0., .906), (.052, .051, .065), skin, 24)
    head = _empty(bpy, "Child gaze", torso, (.020, 0, 1.028))
    _sphere(bpy, "Soft oval face", head, (-.001, 0., 0.), (.095, .087, .119), skin, 48)
    _hair(bpy, head, hair)
    _sphere(bpy, "Button nose", head, (.092, 0., -.016), (.027, .019, .023), cheek)
    lip = np.array([[.085, -.034, -.052], [.096, -.017, -.057], [.097, 0., -.058],
                    [.096, .017, -.057], [.085, .034, -.052]])
    _ring_curve(bpy, "Soft smile", [lip], .0021, cheek, parent=head)
    legs, arms = [], []
    for side in (-1, 1):
        _sphere(bpy, "Ear", head, (-.013, side * .086, -.016), (.021, .013, .029), skin)
        _sphere(bpy, "Eye corner", head, (.078, side * .040, .017), (.012, .021, .010), canvas)
        _sphere(bpy, "Iris", head, (.087, side * .040, .017), (.006, .008, .008), iris)
        _sphere(bpy, "Eyebrow", head, (.076, side * .042, .039), (.004, .022, .003), hair)
        thigh = _limb(bpy, "Denim leg", root, denim, .065, .051)
        calf = _limb(bpy, "Lower leg", root, skin, .042, .025)
        shoe = _empty(bpy, "Canvas sneaker", root)
        _sphere(bpy, "Sneaker upper", shoe, (.029, 0., -.017), (.084, .046, .038), canvas)
        _sphere(bpy, "Rubber sole", shoe, (.034, 0., -.049), (.089, .048, .019), sole)
        for x in (.019, .037, .054):
            line = np.array([[x, -.021, .014], [x-.006, 0., .022], [x, .021, .014]])
            _ring_curve(bpy, "Cotton lace", [line], .0018, sole, parent=shoe)
        legs.append((side, thigh, calf, shoe))
        sleeve = _limb(bpy, "Short sleeve", root, cotton, .052, .038)
        lower = _limb(bpy, "Forearm", root, skin, .032, .021)
        hand = _sphere(bpy, "Rounded hand", root, (0., 0., 0.), (.033, .026, .036), skin)
        arms.append((side, sleeve, lower, hand))
    return Child(root, torso, head, legs, arms, z)


@dataclass
class Showcase:
    camera: object
    child: Child
    scene: object

    def apply(self, time):
        from mathutils import Vector

        origin, target, hfov, focus = look(time)
        delta = Vector(target) - Vector(origin)
        self.camera.location = origin
        self.camera.rotation_euler = delta.to_track_quat("-Z", "Y").to_euler()
        self.camera.data.lens = self.camera.data.sensor_width / (2 * np.tan(np.deg2rad(hfov) / 2))
        self.camera.data.dof.use_dof = True
        self.camera.data.dof.focus_distance = delta.length
        self.camera.data.dof.aperture_fstop = focus
        child_visible = 26 <= time < 31
        for part in self.child.root.children_recursive:
            part.hide_render = not child_visible
        self.scene.view_settings.exposure = (.58 if time < 5 else .36 if child_visible else .19)
        self.child.apply(time)


def make_showcase(bpy, replay):
    if replay.yard_life is None:
        raise ValueError("the garden montage requires a yard large enough for the play area")
    return Showcase(replay.camera, _child(bpy, replay.bundle.terrain), replay.scene)
