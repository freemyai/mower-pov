"""Reusable leaf meshes instanced on persistent grass roots."""

from dataclasses import dataclass

import numpy as np

from .blender_wind import breeze, leaf_tilt
from .grass_geometry import clip_blades

CUT_BIN_M = .001


def grass_time_s(blender_frame, fps):
    return blender_frame / fps


def blade_length(h0, first_cut, absolute_s, target):
    return np.where(absolute_s >= first_cut, np.minimum(h0, target), h0)


def wind_angle(absolute_s, phase, position):
    return .018 * np.sin(5.8 * absolute_s + position) + .015 * np.sin(9.3 * absolute_s + phase)


@dataclass
class GrassReplay:
    mesh: object
    xy: np.ndarray
    h0: np.ndarray
    first_cut: np.ndarray
    target: float
    widths: np.ndarray
    yaw: np.ndarray
    phase: np.ndarray
    wheel_times: np.ndarray
    wheel_points: np.ndarray
    wheel_width: float
    original_variants: np.ndarray
    cut_variants: np.ndarray
    prototypes: tuple

    def apply(self, absolute_s):
        h = blade_length(self.h0, self.first_cut, absolute_s, self.target)
        x, y = self.xy.T
        spatial_phase = 1.8 * x + 1.3 * y
        press = np.zeros(len(x), dtype=np.float32)
        if self.wheel_times[0] <= absolute_s:
            # Wheel compression follows recent cached contact paths and relaxes over 0.75 s.
            earliest = max(self.wheel_times[0], absolute_s - .75)
            query = np.linspace(earliest, absolute_s, 19)
            path = np.stack([
                np.interp(query, self.wheel_times, axis)
                for axis in self.wheel_points.reshape(len(self.wheel_times), 4).T
            ], axis=-1).reshape(-1, 2, 2)
            points = path.reshape(-1, 2)
            pad = .05
            near = np.flatnonzero(((self.xy >= points.min(axis=0) - pad)
                                  & (self.xy <= points.max(axis=0) + pad)).all(axis=1))
            if near.size:
                distance = np.linalg.norm(self.xy[near, None] - points, axis=2)
                footprint = np.maximum(0, 1 - distance / (self.wheel_width * .73))
                decay = np.repeat(np.maximum(0, 1 - (absolute_s - query) / .75), 2)
                press[near] = (footprint * decay).max(axis=1)

        clipped = (self.cut_variants >= 0) & (absolute_s >= self.first_cut)
        visible_height = np.where(clipped, self.h0, h)
        spread = self.widths * (.67 + .33 * visible_height / self.h0)
        scale = np.column_stack((spread, spread, visible_height * 10)).astype("f")
        displayed = np.where(clipped, self.cut_variants, self.original_variants)
        self.mesh.attributes["leaf_variant"].data.foreach_set("value", displayed)
        wind = leaf_tilt(breeze(absolute_s, x, y), self.yaw)
        angles = np.column_stack((
            wind[:, 0] + wind_angle(absolute_s, self.phase, spatial_phase) + .65 * press,
            wind[:, 1] + wind_angle(absolute_s, self.phase + 1.3, .7 * spatial_phase),
            self.yaw,
        )).astype("f")
        self.mesh.attributes["leaf_scale"].data.foreach_set("vector", scale.ravel())
        self.mesh.attributes["leaf_angles"].data.foreach_set("vector", angles.ravel())
        self.mesh.update()


def _leaf_material(bpy, name, tip_tint):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*tip_tint, 1.)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = .53
    bsdf.inputs["Specular IOR Level"].default_value = .28
    color = nodes.new("ShaderNodeVertexColor")
    color.layer_name = "LeafTint"
    links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    transmission = nodes.new("ShaderNodeBsdfTranslucent")
    links.new(color.outputs["Color"], transmission.inputs["Color"])
    mix = nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = .29
    links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    links.new(transmission.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs[0], nodes.get("Material Output").inputs["Surface"])
    return mat


def _prototype(bpy, scene, variant, seed, cutoff=None):
    rng = np.random.default_rng([seed, 41, variant])
    vertices, faces, colors = [], [], []
    palette = [
        (.144, .224, .043), (.103, .190, .039), (.184, .269, .049),
        (.080, .153, .031), (.205, .286, .065), (.117, .197, .036),
    ]
    tip = np.array(palette[variant % len(palette)])
    num_leaves = 8 if variant < 6 else 4
    for i in range(num_leaves):
        azimuth = i * 2.39996 + rng.uniform(-.35, .35)
        along = np.array([np.cos(azimuth), np.sin(azimuth)])
        across = along[::-1] * [-1, 1]
        height = rng.uniform(.39, .91) * .10 if i else .10
        curl = rng.uniform(.017, .056) * (.72 if i == 0 else 1.)
        if i >= 6:
            # Short basal blades close gaps below the canopy.
            height = rng.uniform(.024, .043)
            curl = rng.uniform(.023, .040)
        width = rng.uniform(.00135, .00345) * (1.26 if variant in (1, 4, 8) else 1.)
        root = along * rng.uniform(.001, .006)
        start = len(vertices)
        profile = [0., .18, .35, .54, .71, .87, 1.] if variant < 6 else [0., .27, .54, .79, 1.]
        skew = rng.uniform(-.28, .28)
        for j, f in enumerate(profile):
            center = root + along * curl * f ** 1.9 + across * curl * skew * f ** 2
            half = width * max(.02, (1 - f ** 1.7) * (.68 + .32 * np.sin(np.pi * f)))
            leaf_tint = tip * (.46 + .68 * f)
            if variant in (2, 4) and i == 2 and f >= .87:
                leaf_tint = np.array([.29, .225, .06])
            for side in (-1, 0, 1):
                xy = center + side * half * across
                vertices.append((*xy, f * height - abs(side) * half * .32))
                colors.append((*np.clip(leaf_tint * (1.12 if side == 0 else 1.), 0, 1), 1))
            if j:
                for side in (0, 1):
                    k = start + j * 3 + side
                    faces.append((k - 3, k - 2, k + 1, k))
    name = f"Leaf tuft {variant}"
    if cutoff is not None:
        vertices, faces, colors = clip_blades(vertices, faces, colors, cutoff)
        name += f" clipped at {cutoff:.3f} m"
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    rgba = mesh.color_attributes.new(name="LeafTint", type="FLOAT_COLOR", domain="POINT")
    rgba.data.foreach_set("color", np.asarray(colors, dtype="f").ravel())
    mesh.materials.append(_leaf_material(bpy, f"Grass leaf {variant}", tip))
    obj = bpy.data.objects.new(mesh.name, mesh)
    scene.collection.objects.link(obj)
    obj.hide_render = True
    obj.hide_set(True)
    return obj


def _instance_nodes(bpy, obj, prototypes):
    tree = bpy.data.node_groups.new("Persistent grass instances", "GeometryNodeTree")
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    inputs = nodes.new("NodeGroupInput")
    outputs = nodes.new("NodeGroupOutput")
    instances = nodes.new("GeometryNodeGeometryToInstance")
    # Blender prepends new links on a multi-input socket.
    for prototype in reversed(prototypes):
        info = nodes.new("GeometryNodeObjectInfo")
        info.inputs["Object"].default_value = prototype
        info.inputs["As Instance"].default_value = False
        links.new(info.outputs["Geometry"], instances.inputs["Geometry"])
    placement = nodes.new("GeometryNodeInstanceOnPoints")
    placement.inputs["Pick Instance"].default_value = True
    links.new(inputs.outputs["Geometry"], placement.inputs["Points"])
    links.new(instances.outputs["Instances"], placement.inputs["Instance"])
    for name, kind, socket in (
        ("leaf_variant", "INT", "Instance Index"),
        ("leaf_scale", "FLOAT_VECTOR", "Scale"),
        ("leaf_angles", "FLOAT_VECTOR", "Rotation"),
    ):
        attribute = nodes.new("GeometryNodeInputNamedAttribute")
        attribute.data_type = kind
        attribute.inputs["Name"].default_value = name
        if socket == "Rotation":
            rotation = nodes.new("FunctionNodeEulerToRotation")
            links.new(attribute.outputs["Attribute"], rotation.inputs[0])
            links.new(rotation.outputs[0], placement.inputs[socket])
        else:
            links.new(attribute.outputs["Attribute"], placement.inputs[socket])
    links.new(placement.outputs["Instances"], outputs.inputs["Geometry"])
    modifier = obj.modifiers.new("Grass root instances", "NODES")
    modifier.node_group = tree


def make_grass(scene, bundle):
    import bpy

    roots, state = bundle.roots, bundle.cut_state
    xyz = np.ascontiguousarray(roots.root_xyz_m, dtype="f")
    # Distance to the union of all poses is fixed for an entire run.
    route = bundle.trajectory.position_m[::120, :2]
    near = np.zeros(len(xyz), dtype=bool)
    for xy in route:
        near |= ((xyz[:, :2] - xy) ** 2).sum(axis=1) < 2.8 ** 2
    x, y = xyz[:, :2].T
    patch = .5 + .25 * np.sin(2.3 * x + 1.7 * y) + .25 * np.cos(1.6 * x - 2.4 * y)
    mixed = ((roots.tint + .7 * patch) * 6).astype("i") % 6
    variant = np.where(near, mixed, 6 + mixed % 4).astype("i")
    mesh = bpy.data.meshes.new("Cached grass roots")
    mesh.vertices.add(len(xyz))
    mesh.vertices.foreach_set("co", xyz.ravel())
    mesh.attributes.new("root_id", "INT", "POINT").data.foreach_set("value", roots.root_id.astype("i"))
    mesh.attributes.new("leaf_variant", "INT", "POINT").data.foreach_set("value", variant)
    mesh.attributes.new("leaf_scale", "FLOAT_VECTOR", "POINT")
    mesh.attributes.new("leaf_angles", "FLOAT_VECTOR", "POINT")
    mesh.update()
    obj = bpy.data.objects.new("Grass from the work-state cache", mesh)
    scene.collection.objects.link(obj)
    seed = bundle.resolved_config.seed
    prototypes = [_prototype(bpy, scene, i, seed) for i in range(10)]
    cut_variants = np.full(len(xyz), -1, dtype="i")
    ids = np.flatnonzero(near & np.isfinite(state.root_first_cut_s) & (roots.h0_m > state.cut_height_m))
    # Quantize the clip plane in prototype space, keeping root scale unchanged.
    bins = np.floor(state.cut_height_m / (10 * roots.h0_m[ids] * CUT_BIN_M)).astype("i")
    groups, group_indices = np.unique(np.c_[variant[ids], bins], axis=0, return_inverse=True)
    for shape, bin_index in groups:
        prototypes.append(_prototype(bpy, scene, int(shape), seed, bin_index * CUT_BIN_M))
    cut_variants[ids] = group_indices + 10
    _instance_nodes(bpy, obj, prototypes)
    wheel_times = np.r_[state.wheel_contact_time_s[:, 0], state.wheel_contact_time_s[-1, 1]]
    wheel_points = np.concatenate((state.wheel_contact_xy_m[:, :, 0],
                                   state.wheel_contact_xy_m[-1:, :, 1]))
    return GrassReplay(mesh, xyz[:, :2], roots.h0_m, state.root_first_cut_s, state.cut_height_m,
                       .68 + roots.width_m * 105 + .15 * roots.bend,
                       roots.azimuth_rad, roots.phase, wheel_times, wheel_points, state.wheel_width_m,
                       variant, cut_variants, tuple(prototypes))
