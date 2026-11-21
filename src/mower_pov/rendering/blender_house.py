"""Textured house and terrace for the matched POV environment comparison."""

from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


CATALOG = Path(__file__).resolve().parents[3] / "assets" / "catalog.lock.json"


def assets_for_house(catalog=CATALOG):
    catalog = Path(catalog)
    with catalog.open("rb") as handle:
        digest = file_digest(handle, "sha256").hexdigest()
    document = json.loads(catalog.read_text())
    if document["schema"] != 1:
        raise ValueError("Unsupported yard asset catalog version")
    assets = {}
    for entry in document["used"]:
        if entry["license"] != "CC0-1.0" or not entry["redistributable"]:
            raise ValueError(f"House texture lacks redistribution permission: {entry['id']}")
        for channel in entry["channels"].values():
            path = catalog.parent / channel["path"]
            with path.open("rb") as handle:
                sha256 = file_digest(handle, "sha256").hexdigest()
            if sha256 != channel["sha256"]:
                raise ValueError(f"House texture digest mismatch: {path.name}")
            channel["path"] = path
        assets[entry["id"]] = entry
    return assets, digest


def _surface_material(bpy, asset, strength):
    from .blender_scene import material

    mat = material(bpy, asset["name"], (.38, .32, .24))
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    coords = nodes.new("ShaderNodeTexCoord")
    for purpose, data in asset["channels"].items():
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(data["path"]), check_existing=True)
        tex.image.colorspace_settings.name = data["colorspace"]
        links.new(coords.outputs["UV"], tex.inputs["Vector"])
        if purpose == "normal":
            convert = nodes.new("ShaderNodeNormalMap")
            convert.inputs["Strength"].default_value = strength
            links.new(tex.outputs["Color"], convert.inputs["Color"])
            links.new(convert.outputs["Normal"], bsdf.inputs["Normal"])
        else:
            name = "Base Color" if purpose == "base_color" else "Roughness"
            links.new(tex.outputs["Color"], bsdf.inputs[name])
    return mat


def _map_box(obj, axes, tile):
    mesh = obj.data
    coordinates = np.asarray([tuple(vertex.co) for vertex in mesh.vertices]) + tuple(obj.location)
    ids = np.asarray([loop.vertex_index for loop in mesh.loops])
    uv = coordinates[ids][:, axes] / tile
    mesh.uv_layers.active.uv.foreach_set("vector", uv.astype("f").ravel())


def make_house(bpy, bundle, expected_digest=None):
    from .blender_scene import box, material, mesh_object, tube

    textures, digest = assets_for_house()
    if expected_digest is not None and digest != expected_digest:
        raise ValueError("House asset catalog changed after render preflight")
    brick_asset = textures["red_brick"]
    slab_asset = textures["concrete_floor_worn_001"]
    brick = _surface_material(bpy, brick_asset, .68)
    slab = _surface_material(bpy, slab_asset, .48)
    cream = material(bpy, "Painted window surrounds", (.56, .52, .43), .47)
    frames = material(bpy, "Dark window frame and gutters", (.057, .061, .058), .37, .21)
    glass = material(bpy, "Blue-gray porch glazing", (.114, .145, .158), .19, .17)
    linen = material(bpy, "Linen behind side windows", (.33, .325, .268), .57)
    sill = material(bpy, "Limestone thresholds", (.38, .342, .274), .68)
    slate = material(bpy, "Matte slate roof", (.109, .107, .093), .79,
                     surface_noise=(34, .40, .007), grain=(1., .65, 1.7))
    edge_y = bundle.resolved_config.world.depth_m * .5
    half_width = bundle.resolved_config.world.width_m * .5
    front_y = edge_y + 1.85
    top, left, right = 2.77, -4.77, 5.53

    def block(name, x0, x1, y, z0, z1, thickness, mat=brick, bevel=0.):
        obj = box(bpy, name, ((x0 + x1) / 2, y, (z0 + z1) / 2),
                  (x1 - x0, thickness, z1 - z0), mat, bevel)
        if mat == brick:
            _map_box(obj, (0, 2), brick_asset["tile_m"])
        return obj

    terrace = block("Grounded terrace slab", left - .13, right + .13, edge_y + .95,
                    -.12, .095, 1.87, slab, .023)
    _map_box(terrace, (0, 1), slab_asset["tile_m"])
    block("Limestone terrace step", left, right, edge_y + .069,
                 -.038, .048, .134, sill, .008)
    # Ground and privacy returns close the gaps alongside the terrace.
    ground = material(bpy, "Soil adjoining the terrace", (.094, .096, .046), .94)
    box(bpy, "Ground behind lawn edge", (0, edge_y + 2.35, -.095),
        (2 * half_width + .5, 4.75, .16), ground)
    wood = material(bpy, "Terrace side screen", (.187, .133, .089), .81, surface_noise=(14, .19, .004))
    for x in (-half_width + .046, half_width - .046):
        for y in np.arange(edge_y + .014, front_y + .42, .13):
            box(bpy, "Privacy screen slat", (x, y, .861), (.038, .127, 1.72), wood)
        for z in (.49, 1.18):
            box(bpy, "Privacy screen bearer", (x, (edge_y + front_y + .38) / 2, z),
                (.078, front_y - edge_y + .38, .067), wood)

    # Openings are modeled as actual gaps between masonry panels.
    openings = ((-3.52, -1.82, .95, 2.12, "Window"),
                (.08, 2.24, .118, 2.27, "Sliding patio door"),
                (3.04, 4.70, .94, 2.12, "Window"))
    next_x = left
    for x0, x1, z0, z1, name in openings:
        block("Masonry pier", next_x, x0, front_y, .095, top, .32)
        block("Masonry below opening", x0, x1, front_y, .095, z0, .32)
        block("Lintel courses", x0, x1, front_y, z1, top, .32)
        next_x = x1
        block(f"{name} backing", x0, x1, front_y + .18, z0, z1, .040, linen)
        block(f"{name} glass", x0 + .04, x1 - .04, front_y - .037, z0 + .04,
              z1 - .04, .015, glass)
        for x in (x0 + .02, (x0 + x1) / 2, x1 - .02):
            block(f"{name} upright", x - .024, x + .024, front_y - .078,
                  z0, z1, .092, cream, .006)
        for z in (z0 + .019, z1 - .019):
            block(f"{name} horizontal frame", x0, x1, front_y - .081, z - .024,
                  z + .024, .095, cream, .006)
        block(f"{name} sill", x0 - .045, x1 + .045, front_y - .11, z0 - .051,
              z0, .24, sill, .007)
    block("End masonry pier", next_x, right, front_y, .095, top, .32)
    for x in (left + .14, right - .14):
        wall = box(bpy, "House side return", (x, front_y + 1.45, 1.416), (.28, 3., 2.70), brick)
        _map_box(wall, (1, 2), brick_asset["tile_m"])
    roof = [(left - .28, front_y + y, z) for y, z in ((-.45, 2.72), (1.35, 3.68), (3.14, 2.72))]
    roof += [(right + .28, y, z) for _, y, z in roof]
    mesh_object(bpy, "Pitched roof", roof, [(0, 3, 4, 1), (1, 4, 5, 2)], slate)
    for x, quad in ((left, (0, 1, 2)), (right, (3, 4, 5))):
        wall = mesh_object(bpy, "Masonry gable", [(x, roof[i][1], roof[i][2]) for i in quad],
                           [(0, 1, 2)], brick)
        wall.data.uv_layers.new()
        _map_box(wall, (1, 2), brick_asset["tile_m"])
    block("Eaves fascia", left - .27, right + .27, front_y - .45, 2.65, 2.79, .105, cream)
    tube(bpy, "Round rain gutter", (left - .27, front_y - .515, 2.697),
         (right + .27, front_y - .515, 2.677), .045, .045, frames, 16)
    tube(bpy, "Downspout", (right - .095, front_y - .18, .105),
         (right - .095, front_y - .18, 2.697), .028, .028, frames, 16)
