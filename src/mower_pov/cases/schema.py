"""Parse strict JSON and resolve the v1 layout rules before running a case."""

import json
import math
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator

from ..contracts import CameraConfig
from .contracts import AssertionSpec, ObstacleSpec, ResolvedCase, RobotSpec, ScenarioSpec, WorldSpec

_SCHEMA = Draft202012Validator(json.loads(
    files(__package__).joinpath("scenario-v1.schema.json").read_text(encoding="utf-8")
))
_DEFAULT_CAMERA = CameraConfig((.18, 0., .22), 100., 12.)


def _fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _invalid_float(text):
    raise ValueError(f"non-finite JSON value {text!r}")


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        _invalid_float(text)
    return value


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_fields,
                      parse_constant=_invalid_float, parse_float=_finite_float)


def _obstacles(records):
    obstacles = []
    for item in records:
        points = tuple(tuple(map(float, point)) for point in item["points_xy_m"])
        if (len(points) == 1) != (item["type"] == "disc"):
            raise ValueError(f"{item['id']}: a disc requires one point; a polyline requires at least two")
        if any(a == b for a, b in zip(points, points[1:])):
            raise ValueError(f"{item['id']}: repeated consecutive obstacle points")
        obstacles.append(ObstacleSpec(item["id"], item["type"], points, float(item["radius_m"])))
    if len({item.id for item in obstacles}) != len(obstacles):
        raise ValueError("obstacle IDs must be unique")
    return tuple(obstacles)


def _world(data):
    x0, x1 = map(float, data["x_limits_m"])
    rows = tuple(map(float, data["row_y_m"]))
    width, depth = float(data["width_m"]), float(data["depth_m"])
    if not (-width / 2 <= x0 < x1 <= width / 2) or x1 - x0 < .4:
        raise ValueError("x_limits_m must span at least 0.4 m inside the yard")
    pitch = rows[0] - rows[1] if len(rows) > 1 else None
    if any(abs(y) > depth / 2 for y in rows) or (
        pitch is not None and (pitch <= 0 or any(not math.isclose(a - b, pitch)
                                                 for a, b in zip(rows, rows[1:])))
    ):
        raise ValueError("row_y_m must be equally spaced in decreasing order within the yard")
    bed = tuple(map(float, data["flower_bed_xyxy_m"])) if "flower_bed_xyxy_m" in data else None
    if bed is not None and not (bed[0] < bed[2] and bed[1] < bed[3]):
        raise ValueError("flower_bed_xyxy_m must describe a positive-area rectangle")
    return WorldSpec(width, depth, int(data["step_hz"]), float(data["cell_m"]), (x0, x1), rows,
                     _obstacles(data["obstacles"]), bed, float(data.get("clearance_m", .035)))


def _assertions(items):
    result = []
    for item in items:
        name = item["name"]
        parameters = set(item) - {"name"}
        required = {"minimum_m"} if name == "continuous_body_clearance" else set()
        if name == "minimum_target_coverage":
            required = {"minimum_fraction"}
        if parameters != required:
            raise ValueError(f"{name}: expected parameters {sorted(required)}")
        result.append(AssertionSpec(name, **{key: float(item[key]) for key in parameters}))
    if len({item.name for item in result}) != len(result):
        raise ValueError("assertion names must be unique")
    return tuple(result)


def load_spec(path: Path) -> ScenarioSpec:
    data = read_json(Path(path))
    error = next(_SCHEMA.iter_errors(data), None)
    if error is not None:
        where = ".".join(map(str, error.absolute_path)) or str(path)
        raise ValueError(f"{where}: {error.message}")
    robot = RobotSpec(**{key: float(value) for key, value in data["robot"].items()})
    if robot.track_m > robot.width_m:
        raise ValueError("robot.track_m must fit inside robot.width_m")
    if 2 * robot.blade_radius_m > min(robot.width_m, robot.length_m):
        raise ValueError("robot blade disc must fit within the chassis")
    camera = _DEFAULT_CAMERA
    if "camera" in data:
        c = data["camera"]
        camera = CameraConfig(tuple(map(float, c["mount_xyz_m"])), float(c["hfov_deg"]), float(c["pitch_deg"]))
    u = data["units"]
    return ScenarioSpec(ResolvedCase(
        int(data["version"]), data["id"], int(data["seed"]), data["mode"], data["fidelity"],
        (u["length"], u["time"], u["angle"]), data["frame"], robot, _world(data["world"]),
        (), data["observation"]["provider"], (), int(data["termination"]["max_sim_time_s"]),
        data["expected"]["outcome"], _assertions(data["assertions"]),
        tuple(sorted(data.get("assets", []))), camera,
    ))
