import json

import pytest


@pytest.fixture
def scenario():
    return {
        "version": 1,
        "id": "S00-contract",
        "seed": 42,
        "mode": "offline_plan_replay",
        "fidelity": "kinematic-geometric",
        "units": {"length": "m", "time": "s", "angle": "rad"},
        "frame": "right-handed-z-up",
        "robot": {
            "length_m": 0.50, "width_m": 0.42, "track_m": 0.32,
            "blade_radius_m": 0.14, "max_speed_mps": 0.30, "max_accel_mps2": 0.35,
            "max_omega_radps": 0.80, "max_alpha_radps2": 1.20,
        },
        "world": {
            "width_m": 5.0, "depth_m": 4.0, "step_hz": 240, "cell_m": .01,
            "x_limits_m": [-1.5, 1.5], "row_y_m": [.50, .25],
            "obstacles": [
                {"id": "hose", "type": "polyline_sweep",
                 "points_xy_m": [[-.05, -.1], [.05, .61]], "radius_m": .025}
            ],
        },
        "actors": [],
        "faults": [],
        "observation": {"provider": "known_static_geometry"},
        "termination": {"max_sim_time_s": 75},
        "expected": {"outcome": "COMPLETED"},
        "assertions": [{"name": "continuous_body_clearance", "minimum_m": .035}],
    }


@pytest.fixture
def write_scenario(tmp_path):
    def write(data, name="scenario.json"):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return path
    return write
