import pytest

from mower_pov.cases.schema import load_spec


@pytest.mark.parametrize("radius", [float("nan"), float("inf"), -.01])
def test_reject_nan_and_negative_radius(scenario, write_scenario, radius):
    scenario["world"]["obstacles"][0]["radius_m"] = radius
    with pytest.raises(ValueError, match="radius_m|NaN|Infinity"):
        load_spec(write_scenario(scenario))


def test_reject_unknown_schema_version(scenario, write_scenario):
    scenario["version"] = 17
    with pytest.raises(ValueError, match="version"):
        load_spec(write_scenario(scenario))


def test_unknown_mode_is_error(scenario, write_scenario):
    scenario["mode"] = "truth_but_secret"
    with pytest.raises(ValueError, match="mode"):
        load_spec(write_scenario(scenario))


def test_reject_duplicate_keys_and_unknown_fields(scenario, write_scenario):
    path = write_scenario(scenario)
    path.write_text(path.read_text().replace('"version": 1', '"version": 1, "version": 1'))
    with pytest.raises(ValueError, match="duplicate.*version"):
        load_spec(path)
    scenario["world"]["obstacles"][0]["ignore_collision"] = True
    with pytest.raises(ValueError, match="ignore_collision"):
        load_spec(write_scenario(scenario))


def test_reject_overflow_and_bad_units(scenario, write_scenario):
    path = write_scenario(scenario)
    path.write_text(path.read_text().replace('"radius_m": 0.025', '"radius_m": 1e999'))
    with pytest.raises(ValueError, match="finite"):
        load_spec(path)
    scenario["units"]["length"] = "cm"
    with pytest.raises(ValueError, match="units"):
        load_spec(write_scenario(scenario))


def test_integral_numeric_fields_are_normalized_for_cache(scenario, write_scenario):
    scenario["version"] = 1.0
    scenario["seed"] = 42.0
    scenario["world"]["step_hz"] = 240.0
    scenario["termination"]["max_sim_time_s"] = 75.0
    case = load_spec(write_scenario(scenario)).case
    assert all(type(x) is int for x in (case.version, case.seed, case.world.step_hz, case.duration_s))
