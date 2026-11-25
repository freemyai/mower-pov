from mower_pov.cases.resolve import resolve_case, scenario_digest
from mower_pov.cases.schema import load_spec


def test_same_seed_same_digest(scenario, write_scenario):
    spec = load_spec(write_scenario(scenario))
    first = scenario_digest(resolve_case(spec, 42))
    assert scenario_digest(resolve_case(spec, 42)) == first
    assert scenario_digest(resolve_case(spec, 43)) != first
    assert len(first) == 64


def test_local_absolute_paths_do_not_define_case_identity(scenario, write_scenario, tmp_path):
    another = tmp_path / "another"
    another.mkdir()
    first = write_scenario(scenario)
    second = write_scenario(scenario, "another/elsewhere.json")
    assert scenario_digest(resolve_case(load_spec(first), 42)) == scenario_digest(
        resolve_case(load_spec(second), 42)
    )


def test_resolved_values_are_immutable(scenario, write_scenario):
    case = resolve_case(load_spec(write_scenario(scenario)), 42)
    assert case.world.obstacles[0].points_xy_m == ((-.05, -.1), (.05, .61))
    assert case.world.step_hz == 240
