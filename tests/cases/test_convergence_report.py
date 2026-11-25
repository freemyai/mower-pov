from math import pi

from mower_pov.cases.metrics import convergence_report


def test_straight_capsule_converges_at_three_precisions(simple_case, simple_trace):
    report = convergence_report(simple_case, simple_trace, (.01, .005, .002))
    r = simple_case.robot.blade_radius_m
    span = simple_case.world.x_limits_m[1] - simple_case.world.x_limits_m[0]
    analytic = 2*r*span + pi*r*r
    assert [row["cell_m"] for row in report["levels"]] == [.01, .005, .002]
    for level in report["levels"]:
        error = abs(level["target_area_m2"] - analytic)
        assert error < (2*span + 2*pi*r) * level["cell_m"]
    assert report["status"] == "STABLE"


def test_underresolved_step_is_never_automatically_stable(simple_case, simple_trace):
    report = convergence_report(simple_case, simple_trace, (.1, .05, .01), area_tolerance_m2=1e-8)
    assert report["status"] == "UNSTABLE"
