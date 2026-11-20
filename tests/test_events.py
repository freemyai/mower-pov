import json

import numpy as np

from mower_pov.cutting import compute_cut_state
from mower_pov.events import build_events, events_to_json
from test_cutting import fixture, trajectory


def test_event_json_has_no_nonfinite_values():
    cfg, terrain, roots = fixture()
    traj = trajectory([(0, 0), (1, 0)], [0., 1.], [False, False])
    log = build_events(cfg, traj, compute_cut_state(cfg, terrain, traj, roots), roots)
    assert isinstance(json.loads(events_to_json(log), parse_constant=lambda _: np.testing.fail()), list)


def test_cut_load_reduces_on_repeat_pass():
    cfg, terrain, roots = fixture()
    traj = trajectory([(0, 0), (1, 0), (0, 0)], [0., 1., 2.])
    log = build_events(cfg, traj, compute_cut_state(cfg, terrain, traj, roots), roots)
    loads = [event for event in log.events if event.kind == "cut_load"]
    assert loads
    assert all(event.time_s < 1 for event in loads)
    assert sum(dict(event.payload)["roots"] for event in loads) == 1


def test_load_summary_never_includes_roots_cut_after_event():
    cfg, terrain, roots = fixture(xy=((.51, 0), (.53, 0)), h=(.10, .11))
    traj = trajectory([(0, 0), (1, 0)], [0., 1.])
    state = compute_cut_state(cfg, terrain, traj, roots)
    load, = [ev for ev in build_events(cfg, traj, state, roots).events if ev.kind == "cut_load"]
    assert load.time_s >= max(state.root_first_cut_s)
