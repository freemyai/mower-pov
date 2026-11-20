"""Shared absolute-time mower and cutting-load events."""

import json
from bisect import bisect_right
from dataclasses import asdict

import numpy as np

from .contracts import BladeRoots, CutState, Event, EventLog, RunConfig, Trajectory


def build_events(config: RunConfig, trajectory: Trajectory, state: CutState,
                 roots: BladeRoots) -> EventLog:
    t = trajectory.time_s
    pos = trajectory.position_m
    events = []

    def emit(time, kind, intensity, **payload):
        position = tuple(float(np.interp(time, t, axis)) for axis in pos.T)
        events.append(Event(float(time), kind, position, float(intensity), tuple(sorted(payload.items()))))

    for signal, on, off in (
        (trajectory.blade_enabled, "blade_on", "blade_off"),
        (np.abs(trajectory.v_mps) + np.abs(trajectory.omega_radps) > .005, "drive_start", "drive_stop"),
    ):
        for i in np.flatnonzero(np.diff(signal, prepend=False)):
            emit(t[i], on if signal[i] else off, float(signal[i]))

    # Estimate vertical acceleration from completed trajectory intervals.
    dt = np.diff(t)
    vertical_speed = np.diff(pos[:, 2]) / dt
    vertical_accel = np.abs(np.diff(vertical_speed) / ((dt[:-1] + dt[1:]) / 2))
    above = vertical_accel > 1.
    for i in np.flatnonzero(above & np.diff(above, prepend=False)):
        emit(t[i + 2], "terrain_impact", vertical_accel[i], acceleration_mps2=float(vertical_accel[i]))

    removal = np.maximum(roots.h0_m - state.cut_height_m, 0)
    affected = np.isfinite(state.root_first_cut_s) & (removal > 0)
    if affected.any():
        times = state.root_first_cut_s[affected]
        buckets = np.floor((times - t[0]) * 10).astype(int)
        volumes = np.bincount(buckets, weights=removal[affected]) / config.grass.roots_per_m2
        counts = np.bincount(buckets)
        # Timestamp each load summary at its last contributing cut.
        latest = np.full(len(counts), -np.inf)
        np.maximum.at(latest, buckets, times)
        for i in np.flatnonzero(counts):
            emit(latest[i], "cut_load", volumes[i] * 10, roots=int(counts[i]),
                 removed_proxy_m3=float(volumes[i]))

    return EventLog(tuple(sorted(events, key=lambda event: event.time_s)))


def events_through(log: EventLog, time_s: float) -> tuple[Event, ...]:
    return log.events[:bisect_right(log.events, time_s, key=lambda event: event.time_s)]


def events_to_json(log: EventLog) -> str:
    events = [{**asdict(ev), "payload": dict(ev.payload)} for ev in log.events]
    return json.dumps(events, allow_nan=False, sort_keys=True, separators=(",", ":"))


def events_from_json(content: str) -> EventLog:
    data = json.loads(content)
    events = tuple(Event(ev["time_s"], ev["kind"], tuple(ev["position_m"]), ev["intensity"],
                         tuple(sorted(ev["payload"].items()))) for ev in data)
    events_to_json(EventLog(events))
    return EventLog(events)
