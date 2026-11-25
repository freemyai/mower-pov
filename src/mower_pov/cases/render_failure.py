"""Produce an absolute-time onboard window from checked static experiment evidence."""

from dataclasses import asdict, replace
from math import ceil, floor, isfinite
from pathlib import Path

from .evidence import hash_file, load_episode, write_json


def failure_window(first_violation_s, duration_s, pad_s=2.):
    if not all(isfinite(x) for x in (first_violation_s, duration_s, pad_s)) or (
        duration_s <= 0 or not 0. <= first_violation_s <= duration_s or pad_s <= 0
    ):
        raise ValueError("failure time is outside the episode or padding is invalid")
    return max(0., first_violation_s - pad_s), min(duration_s, first_violation_s + pad_s)


def window_frames(window, fps, frame_count):
    first, last = window
    if not 0 <= first < last or fps <= 0 or frame_count < 1:
        raise ValueError("an absolute nonempty render window is required")
    start = min(frame_count - 1, floor(first * fps))
    stop = min(frame_count, max(start + 1, ceil(last * fps)))
    return tuple(range(start, stop))


def _bundle(case, trace, output):
    from ..cache import SCHEMA_VERSION, _array_identity, _datasets, compute_run_id, save_bundle
    from ..contracts import RunBundle

    config = replace(trace.config, scenario="static-case")
    data = _datasets(trace.terrain, trace.trajectory, trace.roots, trace.cut_state, trace.story)
    identity = {"schema_version":SCHEMA_VERSION, "config":asdict(config),
                "arrays":{name:_array_identity(values) for name, values in data.items()},
                "events_sha256":trace.events_sha256, "state_id":trace.state_id,
                "static_obstacles":[asdict(item) for item in case.world.obstacles]}
    key = compute_run_id(identity)
    manifest = {"schema_version":SCHEMA_VERSION, "run_id":key, "identity":identity, "assets":[]}
    return save_bundle(RunBundle(key, output/key, config, trace.terrain, trace.trajectory, trace.roots,
                                 trace.cut_state, trace.events, manifest, trace.story))


def _debug_frames(frames, trace, checks, source, target, fps):
    from PIL import Image, ImageDraw
    import numpy as np

    target.mkdir(exist_ok=True)
    trajectory = trace.trajectory
    times = trajectory.time_s
    yaw_rates = np.diff(trajectory.unwrapped_yaw) / np.diff(times)
    first_bad = checks["first_violation_s"]
    status = "PASS" if checks["passed"] else "FAIL"
    for frame in frames:
        name = f"frame-{frame:06d}.png"
        time = frame / fps
        i = max(0, np.searchsorted(times, time, side="right") - 1)
        v = trajectory.v_mps[i]
        yaw_rate = yaw_rates[min(i, len(yaw_rates) - 1)]
        mode = trace.story.mode[i]
        with Image.open(source/name) as png:
            rgb = png.convert("RGB")
        draw = ImageDraw.Draw(rgb)
        draw.rectangle((0, 0, rgb.width, 52), fill="#17241c")
        draw.text((10, 6), f"{status} | {checks['outcome']} | t={time:.3f}s | absolute frame {frame}",
                  fill="#ffffff", font_size=15)
        detail = f"{mode} | {v:.3f} m/s | yaw {abs(yaw_rate):.3f} rad/s"
        if first_bad is not None:
            failed = next((row["name"] for row in checks["assertions"]
                           if row["first_violation_s"] == first_bad), "check")
            detail += f" | {failed} at {first_bad:.3f}s"
        draw.text((10, 29), detail, fill="#ffe3aa", font_size=14)
        rgb.save(target/name)


def render_episode_window(run_dir, case_id, output, blender, *, pad_s=2., at_s=None,
                          preset="smoke", device="CPU"):
    from ..rendering.driver import RenderRequest, render_frames
    from ..rendering.preview import encode_preview

    case, trace, verdict = load_episode(run_dir, case_id)
    if trace.trajectory is None:
        raise ValueError("no executed trajectory is available for POV rendering")
    centre = verdict["first_violation_s"] if at_s is None else at_s
    if centre is None:
        raise ValueError("choose --at-s for a representative window without a timed failure")
    window = failure_window(centre, case.duration_s, pad_s)
    fps = trace.config.render.fps
    frames = window_frames(window, fps, trace.config.frame_count)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bundle = _bundle(case, trace, output/"state")
    raw, annotated = output/"frames", output/"debug-frames"
    primary = render_frames(RenderRequest(bundle, frames, preset, raw, Path(blender), device, png_only=True))
    if primary.failed_frames:
        raise ValueError(f"Blender failed for {len(primary.failed_frames)} frames; see {primary.worker_log}")
    _debug_frames(frames, trace, verdict, raw, annotated, fps)
    evidence = {}
    for name, source in (("onboard", raw), ("debug", annotated)):
        video = output/f"{case_id}-{name}.mp4"
        media = encode_preview([source], frames, fps, video)
        evidence[name] = {**media, "file":video.name, "sha256":hash_file(video)}
    report = {"case_id":case_id, "state_id":trace.state_id, "scenario_digest":trace.scenario_id,
              "render_bundle":bundle.name, "preset":preset, "device":device,
              "window_s":window, "first_violation_s":verdict["first_violation_s"],
              "audio":"none", "videos":evidence}
    write_json(output/"window.json", report)
    return output/"window.json"
