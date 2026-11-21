"""从作业缓存生成路线、覆盖、连续净距及机载短草复核证据。"""

import argparse
import json
from pathlib import Path

import numpy as np

from mower_pov.cache import load_bundle
from mower_pov.camera import camera_pose
from mower_pov.cutting import sample_grass_heights
from mower_pov.events import events_through
from mower_pov.motion import sample_pose, validate_swept_body
from mower_pov.obstacles import distance_to_segment
from mower_pov.terrain import grid_to_xy, sample_height


DETAIL_MODES = {"REPLAN", "BYPASS", "REJOIN", "TURN", "PARK"}


def segment_distances(start, end, other_start, other_end):
    """Distance between each moving-center chord and one obstacle segment."""
    other_start, other_end = np.asarray(other_start), np.asarray(other_end)
    # If the chords do not cross, the closest pair includes at least one endpoint.
    closest = np.minimum.reduce([
        distance_to_segment(start, other_start, other_end),
        distance_to_segment(end, other_start, other_end),
        distance_to_segment(other_start, start, end),
        distance_to_segment(other_end, start, end),
    ])
    moving, fixed = end - start, other_end - other_start
    offset = other_start - start
    determinant = moving[:, 0] * fixed[1] - moving[:, 1] * fixed[0]
    divisor = np.where(determinant != 0, determinant, 1.)
    along_moving = (offset[:, 0] * fixed[1] - offset[:, 1] * fixed[0]) / divisor
    along_fixed = (offset[:, 0] * moving[:, 1] - offset[:, 1] * moving[:, 0]) / divisor
    crosses = ((determinant != 0) & (along_moving >= 0) & (along_moving <= 1)
               & (along_fixed >= 0) & (along_fixed <= 1))
    return np.where(crosses, 0., closest)


def coverage_metrics(terrain, first_cut, rows, radius):
    cell_rc = np.argwhere(terrain.mowable_mask)
    cell_xy = grid_to_xy(terrain, cell_rc)
    actual = np.isfinite(first_cut) & terrain.mowable_mask
    nominal = np.zeros(terrain.mowable_mask.shape, dtype=bool)
    row_coverage = []
    for row, (start, end) in enumerate(rows, 1):
        inside = cell_rc[distance_to_segment(cell_xy, start, end) <= radius]
        rr, cc = inside.T
        nominal[rr, cc] = True
        row_coverage.append({"row": row, "nominal_fraction_cut": float(actual[rr, cc].mean()) if len(rr) else None})
    cell_area = terrain.cell_size_m ** 2
    intersection = int(np.count_nonzero(actual & nominal))
    nominal_count = int(np.count_nonzero(nominal))
    return {
        "cell_size_m": terrain.cell_size_m,
        "actual_m2": int(actual.sum()) * cell_area,
        "nominal_m2": nominal_count * cell_area,
        "actual_in_nominal_m2": intersection * cell_area,
        "actual_outside_nominal_m2": int(np.count_nonzero(actual & ~nominal)) * cell_area,
        "nominal_uncut_m2": (nominal_count - intersection) * cell_area,
        "nominal_fraction_cut": intersection / nominal_count if nominal_count else None,
        "row_coverage": row_coverage,
    }, nominal


def visible_cut_masks(first_cut, visible, turn_started, viewed_at):
    """Separate roots already cut at the start of a turn from those still uncut."""
    return visible & (first_cut <= turn_started), visible & (first_cut > viewed_at)


def optical_coordinates(config, camera_points):
    """Return image coordinates normalized to the half-width and half-height."""
    x, y, depth = camera_points.T
    tangent_h = np.tan(np.deg2rad(config.camera.hfov_deg) / 2)
    tangent_v = tangent_h * config.render.height / config.render.width
    denominator = np.maximum(depth, .01)
    return x / (tangent_h * denominator), y / (tangent_v * denominator), depth


def _fps_frame(time, fps, count):
    return min(count - 1, max(0, round(time * fps)))


def _mode_intervals(bundle):
    tr, story = bundle.trajectory, bundle.story
    cfg = bundle.resolved_config
    changes = (story.mode[1:] != story.mode[:-1]) | (story.obstacle_name[1:] != story.obstacle_name[:-1])
    edges = np.r_[0, np.flatnonzero(changes) + 1, len(tr.time_s)]
    intervals = []
    for first, stop in zip(edges[:-1], edges[1:]):
        last = min(stop, len(tr.time_s) - 1)
        start, end = tr.time_s[[first, last]]
        interval = {
            "mode": str(story.mode[first]),
            "obstacle": str(story.obstacle_name[first]) or None,
            "start_s": float(start), "end_s": float(end),
            "start_frame": _fps_frame(start, cfg.render.fps, cfg.frame_count),
            "end_frame": _fps_frame(end, cfg.render.fps, cfg.frame_count),
            "heading_change_deg": float(np.rad2deg(tr.unwrapped_yaw[last] - tr.unwrapped_yaw[first])),
            "min_speed_mps": float(np.min(tr.v_mps[first:stop])),
            "max_speed_mps": float(np.max(tr.v_mps[first:stop])),
            "blade_command_on_fraction": float(np.mean(story.blade_command[first:stop])),
            "effective_cut_on_fraction": float(np.mean(tr.blade_enabled[first:stop])),
            "rotor_fraction_start": float(story.rotor_fraction[first]),
            "rotor_fraction_end": float(story.rotor_fraction[last]),
        }
        intervals.append(interval)
    return intervals


def _swept_clearances(bundle, obstacles, rows, rectangle_clearance):
    cfg, tr = bundle.resolved_config, bundle.trajectory
    xy = tr.position_m[:, :2]
    yaw = tr.unwrapped_yaw
    corner_radius = .5 * np.hypot(cfg.mower.length_m, cfg.mower.width_m)
    pose_sampling_pad = .5 * (np.linalg.norm(np.diff(xy, axis=0), axis=1)
                              + corner_radius * np.abs(np.diff(yaw)))
    nominal_poses = []
    for start, end in rows:
        direction = np.subtract(end, start)
        samples = max(2, int(np.ceil(np.linalg.norm(direction) / .005)) + 1)
        nominal_poses.append((np.linspace(start, end, samples), np.arctan2(direction[1], direction[0])))
    report = []
    for obstacle in obstacles:
        points = np.asarray(obstacle.points)
        indices = np.arange(max(1, len(points) - 1))
        center_to_obstacle = np.minimum.reduce([
            segment_distances(xy[:-1], xy[1:], points[i], points[min(i + 1, len(points) - 1)])
            for i in indices
        ])
        circular_gap = center_to_obstacle - corner_radius - obstacle.radius_m
        samples = np.concatenate([
            rectangle_clearance(xy[i:i + 1024], yaw[i:i + 1024], cfg, obstacle)
            for i in range(0, len(xy), 1024)
        ])
        bounded_sweep = np.minimum(samples[:-1], samples[1:]) - pose_sampling_pad
        idx = int(np.argmin(bounded_sweep))
        seconds = float(tr.time_s[idx])
        nominal_gaps = [float(rectangle_clearance(centers, heading, cfg, obstacle).min())
                        for centers, heading in nominal_poses]
        report.append({
            "obstacle": obstacle.name,
            "footprint_xy_m": points.tolist(),
            "obstacle_radius_m": obstacle.radius_m,
            "rectangle_swept_lower_bound_m": float(bounded_sweep[idx]),
            "rectangle_sample_min_m": float(np.min(samples)),
            "sampling_pad_for_minimum_m": float(pose_sampling_pad[idx]),
            "circumscribed_circle_swept_m": float(np.min(circular_gap)),
            "nominal_row_sample_gap_m": nominal_gaps,
            "nominal_rows_overlapping_obstacle": [i + 1 for i, gap in enumerate(nominal_gaps) if gap < 0.],
            "swept_min_interval_start_s": seconds,
            "swept_min_interval_end_s": float(tr.time_s[idx + 1]),
            "swept_min_frame": _fps_frame(seconds, cfg.render.fps, cfg.frame_count),
        })
    return report


def _return_views(bundle, intervals):
    cfg, tr = bundle.resolved_config, bundle.trajectory
    roots, state = bundle.roots, bundle.cut_state
    fps = cfg.render.fps
    near_m, far_m = .35, 2.4
    turn_intervals = [interval for interval in intervals if interval["mode"] == "TURN"]
    report = []
    for i, turn in enumerate(turn_intervals):
        first = int(np.ceil((turn["end_s"] + .25) * fps))
        next_turn = turn_intervals[i + 1]["start_s"] if i + 1 < len(turn_intervals) else tr.time_s[-1]
        last = min(cfg.frame_count - 1, int(np.floor(min(next_turn, turn["end_s"] + 5) * fps)))
        best = None
        best_score = 0
        for frame in range(first, last + 1, max(1, round(fps / 4))):
            seconds = frame / fps
            view = camera_pose(cfg, sample_pose(tr, seconds))
            # Restrict the point projection to a small neighborhood around the camera.
            offset = roots.root_xyz_m - view[:3, 3]
            local = np.flatnonzero(np.all(np.abs(offset) <= 2 * far_m, axis=1))
            x, y, z = optical_coordinates(cfg, offset[local] @ view[:3, :3])
            visible = (z >= near_m) & (z <= far_m) & (np.abs(x) <= 1.) & (np.abs(y) <= 1.)
            short, tall = visible_cut_masks(state.root_first_cut_s[local], visible, turn["start_s"], seconds)
            score = int(short.sum()) if tall.any() else 0
            if score > best_score:
                best_score = score
                best = (frame, seconds, local[short], local[tall], int(visible.sum()))
        check = {"turn_start_s": turn["start_s"], "turn_end_s": turn["end_s"], "view": None}
        if best is not None:
            frame, seconds, short_ids, tall_ids, visible_count = best
            heights = sample_grass_heights(roots, state, seconds)
            later_heights = sample_grass_heights(roots, state, tr.time_s[-1])
            check["view"] = {
                "frame": frame, "time_s": seconds,
                "visible_candidate_roots": visible_count,
                "roots_cut_before_turn": len(short_ids), "uncut_roots": len(tall_ids),
                "old_cut_percent_of_candidate_roots": 100 * len(short_ids) / visible_count,
                "old_grass_height_median_m": float(np.median(heights[short_ids])),
                "uncut_grass_height_median_m": float(np.median(heights[tall_ids])),
                "old_grass_height_unchanged_to_end": bool(np.array_equal(heights[short_ids], later_heights[short_ids])),
            }
        report.append(check)
    return report


def _obstacle_views(bundle, intervals, obstacles):
    cfg, tr = bundle.resolved_config, bundle.trajectory
    fps = cfg.render.fps
    report = []
    for obstacle in obstacles:
        xy = np.asarray(obstacle.points)
        if len(xy) > 1:
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            probes = np.linspace(0, length[-1], 128)
            xy = np.column_stack([np.interp(probes, length, axis) for axis in xy.T])
            height_m = .028
        else:
            height_m = min(obstacle.radius_m, .145)
        points = np.column_stack((xy, sample_height(bundle.terrain, xy) + height_m))
        for interval in intervals:
            if interval["mode"] != "BYPASS" or interval["obstacle"] != obstacle.name:
                continue
            best_score, best_frame = 0., None
            for frame in range(max(0, interval["start_frame"] - 4 * fps), interval["end_frame"]):
                view = camera_pose(cfg, sample_pose(tr, frame / fps))
                x, y, depth = optical_coordinates(cfg, (points - view[:3, 3]) @ view[:3, :3])
                # Prefer a nearby centered view while keeping the object in front of the lens.
                center_weight = np.maximum(0., 1. - x ** 2) * np.maximum(0., 1. - y ** 2)
                score = np.sum(center_weight ** 2 * (depth >= .40) / np.maximum(depth, .40))
                if score > best_score:
                    best_score, best_frame = score, frame
            report.append({"obstacle": obstacle.name, "bypass_start_s": interval["start_s"],
                           "candidate_frame": best_frame,
                           "candidate_time_s": None if best_frame is None else best_frame / fps})
    return report


def _replay(bundle, frames):
    restored = load_bundle(bundle.run_dir)
    cfg, fps = bundle.resolved_config, bundle.resolved_config.render.fps
    frames = sorted({0, cfg.frame_count - 1, *frames})

    def snapshot(source, absolute_s):
        sample = np.searchsorted(source.trajectory.time_s, absolute_s + 1e-10, side="right") - 1
        return (*[getattr(source.story, field)[sample].item() for field in (
            "mode", "obstacle_name", "blade_command", "rotor_fraction")],
                events_through(source.events, absolute_s), sample_pose(source.trajectory, absolute_s),
                sample_grass_heights(source.roots, source.cut_state, absolute_s))

    # Replay the loaded bundle out of order against chronologically sampled snapshots.
    expected = {frame: snapshot(bundle, frame / fps) for frame in frames}
    for frame in np.random.default_rng(17).permutation(frames):
        actual = snapshot(restored, frame / fps)
        original = expected[frame]
        if actual[:-2] != original[:-2] or any(
            not np.array_equal(a, b) for a, b in zip(actual[-2:], original[-2:])
        ):
            return {"frames": frames, "matches": False, "first_mismatched_frame": int(frame)}
    return {"frames": frames, "matches": True}


def _review_frames(cfg, intervals, clearances, return_views, obstacle_views):
    fps, count = cfg.render.fps, cfg.frame_count
    reasons = {0: ["START"], count - 1: ["END"]}

    def add(frame, label):
        reasons.setdefault(frame, []).append(label)

    for interval in intervals:
        mode = interval["mode"]
        if mode in DETAIL_MODES:
            center = .5 * (interval["start_s"] + interval["end_s"])
            label = f"{mode} {interval['obstacle']}" if interval["obstacle"] else mode
            add(_fps_frame(center, fps, count), label)
    for clearance in clearances:
        add(clearance["swept_min_frame"], f"CLOSEST {clearance['obstacle']}")
    for check in return_views:
        if check["view"] is not None:
            add(check["view"]["frame"], "OLD CUT IN CAMERA FRUSTUM")
    for view in obstacle_views:
        if view["candidate_frame"] is not None:
            add(view["candidate_frame"], f"APPROACH VIEW {view['obstacle']}")
    return [{"frame": frame, "time_s": frame / fps, "reasons": reasons[frame]} for frame in sorted(reasons)]


def _plot_course(bundle, nominal, rows, obstacles, report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Patch

    terrain, tr, story = bundle.terrain, bundle.trajectory, bundle.story
    cut = bundle.cut_state.t_first_cut_s
    colors = {"REPLAN": "#b47925", "BYPASS": "#b2422b", "REJOIN": "#328076", "TURN": "#765692",
              "MOW": "#203b56", "PARK": "#777e80"}
    duration = tr.time_s[-1]
    xy = tr.position_m[:, :2]
    x, y = terrain.origin_xy_m
    ny, nx = cut.shape
    extent = (x, x + nx * terrain.cell_size_m, y, y + ny * terrain.cell_size_m)

    figure, ax = plt.subplots(figsize=(9.15, 6.6), layout="constrained")
    figure.suptitle("Mowing course and first-cut history", fontsize=14)
    ax.set_facecolor("#faf7f1")
    # Missing portions of the geometric target remain distinguishable from the yard.
    ax.imshow(np.ma.masked_where(~nominal, nominal), extent=extent, origin="lower", interpolation="nearest",
              cmap=ListedColormap(["#eddfb7"]), label="Nominal cut footprint")
    cmap = ListedColormap(plt.colormaps["YlGn"](np.linspace(.36, .94, 256)))
    cmap.set_bad((0, 0, 0, 0))
    cut_image = ax.imshow(np.ma.masked_invalid(cut), extent=extent, origin="lower", interpolation="nearest",
                          cmap=cmap, vmin=0, vmax=duration)
    for row in rows:
        ax.plot(*np.asarray(row).T, color="#9a8e6d", lw=1., ls=(0, (2, 3)), alpha=.8, zorder=3)
    observed_modes = set()
    for interval in report["mode_intervals"]:
        mode = interval["mode"]
        i, j = np.searchsorted(tr.time_s, [interval["start_s"], interval["end_s"]])
        points = xy[i:j + 1]
        ax.plot(*points.T, color=colors.get(mode, "#42494c"), lw=2., zorder=4,
                label=mode if mode not in observed_modes else "_nolegend_")
        observed_modes.add(mode)
    for t in np.arange(0, duration, 10):
        i = np.searchsorted(tr.time_s, t)
        ax.scatter(*xy[i], color="white", edgecolor="#253848", s=27, zorder=6)
        ax.annotate(f"{t:g} s", xy[i], xytext=(2, 7), textcoords="offset points", ha="center",
                    fontsize=8, color="#243746", zorder=7,
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="#faf7f1", alpha=.85)])
    ax.scatter(*xy[-1], marker="x", color="#202728", s=35, zorder=7, label="Finish")
    for obstacle in obstacles:
        points = np.asarray(obstacle.points)
        for point in points:
            ax.add_patch(Circle(point, obstacle.radius_m, facecolor="#944929", edgecolor="none", zorder=5))
        if len(points) > 1:
            delta = np.diff(points, axis=0)
            length = np.maximum(np.linalg.norm(delta, axis=1), 1e-15)
            normal = delta[:, ::-1] * [-1, 1] / length[:, None] * obstacle.radius_m
            polygons = np.stack((points[:-1] + normal, points[:-1] - normal,
                                 points[1:] - normal, points[1:] + normal), axis=1)
            for polygon in polygons:
                ax.fill(*polygon.T, color="#944929", linewidth=0, zorder=5)
        anchor = points[np.argmax(points[:, 0])] if len(points) > 1 else points[0]
        label = obstacle.name.replace("_", " ")
        label_offset = (7, 10) if len(points) > 1 else (5, -18)
        ax.annotate(label, anchor, xytext=label_offset, textcoords="offset points", color="#763d28",
                    fontsize=9, zorder=8, bbox={"facecolor": "#faf7f1", "alpha": .92, "edgecolor": "none"},
                    arrowprops={"arrowstyle": "-", "lw": .7, "color": "#763d28"})
    short_grass_marker = "Old short grass in view"
    for check in report["post_turn_short_grass"]:
        view = check["view"]
        if view is not None:
            point = sample_pose(tr, view["time_s"])[:2, 3]
            ax.scatter(*point, marker="D", s=25, color="#faf7f1", edgecolor="#4d3c66", zorder=6,
                       label=short_grass_marker)
            short_grass_marker = "_nolegend_"
    focus = np.concatenate([xy, *(np.asarray(obstacle.points) for obstacle in obstacles)])
    low, high = focus.min(axis=0) - [.52, .50], focus.max(axis=0) + [.65, .46]
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal", xlabel="World X (m)",
           ylabel="World Y (m)")
    ax.set_axisbelow(True)
    ax.grid(color="#3c493e", alpha=.10)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color="#eddfb7", label="Nominal target"))
    labels.append("Nominal target")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.14), ncols=4, frameon=False)
    figure.colorbar(cut_image, ax=ax, location="right", label="First cut (s)", shrink=.71)
    figure.savefig(output / "course-map.png", dpi=165)
    plt.close(figure)

    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(11.6, 7.7), layout="constrained",
                                height_ratios=[.38, 1, 1, 1])
    for interval in report["mode_intervals"]:
        mode = interval["mode"]
        start, end = interval["start_s"], interval["end_s"]
        color = colors.get(mode, "#42494c")
        axes[0].axvspan(start, end, color=color, alpha=.88, lw=0)
        if end - start > duration * .028:
            axes[0].text(.5 * (start + end), .50, mode, fontsize=7 if mode != "TURN" else 8,
                         ha="center", va="center", color="white", clip_on=True)
        if mode in DETAIL_MODES:
            for ax in axes[1:]:
                ax.axvspan(start, end, color=color, alpha=.08, lw=0)
    axes[0].set(yticks=[], title="Motion, blade state, and accumulated cut area")
    axes[0].spines[:].set_visible(False)
    axes[1].plot(tr.time_s, tr.v_mps, color="#234363", lw=1.6)
    axes[1].set_ylabel("Speed (m/s)")
    axes[2].step(tr.time_s, story.blade_command, where="post", color="#a44738", lw=1.2, label="Blade command")
    axes[2].plot(tr.time_s, story.rotor_fraction, color="#27496e", lw=1.6, label="Rotor fraction")
    axes[2].step(tr.time_s, .07 * tr.blade_enabled, where="post", color="#4a8b63", lw=1.6,
                 label="Effective cut (bottom line)")
    axes[2].set_ylabel("Normalized")
    axes[2].legend(loc="upper right", fontsize=8, ncols=3)
    first_cut = np.sort(cut[np.isfinite(cut)])
    area = np.searchsorted(first_cut, tr.time_s, side="right") * terrain.cell_size_m ** 2
    axes[3].plot(tr.time_s, area, color="#2b7055", lw=1.6)
    axes[3].set(ylabel="Cut area (m²)", xlabel="Absolute job time (s)", xlim=(0, duration))
    for ax in axes[1:]:
        ax.grid(axis="y", alpha=.2)
    figure.savefig(output / "course-timeline.png", dpi=165)
    plt.close(figure)


def measure_course(bundle, output, obstacles, rows, rectangle_clearance):
    if bundle.resolved_config.scenario != "mowing-course" or bundle.story is None:
        raise ValueError("测量需要 mowing-course 作业缓存")
    cfg, tr, story = bundle.resolved_config, bundle.trajectory, bundle.story
    layout = bundle.manifest["identity"].get("course_layout", {})
    coverage, nominal = coverage_metrics(bundle.terrain, bundle.cut_state.t_first_cut_s, rows,
                                         cfg.mower.blade_radius_m)
    intervals = _mode_intervals(bundle)
    clearances = _swept_clearances(bundle, obstacles, rows, rectangle_clearance)
    return_views = _return_views(bundle, intervals)
    obstacle_views = _obstacle_views(bundle, intervals, obstacles)
    review_frames = _review_frames(cfg, intervals, clearances, return_views, obstacle_views)
    report = {
        "run_id": bundle.run_id,
        "scenario": cfg.scenario,
        "duration_s": float(tr.time_s[-1]),
        "frame_count": cfg.frame_count,
        "fps": cfg.render.fps,
        "mode_intervals": intervals,
        "turn_count": sum(interval["mode"] == "TURN" for interval in intervals),
        "distance_m": float(np.linalg.norm(np.diff(tr.position_m[:, :2], axis=0), axis=1).sum()),
        "observed_motion_limits": {
            "max_speed_mps": float(np.max(np.abs(tr.v_mps))),
            "max_acceleration_mps2": float(np.max(np.abs(np.diff(tr.v_mps) / np.diff(tr.time_s)))),
            "max_yaw_speed_radps": float(np.max(np.abs(tr.omega_radps))),
            "max_yaw_acceleration_radps2": float(np.max(np.abs(np.diff(tr.omega_radps) / np.diff(tr.time_s)))),
        },
        "nominal_rows_xy_m": rows,
        "scheduled_rows": [dict(row) for row in layout.get("scheduled_rows", ())],
        "coverage": coverage,
        "cut_roots": int(np.isfinite(bundle.cut_state.root_first_cut_s).sum()),
        "blade_command_off_s": float(np.count_nonzero(~story.blade_command[:-1]) / cfg.world.step_hz),
        "effective_cut_off_s": float(np.count_nonzero(~tr.blade_enabled[:-1]) / cfg.world.step_hz),
        "yard_swept_violations": [{"time_s": violation.time_s, "reason": violation.reason} for violation in
                                  validate_swept_body(cfg, bundle.terrain, tr)],
        "obstacle_clearances": clearances,
        "obstacle_approach_views": obstacle_views,
        "post_turn_short_grass": return_views,
        "short_grass_camera_depth_range_m": [.35, 2.4],
        "recommended_review_frames": review_frames,
        "random_access_replay": _replay(bundle, [entry["frame"] for entry in review_frames]),
        "measurement_notes_zh": {
            "coverage": "分母为无避障的名义直行刀盘扫掠与原始可割网格的交集。绕行在名义范围外割到的面积单列。",
            "obstacle_clearances": "矩形代理在每对相邻姿态间扣除半步最大平移与旋转位移，得到保守下界。"
                                   "同时对相同底盘的外接圆做独立连续线段扫掠。",
            "post_turn_short_grass": "计数使用绝对帧相机视锥内的草根坐标。遮挡和视觉可读性需结合所列机载帧复核。",
            "obstacle_approach_views": "按固定相机投影中道具代理点的大小与距画面中心的位置选择候选帧。"
                                       "道具及草的相互遮挡需看片确认。",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _plot_course(bundle, nominal, rows, obstacles, report, output)
    measurement = output / "course-measurements.json"
    measurement.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    return measurement


if __name__ == "__main__":
    from mower_pov.course_layout import CourseObstacle, obstacle_clearance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    bundle = load_bundle(args.work_bundle)
    layout = bundle.manifest["identity"].get("course_layout")
    if layout is None:
        parser.error("缓存中缺少作业路线及障碍代理布局")
    x0, x1 = layout["x_limits_m"]
    rows = [((x0, y), (x1, y)) for y in layout["row_y_m"]]
    obstacles = [CourseObstacle(item["name"], item["points_m"], item["radius_m"]) for item in layout["obstacles"]]
    print(measure_course(bundle, args.output, obstacles, rows, obstacle_clearance))
