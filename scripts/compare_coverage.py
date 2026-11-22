"""Compare two mowing-course bundles on the baseline's original mowing target."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from measure_course import coverage_metrics
from mower_pov.cache import load_bundle
from mower_pov.cutting import _first_fraction, _segments
from mower_pov.terrain import grid_to_xy, xy_to_grid


def cutter_visits(config, terrain, trajectory):
    """Count separate physical visits during which the blade was effective."""
    radius = config.mower.blade_radius_m
    mowable = terrain.mowable_mask
    visits = np.zeros(mowable.shape, dtype=np.int32)
    contact_through = np.full(mowable.shape, -np.inf)
    visit_counted = np.zeros_like(mowable)
    times = trajectory.time_s
    enabled = trajectory.blade_enabled
    # Keep tracking contact even when the blade stops during a continuous visit.
    physical = replace(trajectory, blade_enabled=np.ones_like(enabled))
    for t0, t1, start, end, _ in _segments(config, physical):
        low, high = xy_to_grid(terrain, [np.minimum(start, end) - radius,
                                        np.maximum(start, end) + radius])
        low, high = np.maximum(low, 0), np.minimum(high + 1, mowable.shape)
        if np.any(high <= low):
            continue
        r0, c0 = low
        r1, c1 = high
        rr, cc = np.nonzero(mowable[r0:r1, c0:c1])
        rr, cc = rr + r0, cc + c0
        points = grid_to_xy(terrain, np.column_stack((rr, cc)))
        entries = _first_fraction(points, start, end - start, radius)
        hits = entries <= 1.
        rr, cc = rr[hits], cc[hits]
        was_counted = (contact_through[rr, cc] == t0) & visit_counted[rr, cc]
        start_i, end_i = np.searchsorted(times, [t0, t1], side="right") - 1
        hit_times = t0 + (t1 - t0) * entries[hits]
        cutting = enabled[start_i] & ((hit_times < t1) | enabled[end_i])
        visits[rr, cc] += cutting & ~was_counted

        # A disc's intersection with a chord is contiguous. Only cells
        # still under the disc at the endpoint can continue into the next chord.
        offset = points[hits] - end
        at_end = np.einsum("ij,ij->i", offset, offset) <= radius ** 2
        contact_through[rr[at_end], cc[at_end]] = t1
        visit_counted[rr[at_end], cc[at_end]] = (was_counted | cutting)[at_end]
    return visits


def travel_bins(route_length, first_cut_distances, bin_m):
    """Group first cuts by cumulative chassis travel, including the final endpoint."""
    full_edges = np.arange(bin_m, route_length, bin_m)
    edges = np.r_[0., full_edges[full_edges < route_length - 1e-10], route_length]
    counts = np.histogram(first_cut_distances, bins=edges)[0]
    return np.diff(edges), counts


def long_no_cut_gaps(trajectory, drive_intervals, route_m, first_cut_times):
    """Find blade-on driving intervals of at least 1.25 s without fresh cuts."""
    blade_drive = drive_intervals & trajectory.blade_enabled[:-1]
    edges = np.r_[0, np.flatnonzero(np.diff(blade_drive)) + 1, len(blade_drive)]
    times = trajectory.time_s
    xy = trajectory.position_m[:, :2]
    sorted_cuts = np.sort(first_cut_times)
    gaps = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if not blade_drive[start]:
            continue
        lo, hi = times[[start, stop]]
        prefix, suffix = np.searchsorted(sorted_cuts, [lo, hi])
        cuts = np.r_[lo, sorted_cuts[prefix:suffix], hi]
        for i in np.flatnonzero(np.diff(cuts) >= 1.25):
            gap = cuts[i:i + 2]
            gaps.append({
                "start_s": float(gap[0]),
                "end_s": float(gap[1]),
                "distance_m": float(np.diff(np.interp(gap, times, route_m))[0]),
                "endpoints_xy_m": np.column_stack([np.interp(gap, times, axis) for axis in xy.T]).tolist(),
            })
    return gaps


def measure_run(bundle, terrain, target_rows, blade_radius):
    tr, story = bundle.trajectory, bundle.story
    first_cut = bundle.cut_state.t_first_cut_s
    active_mask = np.isfinite(first_cut) & terrain.mowable_mask
    coverage, nominal = coverage_metrics(terrain, first_cut, target_rows, blade_radius)
    visits = cutter_visits(bundle.resolved_config, terrain, tr)
    mismatches = np.count_nonzero((visits > 0) != active_mask)
    if mismatches:
        raise ValueError(f"{bundle.run_id} cutter sweep differs from the cut cache at {mismatches} cells")

    step_m = np.linalg.norm(np.diff(tr.position_m[:, :2], axis=0), axis=1)
    route_m = np.r_[0., np.cumsum(step_m)]
    step_s = np.diff(tr.time_s)
    translating = step_m > 1e-9
    driving = translating | (np.abs(np.diff(tr.unwrapped_yaw)) > 1e-9)
    cell_area = terrain.cell_size_m ** 2
    new_cut_times = first_cut[active_mask]
    new_cut_distance = np.interp(new_cut_times, tr.time_s, route_m)
    bin_lengths, new_cells = travel_bins(route_m[-1], new_cut_distance, blade_radius)
    _, new_target_cells = travel_bins(route_m[-1], new_cut_distance[nominal[active_mask]], blade_radius)
    low_yield_area = .10 * (2 * blade_radius) * bin_lengths
    repeated = visits > 1
    visit_counts = np.bincount(visits.ravel())
    step_modes = story.mode[:-1]
    cut_modes = story.mode[np.searchsorted(tr.time_s, new_cut_times, side="right") - 1]
    by_mode = {
        str(mode): {
            "distance_m": float(step_m[step_modes == mode].sum()),
            "drive_active_s": float(step_s[(step_modes == mode) & driving].sum()),
            "new_area_m2": float(np.count_nonzero(cut_modes == mode) * cell_area),
        } for mode in np.unique(story.mode)
    }

    rows = bundle.manifest["identity"]["course_layout"]["scheduled_rows"]
    in_row = np.zeros_like(active_mask)
    row_stats = []
    for row in rows:
        t0, t1 = row["t_start_s"], row["t_end_s"]
        cut = (first_cut >= t0) & (first_cut < t1) & active_mask
        in_row |= cut
        row_distance = np.diff(np.interp([t0, t1], tr.time_s, route_m))[0]
        row_stats.append({
            "row": row["row"],
            "t_start_s": t0,
            "t_end_s": t1,
            "distance_m": float(row_distance),
            "new_area_m2": float(np.count_nonzero(cut) * cell_area),
            "new_target_area_m2": float(np.count_nonzero(cut & nominal) * cell_area),
        })

    return {
        "run_id": bundle.run_id,
        "bundle": str(bundle.run_dir),
        "duration_s": float(tr.time_s[-1]),
        "coverage": coverage,
        "repeated_area_m2": float(np.count_nonzero(repeated) * cell_area),
        "repeated_target_area_m2": float(np.count_nonzero(repeated & nominal) * cell_area),
        "repeat_area_fraction_of_unique": float(np.count_nonzero(repeated) / np.count_nonzero(active_mask))
        if np.any(active_mask) else None,
        "extra_visit_area_m2": float(np.maximum(visits - 1, 0).sum() * cell_area),
        "area_m2_by_exact_visits": {i: float(count * cell_area)
                                    for i, count in enumerate(visit_counts[1:], 1)},
        "distance_m": float(route_m[-1]),
        "drive_active_s": float(step_s[driving].sum()),
        "translating_s": float(step_s[translating].sum()),
        "blade_active_s": float(step_s[tr.blade_enabled[:-1]].sum()),
        "blade_on_distance_m": float(step_m[tr.blade_enabled[:-1]].sum()),
        "dead_travel_m": float(bin_lengths[new_cells == 0].sum()),
        "low_or_no_yield_travel_m": float(bin_lengths[new_cells * cell_area <= low_yield_area].sum()),
        "target_dead_travel_m": float(bin_lengths[new_target_cells == 0].sum()),
        "target_low_or_no_yield_travel_m": float(bin_lengths[new_target_cells * cell_area <= low_yield_area].sum()),
        "blade_on_no_cut_gaps": long_no_cut_gaps(tr, driving, route_m, new_cut_times),
        "new_area_outside_rows_m2": float(np.count_nonzero(active_mask & ~in_row) * cell_area),
        "new_target_area_outside_rows_m2": float(np.count_nonzero(nominal & active_mask & ~in_row) * cell_area),
        "by_mode": by_mode,
        "rows": row_stats,
    }, nominal, visits, np.sort(new_cut_distance)


def plot_comparison(output, runs, target, obstacles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Patch

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), layout="constrained")
    colors = ["#f5f5f0", "#efc466", "#2e8968", "#618caf", "#b76f99", "#783161"]
    labels = ["Missed target", "Cut target", "Cut outside target", "Two passes", "Three or more passes"]
    handles = [Patch(facecolor=c, label=label) for c, label in zip(colors[1:], labels)]
    paths = np.concatenate([bundle.trajectory.position_m[:, :2] for bundle, *_ in runs])
    terrain = runs[0][0].terrain
    target_points = grid_to_xy(terrain, np.argwhere(target))
    focus = np.concatenate((paths, target_points))
    low, high = focus.min(axis=0) - .28, focus.max(axis=0) + .28

    for ax, label, (bundle, metrics, visits, _) in zip(axes[0], ("Baseline", "Candidate"), runs):
        terrain = bundle.terrain
        width, depth = np.array(visits.shape[::-1]) * terrain.cell_size_m
        x, y = terrain.origin_xy_m
        cut = visits > 0
        layers = np.zeros_like(visits, dtype=np.uint8)
        layers[target], layers[target & cut], layers[cut & ~target] = 1, 2, 3
        layers[visits == 2], layers[visits > 2] = 4, 5
        ax.imshow(layers, origin="lower", extent=(x, x + width, y, y + depth),
                  interpolation="nearest", cmap=ListedColormap(colors), vmin=0, vmax=5)
        for obstacle in obstacles:
            points = np.asarray(obstacle["points_m"])
            if len(points) == 1:
                ax.add_patch(Circle(points[0], obstacle["radius_m"], facecolor="#303044", alpha=.70))
            else:
                ax.plot(*points.T, color="#303044", lw=3.6, solid_capstyle="round", alpha=.70)
        ax.plot(*bundle.trajectory.position_m[:, :2].T, color="#202326", lw=.8, alpha=.58)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="Yard X (m)", ylabel="Yard Y (m)",
               title=(f"{label}  ·  {metrics['coverage']['nominal_fraction_cut']:.1%} of target"
                      f"  ·  {metrics['extra_visit_area_m2']:.2f} m² extra sweep"))
    axes[0, 0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0, -.40),
                      ncols=3, frameon=False, fontsize=8)

    cumulative, row_chart = axes[1]
    row_labels = sorted({row["row"] for _, metrics, *_ in runs for row in metrics["rows"]})
    xs = np.arange(len(row_labels) + 1)
    for index, (label, (_, metrics, _, distances)) in enumerate(zip(("Baseline", "Candidate"), runs)):
        color = ["#677487", "#247a5e"][index]
        area = metrics["coverage"]["cell_size_m"] ** 2
        cumulative.plot(np.r_[0., distances, metrics["distance_m"]],
                        np.r_[0., np.arange(1, len(distances) + 1) * area, len(distances) * area],
                        color=color, lw=2, label=label)
        row_areas = {row["row"]: row["new_area_m2"] for row in metrics["rows"]}
        bar_areas = [row_areas.get(row, 0.) for row in row_labels]
        bar_areas.append(metrics["new_area_outside_rows_m2"])
        row_chart.bar(xs + (.35 * index - .175), bar_areas, .35, color=color, label=label)
    cumulative.set(xlabel="Route distance (m)", ylabel="Unique cut area (m²)", xlim=(0, None), ylim=(0, None))
    row_chart.set(xticks=xs, xticklabels=[str(row) for row in row_labels] + ["Turns / park"],
                  xlabel="Scheduled row", ylabel="New cut area (m²)")
    for ax in axes[1]:
        ax.legend(loc="best", frameon=False)
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
    fig.savefig(output / "coverage-comparison.png", dpi=155)
    plt.close(fig)


def compare(baseline, candidate, output):
    terrain = baseline.terrain
    base_mower = baseline.resolved_config.mower
    for bundle in (baseline, candidate):
        if bundle.resolved_config.scenario != "mowing-course" or bundle.story is None:
            raise ValueError("comparisons require mowing-course work bundles")
        mower = bundle.resolved_config.mower
        if (bundle.terrain.cell_size_m != terrain.cell_size_m
                or not np.array_equal(bundle.terrain.origin_xy_m, terrain.origin_xy_m)
                or not np.array_equal(bundle.terrain.mowable_mask, terrain.mowable_mask)
                or mower.blade_radius_m != base_mower.blade_radius_m
                or mower.blade_xy_m != base_mower.blade_xy_m):
            raise ValueError("comparisons require the same mowable grid and cutter")
    layout = baseline.manifest["identity"]["course_layout"]
    x0, x1 = layout["x_limits_m"]
    rows = [((x0, y), (x1, y)) for y in layout["row_y_m"]]
    runs = []
    for bundle in (baseline, candidate):
        metrics, nominal, visits, distances = measure_run(bundle, terrain, rows, base_mower.blade_radius_m)
        runs.append((bundle, metrics, visits, distances))
    report = {
        "target_rows_xy_m": rows,
        "cutter_radius_m": base_mower.blade_radius_m,
        "notes": {
            "target": "Original baseline rows swept without detours, intersected with baseline mowable cells.",
            "repeat": "A repeated pass requires separate physical cutter-disc contact intervals at the same cell. "
                      "A visit counts if the effective blade reaches the cell. Continuous contact counts once.",
            "extra_visit_area": "Each visit beyond the first contributes one cell area.",
            "drive_active_time": "Trajectory intervals with chassis translation or heading change.",
            "travel_bins": "Consecutive chassis-distance bins one cutter radius wide, starting at the route origin.",
            "dead_travel": "Total bin length with zero first-cut area. Low-yield includes dead travel and bins "
                           "that deliver at most 10% of a fresh full-width cut per metre. Target-only figures "
                           "use first-cut area inside the original target.",
            "blade_on_no_cut_gaps": "Continuous blade-on driving spans of at least 1.25 s without a first cut.",
            "row_areas": "First-cut times in each scheduled row interval. Areas outside rows include turns and park.",
        },
        "baseline": runs[0][1],
        "candidate": runs[1][1],
    }
    output.mkdir(parents=True, exist_ok=True)
    plot_comparison(output, runs, nominal, layout["obstacles"])
    path = output / "coverage-comparison.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(compare(load_bundle(args.baseline), load_bundle(args.candidate), args.output))
