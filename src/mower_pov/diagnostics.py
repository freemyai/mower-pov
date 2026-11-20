"""Generate spatial diagnostic figures and numerical replay evidence."""

import json
from pathlib import Path

import numpy as np

from .cache import load_bundle, save_bundle
from .camera import camera_mount, camera_pose
from .cutting import sample_grass_heights
from .events import events_through
from .motion import sample_pose, validate_swept_body


def _image(ax, data, extent, **kwargs):
    return ax.imshow(data, extent=extent, origin="lower", interpolation="nearest", **kwargs)


def _yard(ax, terrain, extent):
    from matplotlib.patches import Rectangle

    _image(ax, terrain.height_m, extent, cmap="terrain", alpha=.52)
    x0, y0, x1, y1 = terrain.obstacle_xyxy_m
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, hatch="//",
                           facecolor="#a77b53", edgecolor="#553c32", label="Flower bed"))


def _label(ax):
    ax.set(xlabel="World X (m)", ylabel="World Y (m)", aspect="equal")


def _focus(state, trajectory, terrain):
    row, col = np.nonzero(np.isfinite(state.t_first_cut_s))
    if not len(row):
        xy = trajectory.position_m[:, :2]
        return (*np.min(xy, axis=0), *np.max(xy, axis=0))
    low = terrain.origin_xy_m + np.array([col.min(), row.min()]) * terrain.cell_size_m
    high = terrain.origin_xy_m + (np.array([col.max(), row.max()]) + 1) * terrain.cell_size_m
    return (*low, *high)


def _plots(bundle, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terrain, tr, roots, state = bundle.terrain, bundle.trajectory, bundle.roots, bundle.cut_state
    x, y = terrain.origin_xy_m
    ny, nx = terrain.height_m.shape
    extent = (x, x + nx * terrain.cell_size_m, y, y + ny * terrain.cell_size_m)
    x0, y0, x1, y1 = _focus(state, tr, terrain)
    margin = .35
    roi = dict(xlim=(x0 - margin, x1 + margin), ylim=(y0 - margin, y1 + margin))

    def save(fig, name):
        fig.savefig(output_dir / f"{name}.png", dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    image = _image(axes[0], terrain.height_m * 100, extent, cmap="terrain")
    fig.colorbar(image, ax=axes[0], label="Elevation (cm)", shrink=.7)
    axes[0].set_title("Deterministic terrain")
    roots_xy = roots.root_xyz_m[::max(1, len(roots.root_id) // 14000), :2]
    _image(axes[1], terrain.mowable_mask, extent, cmap="Greys_r", vmin=-.25, vmax=1.1)
    axes[1].scatter(*roots_xy.T, s=.3, c="#227052", alpha=.5)
    axes[1].set_title("Grass roots (subsample) and mowable area")
    for ax in axes:
        _label(ax)
    save(fig, "terrain")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained",
                             height_ratios=[3, 1])
    xy = tr.position_m[:, :2]
    for ax in axes:
        _yard(ax, terrain, extent)
        ax.plot(*xy.T, c="#253f74", lw=1.4, label="Chassis path")
        idx = np.searchsorted(tr.time_s, np.arange(0, tr.time_s[-1] + 1e-9, tr.time_s[-1] / 8))
        idx = idx.clip(max=len(xy) - 1)
        ax.scatter(*xy[idx].T, c=tr.time_s[idx], cmap="plasma", s=22, zorder=3)
        _label(ax)
    axes[0].legend(loc="lower left")
    axes[0].set_title("Chassis path and flower-bed boundary")
    axes[1].set(**roi)
    axes[1].set_title("Outbound and return rows")
    save(fig, "trajectory")

    fig, ax = plt.subplots(figsize=(11, 3.1), layout="constrained")
    _yard(ax, terrain, extent)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad((0, 0, 0, 0))
    image = _image(ax, np.ma.masked_invalid(state.t_first_cut_s), extent, cmap=cmap, vmin=0,
                   vmax=bundle.resolved_config.world.duration_s)
    ax.plot(*xy.T, lw=.7, ls=":", c="#272727", label="Chassis path")
    fig.colorbar(image, ax=ax, label="First cut (s)")
    ax.set(**roi, title="First cutter contact at each 1 cm grid center"
           if terrain.cell_size_m == .01 else "First cutter contact at each grid center")
    _label(ax)
    save(fig, "first-cut")

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), layout="constrained")
    cfg = bundle.resolved_config
    for ax, time in zip(axes, [tr.time_s[-1] * .55, tr.time_s[-1]]):
        _yard(ax, terrain, extent)
        x, y = roots.root_xyz_m[:, :2].T
        visible = ((x >= roi["xlim"][0]) & (x <= roi["xlim"][1]) &
                   (y >= roi["ylim"][0]) & (y <= roi["ylim"][1]))
        heights = sample_grass_heights(roots, state, time)
        image = ax.scatter(x[visible], y[visible], c=heights[visible] * 100, s=3,
                           linewidths=0, cmap="cividis", vmin=min(cfg.grass.min_height_m, cfg.grass.cut_height_m) * 100,
                           vmax=cfg.grass.max_height_m * 100)
        _label(ax)
        ax.set(**roi, title=f"Root lengths at t = {time:g} s")
    fig.colorbar(image, ax=axes, label="Grass length (cm)")
    save(fig, "final-height")


def write_diagnostics(bundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    persisted = load_bundle(save_bundle(bundle))
    _plots(persisted, output_dir)
    cfg, tr, state = bundle.resolved_config, bundle.trajectory, bundle.cut_state
    end = tr.time_s[-1]
    times = sorted({0., .125 * end, .5 * end, .75 * end, .875 * end, end})
    expected = {t: (sample_pose(tr, t), sample_grass_heights(bundle.roots, state, t),
                    events_through(bundle.events, t)) for t in times}
    replay = True
    for t in np.random.default_rng(17).permutation(times):
        pose, grass, events = expected[t]
        replay &= (np.array_equal(pose, sample_pose(persisted.trajectory, t))
                   and np.array_equal(grass, sample_grass_heights(persisted.roots, persisted.cut_state, t))
                   and events == events_through(persisted.events, t))
    errors = []
    for t in np.random.default_rng(19).uniform(0, end, 20):
        body = sample_pose(tr, t)
        errors.append(np.max(np.abs(np.linalg.inv(body) @ camera_pose(cfg, body) - camera_mount(cfg))))
    cut = np.isfinite(state.t_first_cut_s)
    report = {
        "run_id": bundle.run_id, "duration_s": float(end), "logical_video_frames": cfg.frame_count,
        "motion_samples": len(tr.time_s), "root_count": len(bundle.roots.root_id),
        "cut_roots": int(np.isfinite(state.root_first_cut_s).sum()),
        "covered_grid_area_m2": float(cut.sum() * bundle.terrain.cell_size_m ** 2),
        "swept_body_violations": len(validate_swept_body(cfg, bundle.terrain, tr)),
        "camera_mount_max_abs_error": float(max(errors)), "camera_mount_samples": len(errors),
        "pose_grass_event_replay_times_s": times, "random_access_matches_sequential": replay,
        "frame_900_checked": bool(900 < cfg.frame_count and 900 / cfg.render.fps in times),
        "event_count": len(bundle.events.events),
    }
    (output_dir / "state-checks.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
