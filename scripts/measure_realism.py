"""Measure dog stance in world space and compose the matched yard comparison."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image, ImageDraw

from mower_pov.actors.behavior import BALL_TRACK, DOG_TRACK, HEDGEHOG_TRACK
from mower_pov.actors.gait import DOG_LEGS, evaluate_dog
from mower_pov.cache import load_bundle
from mower_pov.camera import camera_pose
from mower_pov.motion import sample_pose
from mower_pov.rendering.life_motion import (FENCE_ENDPOINTS, FENCE_Y, TOY_XY,
                                            TRAMPOLINE_RADIUS, TRAMPOLINE_XY, trot_step)


def measure(bundle, output):
    output.mkdir(parents=True, exist_ok=True)
    times = [18.0, 20.2, 21.39]
    figure, axes = plt.subplots(1, 3, figsize=(11.6, 3.65), layout="constrained")
    summary = []
    front_right = DOG_LEGS[0]
    for ax, start in zip(axes, times):
        values = np.linspace(start, start + .100, 25)
        poses = [evaluate_dog(float(t), bundle.terrain) for t in values]
        leg = [pose.legs[front_right.name] for pose in poses]
        assert all(point.planted for point in leg)
        new = np.array([point.foot_world for point in leg])
        legacy = []
        for pose in poses:
            root = pose.root
            stride, _ = trot_step(root.distance / .40)
            legacy.append(root.xy[0] - (front_right.x + stride * root.moving))
        window = {"start_s": start, "end_s": start + .100, "samples": len(values),
                  "legacy_world_x_range_mm": float(np.ptp(legacy) * 1000),
                  "new_max_world_displacement_mm": float(np.max(np.linalg.norm(new - new[0], axis=1)) * 1000)}
        summary.append(window)
        ax.plot((values - start) * 1000, (np.asarray(legacy) - legacy[0]) * 1000, label="Legacy", color="#c8603b")
        ax.plot((values - start) * 1000, (new[:, 0] - new[0, 0]) * 1000, label="World-locked", color="#276747")
        ax.set(title=f"t = {start:.2f} s", xlabel="Elapsed time (ms)", ylim=(-35, 14))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Paw displacement along world X (mm)")
    axes[1].legend(loc="lower left")
    figure.suptitle("Right front paw in three 100 ms stance windows")
    figure.savefig(output / "stance-comparison.png", dpi=155)
    plt.close(figure)
    cfg = bundle.resolved_config
    view = camera_pose(cfg, sample_pose(bundle.trajectory, 595 / 30))
    report = {"work_run_id": bundle.run_id, "windows": summary,
              "bone_lengths_m": {leg.name: [leg.upper, leg.lower] for leg in DOG_LEGS},
              "matched_static_frame": 595, "camera_height_body_m": cfg.camera.mount_xyz_m[2],
              "hfov_degrees": cfg.camera.hfov_deg, "world_from_camera": view.tolist()}
    (output / "contact-checks.json").write_text(json.dumps(report, indent=2) + "\n")


def yard_plan(bundle, output):
    w = bundle.resolved_config.world
    fig, ax = plt.subplots(figsize=(8.2, 6.15), layout="constrained")
    ax.add_patch(Rectangle((-w.width_m/2, -w.depth_m/2), w.width_m, w.depth_m,
                           facecolor="#e2ecd4", edgecolor="#475844"))
    ax.add_patch(Rectangle((-4.77, w.depth_m/2), 10.30, 1.85, facecolor="#dfd7ca",
                           edgecolor="#594b3e", label="Terrace"))
    ax.add_patch(Rectangle((-4.77, w.depth_m/2 + 1.85), 10.30, .32, facecolor="#aa634b",
                           label="House facade"))
    ax.add_patch(Circle(TRAMPOLINE_XY, TRAMPOLINE_RADIUS, color="#447b78", alpha=.55, label="Trampoline"))
    ax.scatter(*TOY_XY, marker="s", s=28, c="#ce8124", label="Toys")
    ax.plot(FENCE_ENDPOINTS, [FENCE_Y, FENCE_Y], c="#7d5a41", label="Play-area rail")
    for label, track in (("Dog", DOG_TRACK), ("Hedgehog", HEDGEHOG_TRACK), ("Ball", BALL_TRACK)):
        ax.plot([track.start_xy[0], track.end_xy[0]], [track.start_xy[1], track.end_xy[1]], "--",
                label=label)
    ax.plot(*bundle.trajectory.position_m[:, :2].T, color="#31417b", lw=1.9, label="Mower")
    ax.set(xlim=(-6.4, 6.4), ylim=(-5.35, 7.5), aspect="equal", xlabel="World X (m)", ylabel="World Y (m)")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.savefig(output / "yard-plan.png", dpi=150)
    plt.close(fig)


def compare_images(delivery, output):
    images = [delivery / f"static-{name}" / "frame-000595.png" for name in ("fence", "house")]
    with Image.open(images[0]) as reference:
        width, height = reference.size
    panel = Image.new("RGB", (2 * width + 34, height + 88), "#161c20")
    draw = ImageDraw.Draw(panel)
    for i, (path, label) in enumerate(zip(images, ("Timber fence", "Brick house and terrace"))):
        x = 11 + i * (width + 12)
        with Image.open(path) as image:
            panel.paste(image.convert("RGB"), (x, 37))
        draw.text((x, 15), label, fill="#f0ede4")
    draw.text((12, height + 55),
              "Matched onboard frame 595 / 19.83 s   |   22 cm mount   |   100-degree HFOV   |   Depth of field off",
              fill="#f0ede4")
    panel.save(output / "same-camera-yard-comparison.jpg", quality=92, subsampling=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_bundle", type=Path)
    parser.add_argument("delivery", type=Path)
    args = parser.parse_args()
    bundle = load_bundle(args.work_bundle)
    measure(bundle, args.delivery)
    yard_plan(bundle, args.delivery)
    compare_images(args.delivery, args.delivery)
