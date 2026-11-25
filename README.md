# Mower POV

A deterministic work-state pipeline for a robot-mounted mower camera. Phase 0–4 implements
the 40-second route, terrain, cutting and wheel paths, fixed camera, events, and replay bundle.

Use Python 3.11 or later. Blender is needed for the Cycles environment check.

```sh
uv venv --python 3.11
uv pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python -m mower_pov doctor --blender /path/to/blender --device OPTIX --output runs/phase-0-4/environment.json
.venv/bin/python -m mower_pov prepare --environment runs/phase-0-4/environment.json --diagnostics runs/phase-0-4
```

Select `--device CPU` for a CPU smoke check. The simulation runs with `prepare` even
when a Cycles installation is unavailable. See [reproducibility](docs/reproducibility.md)
for the verified installation and bundle format.

Source code is GPL-3.0-or-later. See [LICENSE](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md).

## Blender review clip

The review clip uses absolute frames 525–764 of the default 40-second run. Use the pinned
Blender executable and FFmpeg with libx264 and zscale.

```sh
run="$(awk '{print $2}' runs/phase-0-4/prepare.txt)"
scripts/render_review.sh "$run" /path/to/blender runs/backyard-review OPTIX
```

The command writes the 8-second MP4, seven keyframes, multilayer linear EXRs, raw and OIDN
comparisons, and render measurements. Set the last argument to `OPTIX4` for four discrete
OptiX devices or `CPU` for an explicitly selected CPU run. An individual frame can be
rendered with `python -m mower_pov render --help`. See the [visual review](docs/artistic-review.md)
and [measured render timings](docs/benchmarks.md).

Set `MOWER_REVIEW_PRESET=beauty` for the 720p garden polish covering the same timeline,
with a fixed lens focus and side-by-side crop comparisons.

## Local feature production

The feature uses the onboard view with the procedural yard animals. A separate four-shot
garden montage includes the trampoline and child animation. See [production](docs/production.md)
for the commands, artifact paths, and review status.

## Dynamic backyard

`configs/backyard-dynamic.toml` records a 40-second fixed-camera scenario with a crossing dog,
a waiting hedgehog, a separately modeled blade rotor, and a return over the cut swath.
It uses scene truth for obstacle occupancy. See the [production guide](docs/dynamic-production.md)
and [dog asset and contact measurements](docs/dog-rig.zh-CN.md).

## Static mowing corner cases

The static development suite runs 30 fixed-seed scenarios across six families.
The runner compares sequential and joint planning on the same known geometry.
A separate evaluator checks continuous chassis clearance, the actual trajectory,
and persistent cutting state. Numeric batches run on CPU. Blender can replay
a recorded failure window. See the [case guide](docs/cases/ADDING_CASES.md).
