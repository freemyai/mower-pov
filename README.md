# Mower POV

A robotic mower simulation with persistent grass cutting, a fixed onboard camera, and reproducible static corner cases.

[简体中文](README.zh-CN.md)

## Watch the 104-second onboard film

https://github.com/user-attachments/assets/939a06dc-019f-4a18-ab2f-4dfc0f027b0c

Five passes and four turns bring the previous cut bands back into view. The mower slows down and routes around a garden hose, a ball, and a watering can.

[1080p original and checksum](https://github.com/ros-claw/mower-pov/releases/tag/demo%2Fcoverage-104s) · [Film provenance and reproduction](docs/releases/v0.1.0-demo.md)

## Run a case on CPU

The static suite has 30 synthetic cases across six families, including a thin hose, an obstacle near a turn, a narrow passage, trampoline supports, a flower bed, and overlapping passes. The evaluator checks the recorded chassis sweep, motion limits, blade state, and first-cut times independently of the planner. Blender is optional for replaying a failure on video.

From the repository root, use Python 3.11, `uv`, and the locked dependencies.

```sh
uv venv --python 3.11
uv pip install -r requirements-lock.txt
uv pip install --no-deps -e .
.venv/bin/python -m mower_pov cases validate scenarios/suites/static-v1.json
.venv/bin/python -m mower_pov cases run scenarios/suites/static-smoke-v1.json \
  --policy legacy-joint --out runs/static-smoke
```

The quick run evaluates six cases and writes `runs/static-smoke/summary.json` plus a verdict and evidence manifest for each case. Use `static-v1.json` in the run command for all 30. The project also supports batch resume, policy comparison, and failure clips with absolute time overlays. [Case guide](docs/cases/ADDING_CASES.md)

## Local benchmark results

| Evaluation | Sequential planning | Joint coverage planning |
| --- | ---: | ---: |
| 104-second film coverage on the 4.2264 m² target | 72.30% | 85.68% |
| 24 paired feasible cases, area-weighted coverage | 61.49% | 79.82% |
| Revisited area across those 24 cases | 1.3416 m² | 0.6798 m² |

Both policies completed 24 cases and safely rejected six cases with an independent infeasibility certificate. Coverage decreased in four of the ball-near-turn cases. The static suite uses known obstacle geometry and a kinematic model. [Method and per-case results](docs/cases/static-v1-results.zh-CN.md) · [Machine-readable evidence](evidence/v0.2/static-v1-results.json)

## Reproduce the work and the film

- [104-second production recipe](docs/dynamic-production.md#割草作业区成片) for the immutable work cache, Blender/Cycles rendering, rendering resume, and audio.
- [Reproduction notes](evidence/v0.1/reproduction.md) for rebuilding the work bundles from the film and baseline commits and comparing coverage.
- [Dynamic backyard replay](docs/dynamic-production.md) for the separate 40-second scene with a crossing dog and a waiting hedgehog.

Source code is GPL-3.0-or-later. [License](LICENSE) · [Third-party assets](THIRD_PARTY_NOTICES.md)
