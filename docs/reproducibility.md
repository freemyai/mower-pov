# Phase 0–4 reproducibility

Verified on Linux x86-64 with Python 3.11.15, NumPy 2.2.6, pytest 9.0.3,
Matplotlib 3.10.9 and FFmpeg/ffprobe 7.1.1. The latter reports libx264 and AAC.

The local Blender executable reported 5.2.4 LTS, build cd5dc428cce1. Use the
[official Linux archive](https://download.blender.org/release/Blender5.2/blender-5.2.4-linux-x64.tar.xz)
and [official checksum list](https://download.blender.org/release/Blender5.2/blender-5.2.4.sha256).
The verified SHA-256 for that archive is
`2fd14c50987426897ba60237ee996edf2b4151f4b1a72a460ca20be58c1d2d86`.
The OptiX doctor selects one device and executes a 16×16 Cycles render.
This establishes device availability for that scene. Scene render time and VRAM use
for dense grass remain to be measured.

`prepare` writes a SHA-256-named run directory. Its manifest records the full normalized config,
source and array hashes, local commit when available, toolchain, assets and file hashes.
Load it with `mower_pov.cache.load_bundle(Path(run_dir))`. Array files are NPZ loaded with
pickle disabled. Terrain arrays index increasing world Y by row and increasing world X by
column. The origin is the lower outside corner. Grass coverage data includes exact root
contact times and separate grid-center times. Root IDs and all appearance attributes
are stable across time for a fixed configuration. Positive infinity is the NPZ sentinel
for a root or grid center with no cut time.

A wheel contact record stores the left and right swept segments and their absolute-time
intervals. The width is a provisional 3.5 cm. Grass bending will be applied in the scene
adapter. Cut load aggregates measured root shortening in 100 ms bins and timestamps the
summary at the last contributing contact. Root counts and removed-volume proxies are
available for later audio shaping.

The phase figures are state diagnostics. Artistic acceptance of grass visibility after the
turn requires inspecting the Blender camera view. At 240 Hz, the core models a moving disc
over piecewise linear center segments and the chassis with a conservative swept envelope.
