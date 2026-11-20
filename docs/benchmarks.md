# Phase 6 render benchmark

Blender 5.2.4 LTS build cd5dc428cce1, Cycles OptiX, NVIDIA RTX A6000 with driver
595.91.07. The run used 960×540, 32 maximum samples per pixel, adaptive threshold 0.035
and GPU OpenImageDenoise with albedo and normal guides. Fixed AgX exposure was +0.14 EV.
Each worker used one GPU and five host threads. The MP4 used FFmpeg 7.1.1.

| Selection | Frames | Median render + PNG write | p95 render + PNG write | Maximum | Peak process RSS | Median PNG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Representative poses on one GPU | 36 | 1.184 s | 1.256 s | 2.488 s | 1,313 MiB | 860,343 B |
| Review clip on four GPUs | 240 | 1.194 s | 1.258 s | 2.505 s | 1,319 MiB | 901,143 B |

The 36 selected frames cover shaded and directly lit grass, close boundary views,
entry into the turn and the later return. The 240-frame sample covers the whole turn.
Scene setup took 2.81–2.88 s per worker for the four preview shards. Peak sampled
GPU allocation was 2,244 MiB across 305 local NVIDIA process samples. That number is a
sampled peak. Host peaks are per-process RSS high-water marks. Measured render time includes
scene synchronization, sampling, denoising and PNG write. Those components were not timed
individually in this run. No 1080p extrapolation is used as an acceptance measurement.

The encoded review clip is 8.000 s, 960×540, 30 fps, H.264, BT.709, yuv420p and
11,978,425 bytes. Its 240 frames were fully decoded. A second render of frame 738 on the
same GPU, reached through a different batch order, differed by at most one 8-bit RGB code
value. The average absolute difference was 0.01957 per channel. The measured RGB changes
at the three shard boundaries were below the 95th percentile across the clip.

[Media probe](evidence/phase-5-6-media.json) · [Measurements](evidence/phase-5-6-measurements.json)
