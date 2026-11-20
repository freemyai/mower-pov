# Mower POV

用程序生成割草机器人固定机载视角。任务 0–4 已实现 40 秒庭院路线、地形草根、刀盘状态、轮迹、固定外参、事件与可乱序读取的作业缓存。

运行方式见 [README](README.md)，实测环境和数据契约见 [复现说明](docs/reproducibility.md)，渲染阶段见 [实施计划](docs/implementation-plan.md)。

任务 5–6 的本地输出在 `runs/phase-5-6/`。主样片是 `mower-pov-preview-8s.mp4`，
关键帧位于 `keyframes/`，连续画面和降噪对照位于 `review/`。
[视觉审查](docs/artistic-review.md)记录了可见的短草带、当前画质限制和后续验收门槛。
