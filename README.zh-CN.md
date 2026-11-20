# Mower POV

用程序生成割草机器人固定机载视角。任务 0–4 已实现 40 秒庭院路线、地形草根、刀盘状态、轮迹、固定外参、事件与可乱序读取的作业缓存。

运行方式见 [README](README.md)，实测环境和数据契约见 [复现说明](docs/reproducibility.md)，渲染阶段见 [实施计划](docs/implementation-plan.md)。

任务 5–6 的本地输出在 `runs/phase-5-6/`。主样片是 `mower-pov-preview-8s.mp4`，
关键帧位于 `keyframes/`，连续画面和降噪对照位于 `review/`。
[视觉审查](docs/artistic-review.md)记录了可见的短草带、当前画质限制和后续验收门槛。

庭院打磨版使用 `beauty` 预设，在 `runs/garden-polish/delivery/` 生成 720p 样片。
运行方式与视觉变化见[庭院打磨记录](docs/artistic-review.md#庭院打磨版)。

## 午后庭院

主片企划为 40 秒，固定机载视角，远处的小狗与滚球位于矮栏另一侧，
刺猬从草间退入矮灌木。返程仍展示作业缓存中留下的短草带。
18 秒花絮用四个独立镜头呈现刺猬、小狗、蹦床和割后草地。
[制作与复现说明](docs/production.md)列出实际使用的命令和交付位置。
