# Freemy Mower POV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. 无对应技能的本地环境，按相同任务顺序与验收门槛执行，不要把安装插件当成项目功能。

**Goal:** 生成可信的割草机器人机载相机视频，并准备可在 `freemyai/mower-pov` 发布的完整复现项目。

**Architecture:** 独立 Python 作业核心先生成地形、轨迹、持续割草状态与共享事件，再由 Blender/Cycles 只读重放。音频、质量检查与渲染使用同一份不可变缓存。首版不并行维护自研 Metal 后端。

**Tech Stack:** 外部 Python 3.11+、NumPy、pytest、Blender 5.2.4 LTS、FFmpeg/ffprobe；具体依赖和二进制校验值在任务 0 锁定。

**Spec:** `DESIGN.md`；调研依据与来源分别见 `RESEARCH.md`、`SOURCES.md`。

## Global Constraints

- 40 秒；1920×1080；30 fps；1200 帧；48 kHz 双声道音频。
- 240 Hz 运动积分；相机固定于机身；世界 Z 向上；机身 X 前、Y 左、Z 上。
- 刀盘有效半径 0.14 m；首版固定割高 0.045 m；状态网格 0.01 m。
- 默认前进速度上限 0.30 m/s、角速度上限 0.80 rad/s；其他参数以 DESIGN.md 第 4 节为准。
- 未割草高 0.08–0.12 m；固定种子 20261120；禁止让草提前消失。
- 只实现一个渲染后端；没有调用 AI 视频模型才能重渲染的依赖。
- 不复制未获许可的 Cat Garden 源码，不将运行时下载当作授权替代方案。
- 方案中的命令、API、测试和路径是待实现接口，不代表目前已经存在或执行通过。
- 不自动创建公开仓库、发布 Release、触发付费算力或修改 freemyai/website；这些动作属于发布阶段的明确授权边界。

## Review Focus

1. 中断后配置已变：必须拒绝混用旧帧，任务 4/7 测试覆盖。
2. 刀盘关闭、转弯或相机前移：只有真实刀盘扫掠位置可切割，任务 3 覆盖。
3. 乱序渲染和子帧求值：状态、曝光、风和随机数必须一致，任务 4/5/7 覆盖。
4. 显卡不支持或空间不足：明确报错或按用户选择降级，不能伪造成功，任务 0/7 覆盖。
5. 近景运动、阴影与停机被误判坏帧：检测输出候选，视觉结论保留人审，任务 9 覆盖。

## 文件结构

```text
mower-pov/
  README.md
  README.zh-CN.md
  LICENSE
  THIRD_PARTY_NOTICES.md
  CONTRIBUTING.md
  SECURITY.md
  AGENTS.md
  pyproject.toml
  configs/
    backyard.toml
    smoke.toml
  src/mower_pov/
    __init__.py
    __main__.py
    cli.py                 # 统一命令入口
    config.py              # 配置载入、合并、校验
    contracts.py           # 共享类型、单位与坐标定义
    environment.py         # 工具、GPU、版本与磁盘探测
    terrain.py
    motion.py
    camera.py
    cutting.py
    events.py
    cache.py
    audio.py
    quality.py
    rendering/
      driver.py            # 外部进程调度、恢复和输出校验
      blender_scene.py     # 在 Blender 内执行
      blender_grass.py
      blender_camera.py
      color.py
  tests/
    test_config.py
    test_environment.py
    test_terrain.py
    test_motion.py
    test_camera.py
    test_cutting.py
    test_events.py
    test_cache.py
    test_audio.py
    test_quality.py
    integration/
      test_blender_smoke.py
      test_grass_replay.py
      test_render_resume.py
  docs/
    design.md
    research.md
    reproducibility.md
    artistic-review.md
    benchmarks.md
    release-checklist.md
  .github/workflows/ci.yml
  runs/                    # 忽略；每次运行单独指纹目录
```

不要为尚不存在的第二后端搭建插件注册中心，不加入数据库、Web API 或任务服务。配置与缓存 schema 可以稳定扩展，但首版接口保持最小。

## 共用数据契约

由任务 0 在 `contracts.py` 定义以下名称，所有模块只使用同一份定义：

| 类型 | 关键字段与约定 |
|---|---|
| RunConfig | world、mower、camera、grass、render、audio、seed；内部 SI 单位；JSON/TOML 输入校验后不可变 |
| EnvironmentReport | OS、外部 Python、Blender 版本与路径、GPU 与后端、FFmpeg 编码器、磁盘、warnings、usable |
| Terrain | height_m、origin_xy_m、cell_size_m、mowable_mask；数组轴与世界坐标映射显式定义 |
| ControlSchedule | 各时间点的目标 v、omega、blade_enabled，时刻单调 |
| Trajectory | time_s、position_m、orientation、unwrapped_yaw、v_mps、omega_radps、wheel_speeds、blade_enabled |
| BladeRoots | root_id、root_xyz_m、h0_m、width_m、bend、phase；固定生成，不逐帧随机 |
| CutState | t_first_cut_s、cut_height_m、root/grid mapping；未切割为正无穷；轮压数据独立 |
| EventLog | 有序 Event 列表；每项 time_s、kind、position_m、intensity、payload；JSON 数值必须有限 |
| RunBundle | run_id、run_dir、resolved_config、terrain、trajectory、cut_state、events、manifest |
| RenderRequest | run_dir、frame_indices、preset、output_dir；帧编号从 0 开始 |
| RenderResult | produced_frames、failed_frames、timings、memory、device、artifacts |
| QAReport | machine_checks、visual_review、audio_review、warnings、evidence_paths；未检查状态不可变成通过 |

NPZ 仅保存数据数组，不使用允许任意对象反序列化的 pickle。缓存应携带 schema_version，未知主版本拒绝加载。时间与状态计算优先 float64；存储降至 float32 前通过误差测试。

---

## 任务 0：可验证的环境与配置入口

**依赖：**无。**产物：**在用户硬件上可运行的 doctor 与配置校验，而不是空仓库骨架。

**文件：**创建 pyproject.toml、contracts.py、config.py、environment.py、cli.py、__main__.py、两个配置文件，以及 test_config.py、test_environment.py。

**接口：**

```text
load_config(path: Path, overrides: Mapping[str, object] | None = None) -> RunConfig
validate_config(config: RunConfig) -> None
probe_environment(blender_path: Path | None = None) -> EnvironmentReport
```

- [ ] 编写 `test_default_config_frame_count`：40×30=1200；音频采样帧数=1,920,000。
- [ ] 编写 `test_invalid_dimensions_or_rates_rejected`：负草高、零帧率、负半径、输出奇数尺寸配 yuv420p、NaN/Inf 配置均拒绝并给出字段名。
- [ ] 编写 `test_missing_blender_reported` 与 `test_cpu_fallback_is_explicit`：无工具路径应标记不可用；GPU 请求失败不自动宣称 GPU 成功。
- [ ] 执行 `python -m pytest tests/test_config.py tests/test_environment.py -q`，先记录失败，再实现配置载入和环境探测，重跑要求通过。
- [ ] 实现 `python -m mower_pov doctor`，输出可读摘要和 `environment.json`；检查 FFmpeg 是否具备实际选用编码器。
- [ ] 锁定本机验证过的 Blender 5.2.4 安装来源/校验值和依赖版本。不能装上 5.3 后仍在报告写 5.2.4。
- [ ] 提交本任务；GPU 不可用时保留真实报告，只可运行明确标记的 CPU smoke。

## 任务 1：确定性地形与草根数据

**依赖：**0。**文件：**terrain.py、test_terrain.py。

**接口：**

```text
build_terrain(config: RunConfig) -> Terrain
sample_height(terrain: Terrain, xy_m: ndarray) -> ndarray
generate_blade_roots(config: RunConfig, terrain: Terrain) -> BladeRoots
```

- [ ] 编写 `test_terrain_seed_is_deterministic`：同配置、同种子生成相同高度场与 root_id；更改种子改变场景指纹。
- [ ] 编写 `test_blade_roots_stay_on_terrain`：根高度等于采样地形，所有草根位于可割/植被区域，草高在配置范围内。
- [ ] 编写 `test_world_grid_coordinate_roundtrip`：非零原点和非方形网格也能正确往返，避免 x/y 轴交换。
- [ ] 先运行 `python -m pytest tests/test_terrain.py -q` 记录失败，再实现并要求全部通过。
- [ ] 输出高度、可割区域与根分布的诊断图；这是调试资料，不当作正式画面。
- [ ] 提交本任务。

## 任务 2：机器运动、碰撞与固定相机

**依赖：**0–1。**文件：**motion.py、camera.py、test_motion.py、test_camera.py。

**接口：**

```text
make_demo_schedule(config: RunConfig, terrain: Terrain) -> ControlSchedule
integrate_motion(config: RunConfig, terrain: Terrain, controls: ControlSchedule) -> Trajectory
sample_pose(trajectory: Trajectory, time_s: float) -> ndarray  # 4×4
camera_pose(config: RunConfig, body_pose: ndarray) -> ndarray  # canonical optical frame
validate_swept_body(config: RunConfig, terrain: Terrain, trajectory: Trajectory) -> list[Violation]
```

- [ ] `test_straight_drive_and_differential_turn`：直线不漂移；轮距 b=0.32、v=0.20、omega=0.50 时左右轮线速度分别为 0.12、0.28 m/s。
- [ ] `test_limits_and_heading_unwrap`：全部时间步满足上限；穿越 ±π 不出现镜头跳转。
- [ ] `test_camera_mount_is_constant`：随机选 20 个姿态，`inverse(T_world_body) @ T_world_camera` 等于同一安装矩阵，误差目标 1e-6。
- [ ] `test_swept_body_rejects_thin_obstacle`：障碍位于两个采样点之间也会被发现；不能只查路径点。
- [ ] 先运行 `python -m pytest tests/test_motion.py tests/test_camera.py -q`，再最小实现、重跑、保存轨迹俯视图。
- [ ] 调整路径长度与时间，保证 40 秒容纳合理加减速和一次返程；禁止靠瞬移补时。
- [ ] 提交本任务。

## 任务 3：刀盘扫掠与持久割草状态

**依赖：**1–2。**文件：**cutting.py、test_cutting.py。

**接口：**

```text
compute_cut_state(config: RunConfig, terrain: Terrain, trajectory: Trajectory, roots: BladeRoots) -> CutState
sample_grass_heights(roots: BladeRoots, state: CutState, time_s: float) -> ndarray
measure_sweep_error(reference: SweepFixture, state: CutState) -> float
```

关键测试夹具：平地上刀盘中心从 (0,0) 在 1 秒内走到 (1,0)，有效半径 0.14 m，全程开启。用于算法单测的轨迹不要求遵循演示速度上限。

- [ ] `test_first_cut_time_matches_straight_sweep`：草根 (0.5,0) 第一次进入刀盘时间应约为 0.36 秒，细化求交误差不超过一个 240 Hz 步长。
- [ ] `test_outside_or_blade_off_never_cuts`：草根 (0.5,0.16) 不被割；刀盘全关时任何位置不被割。
- [ ] `test_camera_offset_is_not_cutter_offset`：相机前移 0.18 m 不改变上述切割时刻。
- [ ] `test_no_future_cut_leak`：t_cut=5.0 的草在 4.9 秒仍保持 h0。
- [ ] `test_repeat_pass_preserves_short_grass`：返程后不恢复；已经比目标矮的草不被拉长。
- [ ] `test_curved_sweep_converges`：缩小时间/空间步长后结果收敛；1 cm 原型网格的覆盖边界误差目标 ≤2 cm。
- [ ] 先运行 `python -m pytest tests/test_cutting.py -q`；再实现扫掠、首次进入时间与高度查询；重跑要求通过。
- [ ] 输出 t_first_cut、最终草高和刀盘覆盖诊断图；用不同时间抽查状态。
- [ ] 提交本任务。此任务不通过，不允许用渲染效果掩盖状态错误。

## 任务 4：统一事件、不可变缓存与乱序重放

**依赖：**0–3。**文件：**events.py、cache.py、test_events.py、test_cache.py。

**接口：**

```text
build_events(config: RunConfig, trajectory: Trajectory, state: CutState) -> EventLog
prepare_run(config: RunConfig) -> RunBundle
save_bundle(bundle: RunBundle) -> Path
load_bundle(run_dir: Path) -> RunBundle
compute_run_id(inputs: Mapping[str, object]) -> str
```

- [ ] `test_sequential_and_random_access_agree`：时间按顺序或乱序访问时，位置、草高、刀盘事件一致。
- [ ] `test_cache_invalidates_on_semantic_change`：草高、割高、相机、种子、轨迹、渲染版本变化均不复用旧运行结果；无关报告时间戳不改变仿真身份。
- [ ] `test_partial_or_unknown_schema_rejected`：半写文件、缺数组、未知主 schema 版本有明确错误。
- [ ] `test_event_json_has_no_nonfinite_values`：未割时间不会以非标准 Infinity 进入 JSON。
- [ ] `test_cut_load_reduces_on_repeat_pass`：重复通过已割草区域的新增去除草量下降，不能重复发相同切割事件。
- [ ] 先运行 `python -m pytest tests/test_events.py tests/test_cache.py -q`；最小实现后重跑，产出一个可独立加载的完整 RunBundle。
- [ ] 所有 NPZ 用 `allow_pickle=False` 加载；写文件采用临时路径和原子替换。
- [ ] 提交本任务。

## 任务 5：Blender 场景、草叶与相机适配

**依赖：**4。**文件：**rendering/blender_scene.py、blender_grass.py、blender_camera.py；两个 integration 测试。

**接口：**Blender 内部入口 `build_scene(run_dir: str, preset: str) -> None`；读取 RunBundle 对应 JSON/NPZ，不重新计算路线或切割。

- [ ] 编写 `test_blender_smoke_outputs_valid_frame`：一个平面、少量草和固定机壳，输出 640×360 非空、可解码帧。
- [ ] 编写 `test_camera_axis_conversion`：校准几何分别位于前、右、上，投影方向符合 canonical optical 定义。
- [ ] 编写 `test_grass_height_and_identity_replay`：t_cut 前后根 ID/根坐标不变，草高从 h0 变为 min(h0,0.045)。
- [ ] 编写 `test_subframe_grass_and_wind_evaluation`：两个整数帧之间的子帧不读取错误时间；运动模糊开关对状态本身无影响。
- [ ] 运行 `python -m pytest tests/integration/test_blender_smoke.py tests/integration/test_grass_replay.py -q`；缺 Blender 时明确跳过并记未验证，不声称通过。
- [ ] 实现固定草根、实例/节点几何、草叶材质、地形与简单花坛；整段预分配细节，避免动态 LOD 跳变。
- [ ] 输出至少五张关键帧：近草、树影、边界、掉头、返程旧草带；实际打开图像检查。
- [ ] 提交本任务。

## 任务 6：8 秒关键样片与性能门槛

**依赖：**5。**文件：**rendering/driver.py 初版、docs/artistic-review.md、docs/benchmarks.md；复用集成测试。

**接口：**`render_frames(request: RenderRequest) -> RenderResult`。首个样片选择主片内能证明割草持久性的连续 8 秒，保持全局时间，不从该段重新清空草状态。

- [ ] 写 `test_clip_uses_absolute_timeline`：返程片段中的草状态来自完整仿真，而不是片段起点全新草坪。
- [ ] 用低分辨率先验证掉头和旧草带确实入镜；若不明显，改构图、光照或轨迹，不夸大割宽。
- [ ] 选至少 32 个代表帧，覆盖近草、密草、树影、转弯和远景；记录帧耗时、内存与文件大小。
- [ ] 输出 8 秒、30 fps 关键样片；保存原始与降噪对照、五类区域裁切和联系表。
- [ ] 人工审查脚本假动作、轮地关系、草闪烁、机壳遮挡与真实宽度草带；artistic-review 默认未批准，必须记录实际结论。
- [ ] 关键样片未通过时停止全片正式渲染；继续修复本阶段，而不是堆采样或延长片长。
- [ ] 提交代码与小型证据；大中间结果不进 Git。

## 任务 7：恢复、降噪、色彩与完整渲染

**依赖：**6 已批准。**文件：**rendering/driver.py 完整版、color.py、test_render_resume.py。

**接口：**`render_frames` 保持不变；增加 `validate_frame(path: Path, manifest: Mapping) -> bool` 与 `assemble_video(run_dir: Path, audio_path: Path | None) -> Path`。

- [ ] `test_resume_renders_missing_frames_only`：模拟中断，恢复后完整帧不被重渲染，半写帧被重做。
- [ ] `test_same_size_wrong_run_frame_rejected`：尺寸/字节数相同但指纹不同的帧不可复用。
- [ ] `test_segment_boundaries_match`：整段与分段对同一边界时间的状态和曝光一致；同软件栈像素比较采用事先约定的容差。
- [ ] `test_no_disk_space_fails_cleanly`：提前/运行中空间不足保留已完成帧，输出非零退出码，不伪造完整 MP4。
- [ ] `test_color_reference_patches`：固定参考色块验证显示转换/编码解码路径，避免双 gamma 和只贴标签。
- [ ] 先执行 `python -m pytest tests/integration/test_render_resume.py -q` 记录失败，再实现并要求通过。
- [ ] 若使用自动曝光，先缓存完整曲线；若启用时间滤波，验证历史/重叠区，否则保持关闭。
- [ ] 根据任务 6 基准选择最终采样配置，再生成 1200 帧。中间产物保留 scene-linear 数据和必要辅助层。
- [ ] 提交本任务；实测结果写入 benchmarks，不拿参考作者机器耗时代替自己的测试。

## 任务 8：事件驱动声音

**依赖：**4；可在 5–7 期间独立进行，最终集成需 7。**文件：**audio.py、test_audio.py。

**接口：**`synthesize_audio(bundle: RunBundle, output_wav: Path) -> Path`。

- [ ] `test_audio_length_and_channels`：40 秒对应 1,920,000 个双声道 48 kHz 采样帧。
- [ ] `test_blade_off_removes_blade_component`：刀盘关时其声部关闭；行走声仍可存在。
- [ ] `test_events_are_sample_aligned`：在确定性冲击夹具中，事件位置误差不超过指定采样舍入范围。
- [ ] `test_mix_has_headroom_and_no_nan`：浮点混音无 NaN/Inf、无削波，不依赖整型溢出。
- [ ] `test_repeat_cut_load_is_lower`：由真实新增去除草量驱动负载，而非只由空间位置或全局时间驱动。
- [ ] 先执行 `python -m pytest tests/test_audio.py -q`，最小实现、重跑，输出 WAV 与声部对照。
- [ ] 试听启动、稳定行驶、密草、掉头与停止段；没有试听工具则明确记未试听，发布前交人工检查。
- [ ] 提交本任务。

## 任务 9：整片质量报告与复现验收

**依赖：**7–8。**文件：**quality.py、test_quality.py、docs/reproducibility.md。

**接口：**`inspect_run(run_dir: Path, video_path: Path) -> QAReport`。

- [ ] `test_missing_frame_and_bad_decode_detected`：故意删帧/损坏帧，质量检查必须失败。
- [ ] `test_normal_stop_not_automatically_bad`：冻结/黑帧检测输出候选和上下文，不自动把合法停机/阴影判为坏帧。
- [ ] `test_unreviewed_is_not_passed`：visual_review/audio_review 缺失时标为未检查，整体报告不可变成完全通过。
- [ ] `test_output_format_contract`：ffprobe 检查 1920×1080、30 fps、1200 帧，完整解码无错；音视频时序按一帧容差检查。
- [ ] 先运行 `python -m pytest tests/test_quality.py -q`，实现后重跑，再运行全部适用测试。
- [ ] 生成 QA JSON、可读报告、联系表、作业状态图、关键问题截图，以及来源/版本/命令记录。
- [ ] 在干净环境从固定提交与配置重建 smoke；条件允许再抽样重渲染，并明确实际验证的平台范围。
- [ ] 提交本任务。CPU 核心通过不等于 GPU 画质通过；MP4 可解码不等于作业正确。

## 任务 10：开源仓库与官网展示

**依赖：**9。**文件：**README、README.zh-CN、LICENSE、THIRD_PARTY_NOTICES、CONTRIBUTING、SECURITY、AGENTS、ci.yml、release-checklist；官网修改另一个 PR。

- [ ] 生成中英文 README：前置真实成片/预览、准确技术栈、可复现命令、已验证硬件、已知限制、授权与致谢。
- [ ] 发布代码建议 GPL-3.0-or-later；记录实际代码复用的逐文件归属，不把参考链接当成已获取授权。
- [ ] CI 在托管环境跑纯 Python 测试和小型 CPU smoke；不把公共 PR 接入私人的 GPU 自托管 runner。
- [ ] 审查仓库历史、配置和产物无 token、私人路径、无权分发资产与大体积中间帧。
- [ ] 在组织权限和命名可用性确认后，经用户明确发布授权，创建 `freemyai/mower-pov`；本方案不代表此动作已完成。
- [ ] 创建 `v0.1.0` Release，附 40 秒成片、8 秒预览、联系表、配置、manifest、QA 与校验和；源码由版本标签提供。
- [ ] 官网单独读取 `data/gallery.json` 实际 schema、来源字段、媒体规范和分享别名规则，添加真实截图与短片，未验证设备保持未知。
- [ ] 按网站 README 执行内容校验、测试、构建、分享/SEO 检查，并完成受影响页面的浏览器验收；不要顺带重构网站。
- [ ] 发布完成的证据必须是实际远端仓库/标签/Release/网页检查结果，不是本地文件存在。

## 执行策略

主线顺序为 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 9 → 10；任务 8 在任务 4 之后可并行，但不共享修改同一源文件。单个 Codex 会话就逐项执行，无需为了“多代理”引入额外框架。

每个任务保存：目标、实际变更、测试命令与原始结果、失败原因、图片证据、已知问题、下一任务。不要给每个小修改重新读取全部文档；读取当前模块契约与相邻依赖即可。

## 计划自查

设计目标均对应任务：环境/参数→0，确定性草坪→1/5，运动相机→2，作业状态→3/4，关键样片→6，画面与恢复→7，声音→8，质量→9，开源官网→10。跨模块共享接口集中在 contracts.py。首版没有加入第二后端、训练平台或完整导航系统。

本文件只定义预期测试，不含已经运行通过的测试结果。实施者必须逐条执行，不能把本计划复制成“完成报告”。
