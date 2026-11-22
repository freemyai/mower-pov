# 动态后院的机载成片制作

三个样片沿用同一份 40 秒动态作业缓存和固定机载机位。样片默认为 960×540、30 fps，
主片为 1920×1080、30 fps、1200 帧。输出 H.264 BT.709 视频和 AAC 48 kHz 立体声。
控制器读取场景真值。画面中的停车和等待来自作业缓存里的运动、障碍占用和刀盘状态。

## 准备缓存

从仓库根目录运行。使用依赖锁文件里的 Python 环境、固定版本的 Blender 和包含
zscale、libx264、AAC、realtime 和 arealtime 的 FFmpeg。每段使用同一份缓存，
`prepare` 输出缓存路径。

```sh
python=.venv/bin/python
blender="$PWD/.tools/blender-5.2.4-linux-x64/blender"
root="$PWD/runs/dynamic-production"
mkdir -p "$root"
"$python" -m mower_pov prepare --config configs/backyard-dynamic.toml \
    --runs "$root/work" --diagnostics "$root/diagnostics" | tee "$root/prepare.log"
run="$(awk '/^Bundle: / { print substr($0, 9) }' "$root/prepare.log")"
reviews="$root/reviews"
```

## 审查样片

下列区间采用左闭右开绝对帧号。根据缓存里的控制事件选择区间，保持片长分别为庭院
8–10 秒、小狗 8–12 秒、刺猬 6–10 秒。每个命令保留自己的样片、联系表、状态日志
和审查记录。默认逐段使用四张 OptiX GPU，每块 GPU 分配一个连续分段。
样片可用 `--preset beauty` 指定 1280×720。

```sh
scripts/render_dynamic_feature.sh review yard "$run" "$blender" "$reviews" 0:270
scripts/render_dynamic_feature.sh review dog "$run" "$blender" "$reviews" 207:507
scripts/render_dynamic_feature.sh review hedgehog "$run" "$blender" "$reviews" 516:801
```

检查庭院的空间关系、近景草和地面接触。检查小狗进出视野的时间、支撑脚接触、
机器制动和等待，结合状态日志核对清空确认及恢复。检查刺猬静止期间的画面可读性
和机器等待。使用联系表选关键帧，并按原始速度播放 MP4 观察闪烁、拖影和运动连贯性。
试听时检查轮声、刀盘和切草声与画面的对应关系。

```sh
ffplay -autoexit -sync audio "$reviews/dog/dog.mp4"
scripts/render_dynamic_feature.sh accept dog "$reviews" --visual-method playback \
    --visual-notes "$DOG_VISUAL_NOTES" --audio-notes "$DOG_AUDIO_NOTES"
```

使用实际观察填入笔记，每段各记录一次。`--visual-method contact-sheet` 记录联系表方式。
没有主观试听时省略 `--audio-notes`，审查记录里的试听状态保持 `pending`。
每段同时执行按原速节拍的完整 A/V 解码，解码时间、黑帧和静帧候选区间及音频峰值
写入机器检查数据。候选区间使用片内时间，状态日志和联系表使用全局时间。

CPU 上可验证一小段完整流程。以下命令输出 640×360 三帧冒烟检查。

```sh
MOWER_DEVICE=CPU scripts/render_dynamic_feature.sh review yard \
    "$run" "$blender" "$reviews" 0:3 --preset smoke
```

## 渲染主片

三个样片都记录审查结果后运行主片命令。脚本按 SHA-256 核对样片和审查记录，
同时核对当前源码指纹和动态缓存身份。渲染用源缓存的采样数（动态后院默认 40）、
Cycles/OIDN 和大景深，并保存 PNG。分段使用同一绝对作业时间线。默认瞬时曝光，
可用 `MOWER_SHUTTER_MS=8 MOWER_SHUTTER_SAMPLES=3` 启用 8 毫秒三子时刻积分。
清单记录使用的快门设置。同一设置可用于重放短窗口以对照成像差异。

```sh
scripts/render_dynamic_feature.sh feature "$run" "$blender" "$root/feature" "$reviews"
```

指定 `MOWER_GPU_INDICES=0,1` 可在两张卡上执行。恢复中断时以相同参数运行，
保留源码、Blender 构建、GPU 配置和分段范围。每个分段按请求清单和帧校验和恢复。
要更改上述设置时使用新的输出目录。

主片输出为 `feature/mower-pov-dynamic-40s.mp4`。每段包括 `manifest.json`、
`decode.log`、`contact.jpg`、`state.jsonl`、`timeline-events.json` 和
`timeline.wav`。片内声音按绝对采样位置从同一份完整 WAV 裁取。每帧状态日志包含
绝对帧号和时间、机器位置与速度、轮速、控制模式、刀盘命令、归一化转速、障碍净距
和动物状态。清单记录缓存、Blender 构建、分段请求和媒体校验和。

## 作业状态与验证

动态缓存使用版本 2，`story.npz` 包含三条演员轨迹、控制模式、通道占用、
制动关联对象、扫掠净距、刀盘指令和归一化转速。它与底盘轨迹共享 240 Hz 时钟，
与割草和轮迹数据一起经过文件及数组哈希校验。版本 1 的旧片缓存仍可读取。

控制器对机器人外接圆、动物代理半径和连续采样之间的相对扫掠做保守检查。
前向占用查询使用场景提供的演员位置与短时预测。刀盘有独立的 0.72 秒停转包络，
割高历史根据转速阈值和新路线重新计算。这些模型只用于本项目的仿真场景。
`hedgehog-stays.toml` 会持续等待至结束。`toy-detour.toml` 比较绕行候选后使用
通过静态扫掠验证的路线。主片用原地掉头把已割草带重新带入固定相机视野。

玩具绕行与动物横穿的联合情况、转弯中新出现的障碍属于当前编排以外的情况。
动作片、住宅材质和犬只资产均可随本地源码复现。

```sh
MOWER_BLENDER="$blender" .venv/bin/python -m pytest -q
```

## 割草作业区成片

从 `configs/mowing-course.toml` 准备缓存。主片直接使用缓存的总帧数、帧率、分辨率和
Cycles 采样数。样片与主片读取同一底盘、割草、避障和声音时间线。

```sh
root="$PWD/runs/mowing-course"
mkdir -p "$root"
"$python" -m mower_pov prepare --config configs/mowing-course.toml \
    --runs "$root/work" --diagnostics "$root/diagnostics" | tee "$root/prepare.log"
run="$(awk '/^Bundle: / { print substr($0, 9) }' "$root/prepare.log")"
reviews="$root/reviews"
```

先用短帧段测试渲染和媒体制作。CPU 冒烟测试使用 640×360。正式分辨率预检使用
`--preset final`，可选择避障或转弯附近的帧，并在清单中比较单帧中位耗时和第 95 百分位耗时。

```sh
MOWER_DEVICE=CPU scripts/render_dynamic_feature.sh course-review work-area \
    "$run" "$blender" "$reviews" 0:3 --preset smoke
scripts/render_dynamic_feature.sh course-review work-area \
    "$run" "$blender" "$reviews" 0:16 --preset final
```

从缓存的 `events.json` 和诊断轨迹选择三个连续的 4–12 秒时段。`work-area` 检查作业区和割草带，
`avoidance` 检查识别障碍后的绕行与回接，`turning` 检查换行。用左闭右开绝对帧号替换
下例的区间变量。逐段观看画面，并用实际观察填写验收记录。

```sh
scripts/render_dynamic_feature.sh course-review work-area "$run" "$blender" "$reviews" "$area_frames"
scripts/render_dynamic_feature.sh course-review avoidance "$run" "$blender" "$reviews" "$avoidance_frames"
scripts/render_dynamic_feature.sh course-review turning "$run" "$blender" "$reviews" "$turning_frames"
scripts/render_dynamic_feature.sh accept avoidance "$reviews" --visual-method playback \
    --visual-notes "$avoidance_visual_notes"
```

为三段都记录审查结果后即可生成主片。作业区样片和主片使用 FFmpeg 完整解码，再由
FFplay 通过 Xvfb 软件窗口原速播放。机器报告记录播放器时钟、音画时差和丢帧计数，
并比较 AAC 解码后的 PCM 和源 WAV 的绝对采样窗口。虚拟音频设备用于自动验证。人工试听后
可在样片验收命令中补充 `--audio-notes`。

```sh
scripts/render_dynamic_feature.sh course "$run" "$blender" "$root/feature" "$reviews"
```

长任务可在独立终端中运行。后台任务把日志和退出码保留在作业目录，便于稍后检查。

```sh
export MOWER_COURSE_STATUS="$root/feature.exit"
printf 'running\n' > "$MOWER_COURSE_STATUS"
setsid -f bash -c '"$@"; result=$?; printf "%s\n" "$result" > "$MOWER_COURSE_STATUS"' \
    _ env MOWER_GPU_INDICES=0,1,2,3 scripts/render_dynamic_feature.sh course \
    "$run" "$blender" "$root/feature" "$reviews" \
    >> "$root/feature.log" 2>&1 < /dev/null
```

如果后台任务中断，原样重跑主片命令即可补齐缺失或校验失败的帧。保持缓存、渲染源码、Blender 构建、快门和 GPU
列表一致。同一输出目录由生产锁保护。任务完成后，`feature/mower-pov-mowing-course.mp4` 与
`feature/manifest.json` 记录成片及可核对的媒体和渲染证据。

五条名义作业行以 25 厘米间距往返。局部规划用场景静态占用几何标记不可直行的段，
比较两侧候选的连续底盘净距、行程、转弯耗时和重复作业距离。轨迹在接近障碍时限速，
绕开安全余量后回接同一行。换行用限角速度、限角加速度的圆弧完成。回放记录的是同一作业
时间线上的刀盘、轮迹、短草和相机。近障压过的草叶只改变显示姿态。

本场景使用预先知道的静态占用与仿真运动模型，`REPLAN` 标记名义路线的绕行段开始。
覆盖率分母是五条名义直行的完整刀盘扫掠，包含因障碍及净距预留而保留的草。
相机视锥候选显示旧草高度在乱序重放中是否保持，最终画面仍要核对视线遮挡。

```sh
.venv/bin/python scripts/measure_course.py "$run" "$root/evidence"
```

## 比较覆盖效率

覆盖规划将相邻行、绕行与掉头的刀盘扫掠投到同一网格。先保留连续底盘净距
符合要求的候选，再联合选择五条作业行，最后成对重试行的组合。目标区的新草
具有更高权重。重访面积、额外扫过的次数、行程和耗时都会影响选择。启动、掉头和停刀
时间纳入同一时间预算。控制仍然使用已知的场景占用。

首割缓存沿用实际 1 厘米网格和独立草根。遇到足够长且没有任何新草被割到的
路段，可以在移动中让刀盘停转。只有从新命令重算的网格和草根首次割到时间
完全一致时，才保留该方案。刀盘状态由同一缓存供渲染和声音使用。

```sh
baseline=$(cat runs/mowing-course/work-bundle.txt)
.venv/bin/python scripts/compare_coverage.py "$baseline" "$run" "$root/comparison"
```

对比时应沿用同一组障碍和目标。一次连续进入刀盘范围计一次覆盖。
离开后带刀再次进入才计入重复。已经定义的五条名义作业行构成目标分母，
其中包含障碍安全余量内的区域。比较图中保留仍未割到的位置。
