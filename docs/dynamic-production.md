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
