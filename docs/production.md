# 午后庭院制作说明

主片沿用 40 秒作业缓存、固定机载相机和真实 28 厘米刀盘宽度。刺猬、小狗、
滚球、玩具车、沙模与蹦床均由源码生成。主片中小狗在矮栏另一侧活动，
孩子的蹦床镜头属于独立花絮。角色、道具和矮栏与机器的完整轨迹做保守净距检查。
自定义庭院中，障碍物占用固定动物路线时会关闭这组场景布置。

## 本地运行

以下命令使用 Python 3.11、Blender 5.2.4 LTS 和带 zscale、libx264、AAC 的 FFmpeg。
本机为四张 RTX A6000。工作缓存已经生成时，可以直接用原始绝对时间重放。

```sh
run="$(awk '{print $2}' runs/phase-0-4/prepare.txt)"
blender=/path/to/blender
scripts/render_life_review.sh "$run" "$blender" runs/life-review
scripts/render_showcase.sh "$run" "$blender" runs/garden-extras
```

主片使用 1920×1080、最高 128 采样和 Cycles/OIDN。每帧同次渲染输出 AgX PNG 与
包含 RawBeauty、Albedo 和 Normal 的场景线性 EXR。分段时间仍从完整作业时间线读取。
相同渲染请求恢复时按清单和 SHA-256 跳过完整帧。主片源代码固定在生产目录
的 `source-commit.txt` 所记版本，恢复时应使用同一版本、Blender 构建和分段范围。

```sh
out="$PWD/runs/feature"
mkdir -p "$out"
realpath "$run" > "$out/work-bundle.txt"
git rev-parse HEAD > "$out/source-commit.txt"
.venv/bin/python -m mower_pov audio "$run" --output "$out/backyard-40s.wav"
scripts/render_feature.sh "$run" "$blender" "$out"
.venv/bin/python scripts/finalize_feature.py "$out"
```

渲染结果应通过主片图像对校验、完整解码、色彩标签、音频起止时间和削波检查。
视觉与试听结果在质量报告中单列。主片的声音由轮速、刀盘开关和实际新增切割事件驱动，
花絮中的蹦床落地声对齐程序化跳跃。未使用第三方模型、录音或真实人物肖像。

## 花絮与重放实测

花絮为 18 秒、1280×720、30 fps、540 帧，输出 AAC 48 kHz 双声道。镜头依次为
刺猬近景、小狗追球、孩子蹦床和割后草地。孩子与动物采用程序化造型。
花絮中断后按原源码指纹继续渲染，完整帧由 SHA-256 验证后复用。

主片第 900 帧在另一张 RTX A6000 上独立渲染，平均 8 位 RGB 绝对误差约 0.017，
最大差异为一个色阶。该记录验证同一机型上的局部重放，完整成片的解码和音画参数
由生产目录中的质量报告另行记录。

## 已生成的本地交付

- 主片 `runs/backyard-final/mower-pov-40s.mp4`。40 秒、1920×1080、30 fps、1200 帧，
  H.264 BT.709 和 AAC 48 kHz 立体声。逐帧校验和完整解码通过。
- 花絮 `runs/backyard-showcase/delivery/garden-extras-18s.mp4`。18 秒的四镜头庭院短片。
- 角色样片 `runs/backyard-life/delivery/`。有声的刺猬 4 秒和小狗加返程 8 秒。
- 本地交付 ZIP 含成片、封面、抽帧联系表、音轨、质量报告、实际作业缓存与源码快照。

实测主片每帧渲染与写入中位数 3.72 秒、95 分位 5.21 秒。抽查了编码主片的
十张关键帧，也核对了三处分段变化和音轨与源 WAV 的采样对齐。角色仍偏风格化，
实时播放里的细草闪烁与主观试听仍待复核。
