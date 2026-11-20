# 8-second onboard review

Status: preview evidence available. Full-length artistic approval is pending.

The silent clip uses world time 17.50–25.50 s from the default 40-second work bundle.
It follows the flower-bed approach through the turn and into the return row. The shot
retains the 100° horizontal lens, 12° downward mount, 25 cm row spacing and 28 cm cutter
width. The seven reference frames also cover the approach under tree shade and direct sunlight.

| Inspection | Observed result |
| --- | --- |
| Previous strip | As the turn finishes, a lower strip appears on the left. In the 24.60 s keyframe, the left-side blade tips sit below the standing blades ahead and to the right. The color and root positions use the same source attributes across the cut. |
| Chassis and boundary | The cowl stays at the bottom edge while the stone edge moves out to the right. The pose comes from the cached chassis rotation and terrain response. The wheels sit below this POV, so direct wheel-to-ground contact cannot be visually approved from these frames. |
| Time continuity | Reviewed 21 positions across the sequence and four adjacent source frames around each of the three shard boundaries. The furniture, horizon and grass state remain continuous in those checks. The full MP4 decoded successfully. |
| Raw and OIDN | Compared near grass, shaded grass, distant grass, the turning boundary and the old strip at native crop scale. OIDN removes visible grain, slightly softens some fine tips in shade, and retains the strip outline. |
| Remaining visual work | The uniform tall-stand silhouette, angular ornamental leaves and repetitive fence still have a procedural look. Camera motion blur was disabled. Frame-by-frame inspection and a numerical temporal screen were performed. Real-time playback review for fine-leaf shimmer remains open. |
| Sound | The review deliverable is silent. Audio audition is pending the audio stage. |

The local video, timeline study, crops and original EXRs are generated in
`runs/phase-5-6/`. A small contact sheet and verified media metadata are included in
[the evidence directory](evidence/).

## 庭院打磨版

按原样片的绝对帧 525–764 再渲染，用同一作业包比较画面。镜头外参、路线、切割宽度
与割高沿用原配置。美术选择为暖色侧光和冷色树影，花坛使用白色雏菊、紫色鼠尾草、
深色覆盖土和错落的石边，围栏后增加有枝干的树冠作为远景。草根沿用原作业数据。
近景草叶有更细的曲线、低层叶丛与透光边缘。

草坪上相邻草叶在同一个随世界时间传播的风场里摆动，花叶与树叶也使用这套风场。
花茎底部固定，越靠近花头摆幅越大。顺序播放或跳到同一绝对时间会得到相同的几何状态。
`beauty` 预设采用 1280×720、每像素最多 48 个采样、固定焦点 0.85 米和 f/9.5。
这一级的轻微景深主要落在紧贴镜头的机壳和最近草尖。

```sh
run="$(awk '{print $2}' runs/phase-0-4/prepare.txt)"
MOWER_REVIEW_PRESET=beauty scripts/render_review.sh "$run" /path/to/blender runs/new-garden-review OPTIX4
```

单卡改用 `OPTIX`。导出的 MP4 为无声预览。远景叶缘和重复植物仍能看出程序化造型，
成片级时域闪烁仍待实时播放复核。
