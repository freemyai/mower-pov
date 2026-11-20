# 来源索引与证据范围

检索/读取日期：2026-11-20。文中 [R编号] 指向下列来源。仓库文件属于代码静态审阅；作者提供的渲染结果/耗时不是本次独立实测。新项目的源码审阅基线为 `abdb085e7cf3950d68e751c5f11216e5d67063a4`。

| 编号 | 来源与地址 | 用于支持 |
|---|---|---|
| R1 | Rental Cat POV README：`https://github.com/rockeymen/rental-cat-pov/blob/abdb085e7cf3950d68e751c5f11216e5d67063a4/README.md` | 作者报告的片长、规格、运行要求、声轨与许可范围 |
| R2 | build.py：`https://github.com/rockeymen/rental-cat-pov/blob/abdb085e7cf3950d68e751c5f11216e5d67063a4/tools/build.py` | 上游获取、缓存、SHA-256 检查、补丁与编译 |
| R3 | upstream.json：`https://github.com/rockeymen/rental-cat-pov/blob/abdb085e7cf3950d68e751c5f11216e5d67063a4/integration/upstream.json` | 上游固定提交和每文件校验值 |
| R4 | film-motion.h：`https://github.com/rockeymen/rental-cat-pov/blob/abdb085e7cf3950d68e751c5f11216e5d67063a4/scene/film-motion.h` | 时间标记、插值、猫步态、镜头与闭眼 |
| R5 | rental.h：`https://github.com/rockeymen/rental-cat-pov/blob/abdb085e7cf3950d68e751c5f11216e5d67063a4/scene/rental.h` | 程序化房间、escapePosition、两条鱼的反应延迟与姿态 |
| R6 | render.sh：`https://github.com/rockeymen/rental-cat-pov/blob/abdb085e7cf3950d68e751c5f11216e5d67063a4/tools/render.sh` | 预览/正式预设、按文件大小恢复、编码和 hqdn3d |
| R7 | denoise.cpp：`https://github.com/rockeymen/rental-cat-pov/blob/abdb085e7cf3950d68e751c5f11216e5d67063a4/tools/denoise.cpp` | 固定尺寸、RGB-only OIDN RT 逐帧处理 |
| R8 | mix.py：`https://github.com/rockeymen/rental-cat-pov/blob/abdb085e7cf3950d68e751c5f11216e5d67063a4/audio/mix.py` | 实录混音与写死的声音时间点 |
| R9 | CREDITS：`https://github.com/rockeymen/rental-cat-pov/blob/abdb085e7cf3950d68e751c5f11216e5d67063a4/CREDITS.md` | 原创代码 MIT、上游独立授权、录音 CC0 来源声明 |
| R10 | main.mm 集成补丁：`https://github.com/rockeymen/rental-cat-pov/blob/abdb085e7cf3950d68e751c5f11216e5d67063a4/integration/main.mm.json` | 字符偏移替换与固定版本耦合 |
| R11 | Robot Vacuum POV：`https://github.com/moritzkremb/robovac-pov/blob/main/README.md` | 机器人运动、固定相机、事件音轨、已知作业交互限制 |
| R12 | Cat Garden 固定版本 README：`https://github.com/BubbleBrainBB/awesome-fable-5.2/blob/00688eb6aef1dacad5c7f3d726f8b642f911c131/cat-garden/README.md` | 程序化植物、风、路线相关密度、渲染/QC 方法和局限 |
| R13 | GitHub 官方仓库许可说明：`https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository` | 公开仓库与开源授权的区别 |
| R14 | Blender 5.2 LTS 发布页：`https://www.blender.org/releases/5-2/` | 5.2.4 日期、LTS 支持期限；选择固定基线而非追逐主版本 |
| R15 | Blender 5.2 LTS GPU 文档：`https://docs.blender.org/manual/de/5.2/render/cycles/gpu_rendering.html` | 各 GPU 后端及平台条件，使用官方可读取语言版本 |
| R16 | Blender 5.2 LTS 命令行文档：`https://docs.blender.org/manual/id/5.2/advanced/command_line/arguments.html` | 后台渲染与 Cycles 设备参数 |
| R17 | Intel Open Image Denoise 官方文档：`https://www.openimagedenoise.org/documentation.html` | RT 滤波、HDR 与 albedo/normal 辅助层，不代表保证时间一致性 |
| R18 | Blender 官方 FAQ：`https://www.blender.org/support/faq/` | 使用 Python API 的公开脚本与生成内容的许可区分 |
| R19 | OpenAI Codex 工作流文档：`https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex` | AGENTS.md 的项目约定和分阶段执行；不是渲染能力实测 |
| R20 | Freemy 官网 README：`https://github.com/freemyai/website/blob/main/README.md` | gallery.json、真实媒体、来源/验证字段、分享流程与项目测试 |
| R21 | GitHub Actions 安全说明：`https://docs.github.com/en/actions/reference/security/secure-use` | 公共仓库自托管 runner 的风险 |
| R22 | 新项目审阅提交：`https://github.com/rockeymen/rental-cat-pov/commit/abdb085e7cf3950d68e751c5f11216e5d67063a4` | 审阅快照身份 |
| R23 | 用户提供的 X 帖子：`https://x.com/rockeymen/status/2123605250825486383` | 仅作为用户提供的参考入口；本次未成功取得完整正文/视频，不用于独立画质结论 |

## 版本与许可注意

R11 和 R20 使用读取当时的默认分支文件；实施前要再次锁定实际提交。新项目在审阅期间读取到的 build.py blob 为 `d6a3f505beccacf0c8dbe6d8b72293a4d455229a`；README blob 为 `91ba27be0bc2d6902c5782d871cba705dcb6f5bf`。

来源中描述的素材许可是作者声明，本方案没有重新分发任何这些素材。新项目的许可证选择是工程发布建议，发布前应核对每个实际使用的依赖/素材及对应版本；复杂授权情况应获得明确权利确认。

## 事实与建议的区分

RESEARCH.md 的“实际实现”来自上述文件；DESIGN.md 的尺寸、速度、割宽、采样、时序、测试阈值与模块结构均是本项目建议值，不应引用为真实割草机器人实测参数。

整个文档包不包含运行测试、性能测量、图片质量评审或发布完成的证据。那些证据应由实施过程生成。
