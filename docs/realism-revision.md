# 家庭后院写实度修订

这一轮以 `19d3196` 为审阅基线，先验证代理犬的脚掌接触、真实草叶截断和后院空间比例。
沿用原作业缓存。住宅方案用 `--yard house` 选取，`--yard legacy` 对应木围栏后景。

## 源码核对

| 位置 | 基线行为 | 本轮处理 |
| --- | --- | --- |
| `rendering/blender_life.py` | 代理犬用随步态伸缩的管段；蹦床支腿与边界关系欠清楚 | 固定上下肢段长度，解算关节与脚掌姿态；蹦床内移并改用三组随地形落脚的 U 形支架 |
| `rendering/life_motion.py` | 布局失效时返回布尔值，角色可能被静默省略 | 轨迹移入 `actors/behavior.py`，步态移入 `actors/gait.py`。渲染前报告冲突对象、边界和最小间距 |
| `rendering/blender_grass.py` | 割后缩放整个草簇，也挤窄基生叶 | 根据原品种和根部高度选择逐叶截断网格，保持根、颜色与底部短叶；有风时仍使用原有倾斜模型 |
| `rendering/blender_scene.py` | 木围栏和树构成背景 | 增加住宅、窗洞、斜屋顶及露台场景组；静态对照沿用同一相机、草坪、玩具、动物和帧号 |
| `rendering/blender_worker.py` | 逐帧更新数据，预览不开景深 | 透传场景选择和贴图指纹。真实快门采样仍待验证 |

新增 `rendering/blender_house.py` 负责住宅几何和材质，`rendering/grass_geometry.py`
负责截断多边形与插值顶点颜色。`cli.py` 和 `rendering/driver.py` 接入场景选择、布局检查
和独立缓存指纹。自动检查见 `tests/test_grounded_gait.py`、`tests/test_backyard_life.py`、
`tests/test_grass_geometry.py`、`tests/test_yard_assets.py` 和 `tests/integration/test_blender_smoke.py`。

## 可复测的结果

- 右前脚的三个支撑窗口各取 100 ms。旧公式的世界 X 位移范围为 27.69、0.05、22.36 mm。
  新接触求解在三个窗口的三维位移均为零。Blender 求值取每窗 7 点，实测三维位移最大低于 0.001 mm。
- 草根 48269 在 18.133745 s 首次被刀盘覆盖。直接读取 Blender 实例网格，静止形态的草由 99.98 mm
  变为 44.99 mm，目标割高为 45 mm。独立抽验中，现有风和轮压倾斜叠加后有 46/1000 个
  已割草簇超过目标高度 5 mm，最大约 9.7 mm。上面的截断精度仅适用于倾斜前的网格。
- A/B 取第 595 帧，19.833 s。相机固定安装高度 22 cm，水平视场角 100°，预览关闭景深。
  两份运行清单只有场景选项和资产摘要不同。庭院 12 × 10 m，住宅正面宽 10.30 m，
  到草坪边缘约 1.85 m。蹦床主环直径 1.46 m，外包络约 1.58 m。低机位下，露台大部分被草坪与矮栏遮挡。
- 5.4 秒侧面样片用于检查代理犬支撑与摆动。3 秒机载样片用于检查固定视角中的住宅、矮栏、
  蹦床与草坪关系。两段 960 × 540、30 fps 的静音样片均完成解码校验，已抽帧目检。
  正常速度的主观评审留待后续正式素材。

## 资产与角色

正式犬模型的准入条件包括连续蒙皮、可核对的米制比例、关节权重与落脚检查，
以及明确允许在公开基线中再分发。当前样片中的犬是程序代理。

| 资产 | 处理结果 |
| --- | --- |
| Poly Haven Red Brick 与 Concrete Floor Worn 001 | 两组 1K 贴图，CC0。已入库。贴图用途、物理比例、作者和逐文件 SHA-256 见[资产锁](../assets/catalog.lock.json)。 [发布方许可](https://polyhaven.com/license) |
| FeralMotion Labrador | 正式犬的候选。作者页面描述了 Blender 文件、骨架和动画。原始资产再分发受限，未购买或试装。 [作者页面](https://feralmotion.itch.io/labrador) |
| Poly Haven Suburban Garden | CC0 的环境光候选。当前 A/B 沿用共同光照。 [发布方资产页](https://polyhaven.com/a/suburban_garden) |

正式犬接入时可用 `actors/gait.py` 的世界空间落脚目标驱动四足 IK。需先在候选资产上实测每肢骨长、
趾部偏移和可达范围，再检查肩胛、躯干与髋部蒙皮过渡，最后用同一地形与侧面镜头复测支撑窗口。

## 复现

这次用 Blender 5.2.4 LTS 与本地四张 RTX A6000 的 OptiX 渲染。已有原片作业缓存时可直接复用。

```sh
bundle=$(cat runs/backyard-final/work-bundle.txt)
blender=.tools/blender-5.2.4-linux-x64/blender
output=runs/realism-revision/validated
MOWER_BLENDER="$blender" .venv/bin/python -m pytest -q
scripts/render_realism_review.sh "$bundle" "$blender" "$output"
.venv/bin/python scripts/measure_realism.py "$bundle" "$output"
```

渲染脚本采用四设备并行。其他平台可按脚本中的帧范围和场景参数逐组调用
`python -m mower_pov render`，并传入实际可用的渲染设备。
本次静帧对照、样片、接触图、比例图与 Blender 求值数据在 `runs/realism-revision/validated/`。

后续验收包括正式蒙皮犬、以动物占用驱动机器刹停和恢复、动态刀盘状态、快门子帧及三段各自独立的
机载验收片。切割、音轨和 40 秒成片的重新生成应在这些验收完成后进行。
