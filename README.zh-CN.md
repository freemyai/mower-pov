# Mower POV

面向家庭后院的割草机器人仿真。固定机载相机记录作业过程，草高和轮迹随作业保存，静态困难场景可批量复测。

[English](README.md)

## 观看 104 秒机载成片

https://github.com/user-attachments/assets/939a06dc-019f-4a18-ab2f-4dfc0f027b0c

五行往返、四次掉头。割草机绕过软管、彩球和浇水壶，返程时可以看到前一趟留下的短草带。

[1080p 原片与校验和](https://github.com/ros-claw/mower-pov/releases/tag/demo%2Fcoverage-104s) · [成片版本与复现](docs/releases/v0.1.0-demo.md)

## 用 CPU 跑静态案例

静态套件有 30 个合成案例，覆盖细软管、掉头区障碍、玩具窄通道、蹦床支架、花坛与边界、漏割与重访六类情况。判定器从实际轨迹独立检查车身连续扫掠、运动限制、刀盘状态和首次割草时间。需要看失败过程时，可用 Blender 渲染该时间段。

从仓库根目录运行，使用 Python 3.11、`uv` 和锁定依赖。

```sh
uv venv --python 3.11
uv pip install -r requirements-lock.txt
uv pip install --no-deps -e .
.venv/bin/python -m mower_pov cases validate scenarios/suites/static-v1.json
.venv/bin/python -m mower_pov cases run scenarios/suites/static-smoke-v1.json \
  --policy legacy-joint --out runs/static-smoke
```

预检运行六个案例，输出 `runs/static-smoke/summary.json` 及逐例判定和证据清单。把运行命令中的配置换成 `static-v1.json` 即可跑全部 30 例。工具还支持断点恢复、策略对照和带绝对时间的失败短片。[案例使用与提交手册](docs/cases/ADDING_CASES.md)

## 本地对照结果

| 评测范围 | 逐行规划 | 联合覆盖规划 |
| --- | ---: | ---: |
| 104 秒演示的五行目标区覆盖率（4.2264 m²） | 72.30% | 85.68% |
| 24 个配对可行案例的面积加权覆盖率 | 61.49% | 79.82% |
| 上述 24 个案例的累计重访面积 | 1.3416 m² | 0.6798 m² |

两套策略均完成了 24 个可行案例，并对 6 个有独立不可行证明的案例安全拒绝。联合覆盖在四个掉头球案例中覆盖率有所下降。静态评测使用已知障碍几何和运动学模型。[指标口径与逐例结果](docs/cases/static-v1-results.zh-CN.md) · [机器可读证据](evidence/v0.2/static-v1-results.json)

## 复现作业与成片

- [104 秒成片制作流程](docs/dynamic-production.md#割草作业区成片)，包括作业缓存、Blender/Cycles 渲染、断点恢复与音轨合成。
- [一期复现证据](evidence/v0.1/reproduction.md)，说明如何从成片与对照提交重建作业缓存并比较覆盖。
- [40 秒动态后院重放](docs/dynamic-production.md)，单独展示小狗横穿与刺猬等待。

源码采用 GPL-3.0-or-later。[代码许可](LICENSE) · [第三方资产归属](THIRD_PARTY_NOTICES.md)
