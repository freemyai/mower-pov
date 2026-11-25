# 静态 corner case 使用与提交说明

`static-v1` 是 30 个固定种子的开发回归集。六个场景族覆盖软管、掉头附近的球、玩具窄通道、蹦床支架投影、花坛与庭院边缘，以及遮挡造成的漏割和重访。执行模式为 `offline_plan_replay`。策略从同一份已知平面地图规划，数值重放使用 240 Hz 时钟。

## 安装与运行

在仓库根目录使用 Python 3.11.15，按锁安装依赖。

```bash
uv venv --python 3.11.15
uv pip install -r requirements-lock.txt
uv pip install --no-deps -e .
python=.venv/bin/python
"$python" -m mower_pov cases validate scenarios/suites/static-v1.json
"$python" -m mower_pov cases run scenarios/suites/static-v1.json \
  --policy legacy-greedy --no-render --jobs 4 --out runs/cases-greedy
"$python" -m mower_pov cases run scenarios/suites/static-v1.json \
  --policy legacy-joint --no-render --jobs 4 --out runs/cases-joint
"$python" -m mower_pov cases compare runs/cases-greedy runs/cases-joint \
  --out runs/cases-comparison.json
```

批跑使用 CPU，遇到失败或缺例时返回非零。使用同一输出路径和请求恢复。坏文件会报错并保留现场，策略或请求变化时使用新目录。`request.json` 记录策略、内核、判定器、用例和请求身份。`summary.json` 汇总全部案例，逐例 `verdict.json` 记录通过条件、首个问题时间与覆盖明细。计时单独写入 `timings/`。

`scenario_digest` 标识完整场景与验收条件。`state_id` 绑定策略、数值内核和实际轨迹、刀盘、草坪及事件。只改相机时数值状态标识不变。渲染请求另外绑定镜头、Blender 和场景表现。比较工具要求两次作业的场景标识一致。

## 增加一个案例

1. 复制同族的一个 [示例](../../scenarios/static/S01-01-thin-diagonal.json)。使用 [v1 场景契约](../../schemas/scenario-v1.schema.json) 校验，并为不同几何或预算使用稳定唯一 ID。
2. 固定种子，使用米、秒、弧度。作业行由共同的左右端点和从北到南的等距行中心定义。初始姿态为第一行左端、朝正 X。场景约束来自世界边界、可选矩形花坛及经过指定半径膨胀的点或折线。蹦床的支腿和横杆只使用保守的二维投影。
3. 先写预期行为和断言。可行案例给出独立的参考通路及时间检查。期望拒绝时用时间预算下界、必经点不可达或完整横断面封闭之类的证明。仅搜索不到候选会记录具体失败原因。
4. 用显式、有序的 suite 引用 JSON。参考路径与期望只供测试和判定器使用。保留失败用例，生成 `summary.json` 后查看各断言，修复时用同一场景身份重跑。

覆盖目标是名义刀盘扫掠与可割草地的交集，保留静态道具净距范围内的草，排除配置的花坛。通过条件可设置 `minimum_target_coverage`。几何净距、边界、转速、轮速与实际位姿、首割记录及完成必经点逐项检查。代码把底盘位置与航向分别插值，前进、侧向速度与姿态角速度对照允许 0.001 m/s、0.010 m/s 和 0.150 rad/s 采样残差。配置的速度、角速度上限另行检查。

## 数值收敛和失败证据

```bash
.venv/bin/python scripts/check_case_convergence.py runs/cases-joint S01-01-thin-diagonal \
  --out runs/convergence-s01.json
.venv/bin/python scripts/benchmark_cases.py scenarios/suites/static-v1.json S06-03-two-obstacles \
  --out runs/profile-s06.json
```

收敛报告用 10、5、2 mm 栅格复算面积。连续底盘扫掠仍使用独立的姿态间运动上界。对已保存实际轨迹的案例，以下命令重放首次失败点前后各两秒，同时保存干净机载版、调试版和清单。通过的案例可指定 `--at-s` 查看代表时段。

```bash
.venv/bin/python -m mower_pov cases render runs/cases-joint CASE_ID \
  --out runs/failure-replay --blender /path/to/blender
```

录像保留绝对时间和帧号，默认为 640×360、无音轨。道具呈现静态判定采用的简化几何。安全拒绝及无可执行轨迹的案例保留数值证据。使用 [缺陷模板](../../.github/ISSUE_TEMPLATE/corner-case.yml) 提交用例和输出。

本集为合成开发回归数据。相机感知、在线动态响应、真实动物保护距离和外部独立测试属于后续验证范围。
