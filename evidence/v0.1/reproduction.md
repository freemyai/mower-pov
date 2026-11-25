# 一期基线复现与发布证据

[版本清单](manifest.json)记录用来复算的历史提交、原始制品与本次验证。附件 `release-manifest.json` 绑定整理提交及附件校验和。附件现存于本地，公开下载入口以实际发布结果为准。

## 1. CPU 数值预检

从结题文档所在仓库根目录运行。实测环境为 Linux x86-64、Python 3.11.15、NumPy 2.2.6、pytest 9.0.3、FFmpeg 7.1.1 和 Blender 5.2.4 LTS 构建 `cd5dc428cce1`。这一节只需要 Python。

```bash
uv venv --python 3.11.15
uv pip install -r requirements-lock.txt
uv pip install --no-deps -e .
.venv/bin/python -m pytest -q \
  tests/test_demo_golden.py tests/test_cutting.py tests/test_camera.py tests/test_cache.py
```

[解析夹具](golden-straight-cut.json)固定半径 0.14 m。前向扫掠时，`(0.5, 0)` 草根的首割时刻为 `(0.5 − 0.14) / 1 = 0.36 s`，偏离中心线 0.16 m 的草根始终保持原始高度。JSON 的 `null` 在测试中对应正无穷。测试检查返程草高和相机平移前后完整首割数组。夹具用构造轨迹隔离扫掠计算，整机限速另由作业测试验收。
[夹具测试](../../tests/test_demo_golden.py)

## 2. 干净检出与指标重算

以下在 Bash 中运行。将 `blender` 指向已安装的固定构建。通过 `PYTHONPATH` 指定每个工作树的源码，实现同依赖环境下的独立执行。省略 `--environment` 可复现原缓存的工具链身份，实测环境单独留档。

```bash
repo="$PWD"
python="$repo/.venv/bin/python"
blender="$repo/.tools/blender-5.2.4-linux-x64/blender"
run="$(mktemp -d)"

git worktree add --detach "$run/baseline" f577518db22a1e5b85c6fa9cbf89b92ef8b27d35
git worktree add --detach "$run/film" d6300c6ad5ef5f81d8604d671e41936585f48f84
git worktree add --detach "$run/patched" 3b744cb5b25810b039da4fba92a637d4a943a0f7
set -o pipefail
for version in baseline film patched; do
  (
    cd "$run/$version"
    PYTHONPATH="$PWD/src" "$python" -m mower_pov prepare \
      --config configs/mowing-course.toml --runs "$run/state-$version" \
      --diagnostics "$run/diagnostics-$version" | tee "$run/prepare-$version.log"
  ) || exit 1
done
baseline="$(awk '/^Bundle: / {print substr($0,9)}' "$run/prepare-baseline.log")"
film="$(awk '/^Bundle: / {print substr($0,9)}' "$run/prepare-film.log")"
(
  cd "$run/film"
  export PYTHONPATH="$PWD/src"
  "$python" scripts/compare_coverage.py "$baseline" "$film" "$run/coverage"
  "$python" scripts/measure_course.py "$film" "$run/course"
)
for version in film patched; do
  (
    cd "$run/$version"
    PYTHONPATH="$PWD/src" MOWER_BLENDER="$blender" "$python" -m pytest -q
  ) | tee "$run/tests-$version.log" || exit 1
done
```

两个原始缓存也可从 `baseline-state.tar` 和 `film-state.tar` 解出，包内保留原清单、规范化配置、四个 NPZ 数据文件和事件。重算时核对 `identity.arrays`、`events_sha256` 和 `course_layout`。缓存准备时的 Git HEAD 属于历史记录，渲染提交和源码指纹共同确认视频版本。

## 3. 媒体与三帧预检

```bash
(
  cd "$run/film"
  export PYTHONPATH="$PWD/src" MOWER_PYTHON="$python" MOWER_DEVICE=CPU
  scripts/render_dynamic_feature.sh course-review work-area \
    "$film" "$blender" "$run/smoke" 0:3 --preset smoke
)
```

本次基于原始成片缓存，用成片提交在 CPU 完成同样的三帧渲染、编码及解码。[全片制作与恢复](../../docs/dynamic-production.md#割草作业区成片)

在附件目录校验媒体和制品哈希。

```bash
sha256sum -c SHA256SUMS
video=mower-pov-mowing-course.mp4
ffprobe -v error -show_streams -show_format "$video"
ffmpeg -v error -xerror -i "$video" -map 0:v:0 -map 0:a:0 -f null -
```

人工审看记录应写明媒体 SHA-256、观看范围、片段时间及观察到的问题。试听记录覆盖轮声、刀盘声和切割负载。完整人工审看及试听均待登记。

## 4. 制品与迁移

| 制品 | 用途 |
| --- | --- |
| 成片、联系表、三个 MP4 样片 | 一期演示、绕行、换行与返程作业 |
| `baseline-state.tar` 和 `film-state.tar` | 原作业缓存，已解包并通过缓存读取校验 |
| `verification-evidence.tar` | 原测试与复测日志、覆盖对照、运控报告、播放校验、CPU 小样片 |
| `release-manifest.json` 和 `SHA256SUMS` | 发布包提交与附件校验值 |

生产报告的 `runs/coverage-optimized/` 和 `runs/mowing-course/` 属于本地工作区。Git 收录源码、配置和小型证据，原始中间帧及私有备份留在本地。对外准备的报告附件使用可移植路径。

源仓库 ID 为 `1466523025`。受认证账号拥有源仓库管理权限及目标组织有效管理员身份，目标组织可见仓库清单中未发现同名库。原生转移前，由授权管理员在 GitHub 界面核对组织策略与冲突 fork。建议 `demo/coverage-104s` 指向成片提交，`v0.1.0-demo` 指向结题提交。转移后核对仓库 ID、各引用和访问权限，再使用实际发布下载地址验证附件 SHA-256，并从新地址干净检出运行数值和短样片验收。
