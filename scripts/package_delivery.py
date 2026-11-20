"""Package the local finished films, evidence, work cache, and exact source snapshots."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

README = """# Mower POV · 午后庭院

- 主片为 40 秒、1920×1080、30 fps、立体声固定机载视角。
- 花絮为 18 秒、1280×720、30 fps、立体声，使用独立庭院相机。
- 两条短样片分别展示刺猬，以及小狗追球、掉头和返程短草带。
- 质量报告区分机器检查、画面抽查和试听状态。

work-bundle 包含实际作业缓存，按绝对世界时间重放。code 中有当前项目和主片渲染时
固定的源码版本。若花絮单独固定源码，则另附其 Python 快照。运行命令、锁定依赖和依赖许可见各源码包内的 README 和 docs。
SHA256SUMS 列出交付文件的校验值。线性 EXR 和 PNG 帧序列留在本地制作目录。
"""


def _digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def package(main, showcase, review, output):
    root = Path.cwd()
    qa = json.loads((main / "feature-qa.json").read_text())
    if not all(qa["machine_checks"].values()):
        raise ValueError("machine QA must pass before creating the local delivery package")
    showcase_metadata = json.loads((showcase / "garden-extras-18s.json").read_text())
    if (showcase_metadata["frames"], showcase_metadata["fps"]) != (540, 30):
        raise ValueError("showcase metadata does not match the 18-second delivery")
    work = Path((main / "work-bundle.txt").read_text().strip())
    files = {}
    for folder, names in (
        (main, ("mower-pov-40s.mp4", "backyard-40s.wav", "mower-pov-cover.jpg", "feature-contact.jpg")),
        (showcase, ("garden-extras-18s.mp4",)),
        (review, ("hedgehog-4s-with-sound.mp4", "dog-and-return-8s-with-sound.mp4")),
    ):
        files.update({f"media/{name}": folder / name for name in names})
    for entry in sorted(work.iterdir()):
        if entry.is_file():
            files[f"work-bundle/{entry.name}"] = entry
    for name in ("mower-pov-40s.json", "feature-qa.json", "pytest.txt", "source-commit.txt"):
        files[f"evidence/{name}"] = main / name
    files["evidence/garden-extras-18s.json"] = showcase / "garden-extras-18s.json"
    for section, directory in (("production", main), ("showcase", showcase)):
        for batch in sorted(directory.glob("batch-*")):
            for name in ("frames.jsonl", "request.json"):
                files[f"evidence/{section}/{batch.name}/{name}"] = batch / name

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".package-", dir=output.parent) as staging:
        staging = Path(staging)
        readme = staging / "README.md"
        readme.write_text(README)
        files["README.md"] = readme
        for name, revision in (("project", "HEAD"), ("production", qa["render"]["source_commit"])):
            snapshot = staging / f"{name}.zip"
            subprocess.run(["git", "-C", str(root), "archive", "--format=zip",
                            f"--prefix=mower-pov-{name}/", "-o", str(snapshot), revision], check=True)
            files[f"code/mower-pov-{name}.zip"] = snapshot
        montage_source = showcase.parent / "source" / "src"
        if montage_source.is_dir():
            snapshot = staging / "showcase.zip"
            with ZipFile(snapshot, "w", compression=ZIP_DEFLATED) as archive:
                for path in sorted(montage_source.rglob("*.py")):
                    archive.write(path, f"mower-pov-showcase/src/{path.relative_to(montage_source)}")
            files["code/mower-pov-showcase.zip"] = snapshot
        partial = staging / "delivery.zip"
        sums = "\n".join(f"{_digest(path)}  {name}" for name, path in sorted(files.items())) + "\n"
        with ZipFile(partial, "w", compression=ZIP_DEFLATED, compresslevel=5) as archive:
            for name, path in sorted(files.items()):
                method = ZIP_STORED if path.suffix in {".mp4", ".jpg", ".npz", ".zip"} else ZIP_DEFLATED
                archive.write(path, f"mower-pov-local/{name}", compress_type=method)
            archive.writestr("mower-pov-local/SHA256SUMS.txt", sums)
        with ZipFile(partial) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"damaged archive member: {bad}")
        os.replace(partial, output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production", type=Path)
    parser.add_argument("showcase", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package(args.production, args.showcase, args.review, args.output)
