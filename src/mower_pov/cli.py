"""Command line entry points for local preparation and verification."""

import argparse
import json
from pathlib import Path

from .environment import probe_environment


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mower_pov")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--blender", type=Path)
    doctor.add_argument("--device", default="CPU", choices=["CPU", "OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"])
    doctor.add_argument("--output", type=Path, default=Path("environment.json"))
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=Path("configs/backyard.toml"))
    prepare.add_argument("--runs", type=Path, default=Path("runs"))
    prepare.add_argument("--environment", type=Path)
    prepare.add_argument("--diagnostics", type=Path)
    render = commands.add_parser("render")
    render.add_argument("run_dir", type=Path)
    render.add_argument("--blender", type=Path, default=Path("blender"))
    render.add_argument("--frames", required=True, help="Absolute indices or half-open ranges")
    render.add_argument("--output", type=Path, required=True)
    from .rendering.blender_scene import PRESETS

    render.add_argument("--preset", choices=PRESETS, default="preview")
    render.add_argument("--device", choices=["CPU", "OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"], default="CPU")
    render.add_argument("--gpu-index", type=int, default=0)
    render.add_argument("--undenoised", action="store_true")
    render.add_argument("--exr", action="store_true")
    encode = commands.add_parser("encode-preview")
    encode.add_argument("directories", nargs="+", type=Path)
    encode.add_argument("--frames", required=True, help="A contiguous half-open absolute range")
    encode.add_argument("--fps", type=int, default=30)
    encode.add_argument("--output", required=True, type=Path)
    audio = commands.add_parser("audio")
    audio.add_argument("run_dir", type=Path)
    audio.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "audio":
        from .audio import synthesize_audio
        from .cache import load_bundle

        output = synthesize_audio(load_bundle(args.run_dir), args.output)
        print(f"Rendered audio to {output}")
        return 0
    if args.command == "encode-preview":
        from .rendering.driver import parse_frames
        from .rendering.preview import encode_preview

        report = encode_preview(args.directories, parse_frames(args.frames), args.fps, args.output)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Encoded {report['read_frames']} frames to {args.output}")
        return 0
    if args.command == "render":
        from .rendering.driver import RenderRequest, parse_frames, render_frames

        request = RenderRequest(args.run_dir, parse_frames(args.frames), args.preset,
                                args.output, args.blender, args.device, args.gpu_index,
                                not args.undenoised, args.exr)
        result = render_frames(request)
        print(f"Rendered {len(result.produced_frames)}/{len(request.frame_indices)} frames in {result.output_dir}")
        if result.failed_frames:
            print(f"Failed frames: {result.failed_frames}. See {result.worker_log}")
        return 1 if result.failed_frames else 0
    if args.command == "prepare":
        from .cache import prepare_run, save_bundle
        from .config import load_config

        environment = json.loads(args.environment.read_text()) if args.environment else None
        bundle = prepare_run(load_config(args.config), args.runs, environment)
        path = save_bundle(bundle)
        if args.diagnostics:
            from .diagnostics import write_diagnostics

            write_diagnostics(bundle, args.diagnostics)
        print(f"Bundle: {path}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = probe_environment(args.blender, args.device, args.output.parent)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Cycles: {report['verified_device'] or 'unverified'}. Report: {args.output}")
    for warning in report["warnings"]:
        print(warning)
    return 0 if report["usable"] else 1
