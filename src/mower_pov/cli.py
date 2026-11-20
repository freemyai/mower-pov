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
    args = parser.parse_args(argv)
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
