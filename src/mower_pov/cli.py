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
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = probe_environment(args.blender, args.device, args.output.parent)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Cycles: {report['verified_device'] or 'unverified'}. Report: {args.output}")
    for warning in report["warnings"]:
        print(warning)
    return 0 if report["usable"] else 1
