"""Reevaluate a verified trace at 10 mm, 5 mm and 2 mm cell sizes."""

import argparse
import json
from pathlib import Path

from mower_pov.cases.evidence import load_episode, write_json
from mower_pov.cases.metrics import convergence_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("case")
    p.add_argument("--area-tolerance-m2", type=float, default=.01)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    case, trace, _ = load_episode(args.run_dir, args.case)
    report = convergence_report(case, trace, (.01, .005, .002),
                                area_tolerance_m2=args.area_tolerance_m2)
    report.update(case=case.id, state_id=trace.state_id)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, report)
    else:
        print(json.dumps(report, indent=2))
    return 0 if report["status"] == "STABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
