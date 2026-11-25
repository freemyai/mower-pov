"""Profile one reproducible policy run and independent evaluation on CPU."""

import argparse
import cProfile
import json
from pathlib import Path
import pstats
from time import perf_counter

from mower_pov.cases.evaluation import evaluate_episode
from mower_pov.cases.evidence import write_json
from mower_pov.cases.legacy import run_offline
from mower_pov.cases.runner import load_suite, validate_suite
from mower_pov.cases.resolve import source_digest
from mower_pov.policies.legacy import LegacyPlanner, POLICY_IDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", type=Path)
    p.add_argument("case")
    p.add_argument("--policy", choices=POLICY_IDS, default="legacy-joint")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    case = next((c for c in validate_suite(load_suite(args.suite)) if c.id == args.case), None)
    if case is None:
        p.error("case is absent from the specified suite")
    with cProfile.Profile() as profile:
        start = perf_counter()
        trace = run_offline(case, LegacyPlanner(args.policy))
        evaluated = perf_counter()
        verdict = evaluate_episode(case, trace)
        finished = perf_counter()
    stats = pstats.Stats(profile).stats
    selected = {"_row_candidates", "_greedy_row", "obstacle_clearance", "_pose_box_distance",
                "footprint", "select_coverage_sequence", "compute_cut_state", "trace_sweep"}
    hotspots = [
        {"module":Path(filename).name, "function":name, "calls":counts[1],
         "self_s":counts[2], "cumulative_s":counts[3]}
        for (filename, _, name), counts in stats.items() if name in selected
    ]
    report = {"case":case.id, "policy":args.policy, "kernel_sha256":source_digest("kernel"),
              "state_id":trace.state_id, "outcome":verdict.outcome, "passed":verdict.passed,
              "replay_and_plan_s":evaluated-start, "evaluation_s":finished-evaluated,
              "hotspots":sorted(hotspots, key=lambda item: -item["cumulative_s"])}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, report)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
