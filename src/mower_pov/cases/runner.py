"""Ordered, resumable CPU-only static regression suites."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from time import monotonic
import traceback

from ..cache import _code_identity
from ..policies.legacy import LegacyPlanner, POLICY_IDS
from .evaluation import evaluate_episode
from .evidence import atomic_write, hash_file, save_episode, seal, verify_episode, write_json, write_trace
from .legacy import run_offline
from .metrics import EVALUATOR_VERSION
from .resolve import canonical_digest, resolve_case, scenario_digest
from .schema import load_spec, read_json

_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")


@dataclass(frozen=True)
class CaseEntry:
    id: str
    path: Path
    seed: int | None


@dataclass(frozen=True)
class SuiteSpec:
    id: str
    cases: tuple[CaseEntry, ...]


@dataclass(frozen=True)
class RunRequest:
    suite: SuiteSpec
    policy_id: str
    timeout_s: float = 180.
    jobs: int = 1


@dataclass(frozen=True)
class Row:
    id: str
    case_digest: str
    outcome: str
    passed: bool
    state_id: str | None
    target_fraction_cut: float | None
    first_violation_s: float | None
    directory: str


@dataclass(frozen=True)
class SuiteResult:
    suite_id: str
    policy_id: str
    request_id: str
    rows: tuple[Row, ...]
    incomplete: tuple[str, ...]


def load_suite(path):
    path = Path(path).resolve()
    data = read_json(path)
    if (not isinstance(data, dict) or data.keys() != {"version", "id", "cases"}
            or type(data["version"]) is not int or data["version"] != 1
            or not _is_id(data["id"]) or not isinstance(data["cases"], list) or not data["cases"]):
        raise ValueError("invalid v1 suite header")
    cases = []
    for item in data["cases"]:
        if (not isinstance(item, dict) or set(item) not in ({"id", "path"}, {"id", "path", "seed"})
                or not _is_id(item["id"]) or not isinstance(item["path"], str) or not item["path"]):
            raise ValueError("invalid v1 suite case entry")
        file = Path(item["path"])
        if file.is_absolute() or file.suffix != ".json":
            raise ValueError("suite scenario paths must be relative JSON files")
        seed = item.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("suite seed must be an unsigned 63-bit integer")
        cases.append(CaseEntry(item["id"], path.parent/file, seed))
    if len({entry.id for entry in cases}) != len(cases):
        raise ValueError("suite case IDs must be unique")
    return SuiteSpec(data["id"], tuple(cases))


def _is_id(value):
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _case(entry):
    spec = load_spec(entry.path)
    if spec.case.id != entry.id:
        raise ValueError(f"suite case {entry.id!r} differs from declared scenario ID {spec.case.id!r}")
    return resolve_case(spec, spec.case.seed if entry.seed is None else entry.seed)


def validate_suite(suite):
    return tuple(_case(entry) for entry in suite.cases)


def _requests(request):
    policy = LegacyPlanner(request.policy_id)
    sources = {"policy":[policy.id, policy.version], "kernel_sha256":_code_identity(),
               "evaluator":EVALUATOR_VERSION, "timeout_s":request.timeout_s}
    cases = []
    for entry in request.suite.cases:
        try:
            digest = scenario_digest(_case(entry))
            error = None
        except (ValueError, OSError) as exc:
            digest = hash_file(entry.path) if entry.path.is_file() else canonical_digest({"missing":entry.id})
            error = str(exc)
        record = {"id":entry.id, "digest":digest, "seed_override":entry.seed, "valid":error is None}
        record["request_id"] = canonical_digest({"sources":sources, "case":record})
        cases.append((entry, record, error))
    header = {"suite":request.suite.id, "identity":sources, "cases":[record for _, record, _ in cases]}
    return {**header, "request_id":canonical_digest(header)}, cases


def _failed(status, reason):
    return {"outcome":status, "validity":"INVALID_INPUT" if status == "INVALID_INPUT" else "UNDETERMINED",
            "passed":False, "assertions":[], "reason":reason, "metrics":None, "first_violation_s":None}


def _worker(path, identifier, seed, policy, directory, digest):
    entry = CaseEntry(identifier, Path(path), None if seed == "" else int(seed))
    directory = Path(directory)
    try:
        case = _case(entry)
        if scenario_digest(case) != digest:
            raise ValueError("scenario changed between request and execution")
        data = read_json(entry.path)
        data["seed"] = case.seed
        write_json(directory/"case.json", data)
        trace = run_offline(case, LegacyPlanner(policy))
        # Persist actual state before independent evaluation so a judge crash is inspectable.
        write_trace(trace, directory)
        verdict = evaluate_episode(case, trace)
        save_episode(trace, verdict, directory)
    except Exception as exc:
        atomic_write(directory/"traceback.txt", lambda out: out.write(traceback.format_exc()))
        write_json(directory/"verdict.json", _failed("CRASH", str(exc)))
    return 0


def _run_task(out, request, task):
    entry, record, error = task
    target = out/"episodes"/entry.id
    if target.exists():
        verify_episode(target, record["request_id"])
        return
    begin = monotonic()
    staging = Path(tempfile.mkdtemp(prefix=f".{entry.id}-", dir=target.parent))
    try:
        if error:
            write_json(staging/"verdict.json", _failed("INVALID_INPUT", error))
        else:
            args = [sys.executable, "-m", "mower_pov.cases.runner", str(entry.path), entry.id,
                    "" if entry.seed is None else str(entry.seed), request.policy_id, str(staging),
                    record["digest"]]
            try:
                run = subprocess.run(args, capture_output=True, timeout=request.timeout_s, check=False)
                if run.returncode or not (staging/"verdict.json").is_file():
                    (staging/"stderr.log").write_bytes(run.stderr)
                    write_json(staging/"verdict.json", _failed("CRASH", f"worker exit {run.returncode}"))
            except subprocess.TimeoutExpired as exc:
                if exc.stderr:
                    (staging/"stderr.log").write_bytes(exc.stderr)
                write_json(staging/"verdict.json", _failed("TIMEOUT", f"worker exceeded {request.timeout_s}s"))
        # Remove temporary half-files left by a killed worker before signing.
        for partial in staging.glob(".*"):
            partial.unlink()
        seal(staging, record["request_id"])
        os.rename(staging, target)
        write_json(out/"timings"/f"{entry.id}.json", {"wall_s":monotonic() - begin})
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _read_result(out, header):
    rows, incomplete = [], []
    for case in header["cases"]:
        path = out/"episodes"/case["id"]
        if not path.is_dir():
            incomplete.append(case["id"])
            continue
        verify_episode(path, case["request_id"])
        result = read_json(path/"verdict.json")
        trace = read_json(path/"trace.json") if (path/"trace.json").is_file() else {}
        rows.append(Row(case["id"], case["digest"], result["outcome"], result["passed"],
                        trace.get("state_id"), (result["metrics"] or {}).get("target_fraction_cut"),
                        result["first_violation_s"], str(path.relative_to(out))))
    return SuiteResult(header["suite"], header["identity"]["policy"][0], header["request_id"],
                       tuple(rows), tuple(incomplete))


def read_suite_result(output_dir, request):
    header, _ = _requests(request)
    return _read_result(Path(output_dir), header)


def resume_suite(output_dir, request):
    if type(request.jobs) is not int or request.jobs < 1 or not math.isfinite(request.timeout_s) or request.timeout_s <= 0:
        raise ValueError("jobs and timeout must be positive")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    header, tasks = _requests(request)
    with (out/".suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request_file = out/"request.json"
        if request_file.exists():
            if read_json(request_file) != header:
                raise ValueError("output directory belongs to a different request")
        else:
            write_json(request_file, header)
        (out/"episodes").mkdir(exist_ok=True)
        (out/"timings").mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=request.jobs) as workers:
            list(workers.map(lambda task: _run_task(out, request, task), tasks))
        result = _read_result(out, header)
        write_json(out/"summary.json", asdict(result))
    return result


def run_suite(suite, policy_id, output_dir, *, timeout_s=180., jobs=1):
    return resume_suite(output_dir, RunRequest(suite, policy_id, timeout_s, jobs))


def compare_suites(baseline, candidate):
    first, second = [read_json(Path(path)/"request.json") for path in (baseline, candidate)]
    case_ids = lambda request: {case["id"]:case["digest"] for case in request["cases"]}
    if case_ids(first) != case_ids(second):
        raise ValueError("cannot compare suites with different case identities")
    left, right = _read_result(Path(baseline), first), _read_result(Path(candidate), second)
    left_rows, right_rows = [{row.id:asdict(row) for row in result.rows} for result in (left, right)]
    return {"baseline_policy":left.policy_id, "candidate_policy":right.policy_id,
            "cases":[{"id":name,"baseline":left_rows.get(name),"candidate":right_rows.get(name)}
                     for name in case_ids(first)]}


def add_case_commands(commands):
    cases = commands.add_parser("cases", help="CPU-only static scenario regression")
    actions = cases.add_subparsers(dest="cases_command", required=True)
    validate = actions.add_parser("validate")
    validate.add_argument("suite", type=Path)
    run = actions.add_parser("run", help="run or resume an exact request")
    run.add_argument("suite", type=Path)
    run.add_argument("--policy", choices=POLICY_IDS, default="legacy-joint")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--no-render", action="store_true", help="compatibility flag; all batch runs are numeric")
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--timeout-s", type=float, default=180.)
    compare = actions.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--out", type=Path)


def case_command(args):
    if args.cases_command == "validate":
        cases = validate_suite(load_suite(args.suite))
        print(f"{len(cases)} valid static scenarios")
        return 0
    if args.cases_command == "compare":
        comparison = compare_suites(args.baseline, args.candidate)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.out, comparison)
        else:
            print(json.dumps(comparison, indent=2))
        return 0
    result = run_suite(load_suite(args.suite), args.policy, args.out, timeout_s=args.timeout_s, jobs=args.jobs)
    passing = sum(row.passed for row in result.rows)
    print(f"{passing}/{len(result.rows) + len(result.incomplete)} pass. {args.out/'summary.json'}")
    return 0 if not result.incomplete and passing == len(result.rows) else 1


if __name__ == "__main__":
    raise SystemExit(_worker(*sys.argv[1:]))
