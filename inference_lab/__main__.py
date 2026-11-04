"""Explicit offline defaults; remote requests always require an execution switch."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    from .benchmark import load_workload, run_batch, compare_release
    from .fixture import FixtureServer, original_cases
    from .protocol import Endpoint
    from .report import render, write_json
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Only known-answer loopback requests and actual tiny CPU attention")
    demo.add_argument("--out", type=Path, default=project / "results")
    doctor = sub.add_parser("doctor", help="Read available Torch/CUDA/NVIDIA tooling; no provider requests")
    doctor.add_argument("--out", type=Path)
    kv = sub.add_parser("kv", help="Run tiny attention equivalence/timing on the requested device")
    kv.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    kv.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    kv.add_argument("--out", type=Path, default=project / "runs" / "attention.json")
    kv.add_argument("--prefix", type=int, default=64)
    kv.add_argument("--decode", type=int, default=24)
    kv.add_argument("--repeats", type=int, default=7)
    bench = sub.add_parser("endpoint", help="Measure an explicitly configured OpenAI-compatible endpoint")
    bench.add_argument("--provider", choices=["vllm", "nebius"], required=True)
    bench.add_argument("--execute-remote", action="store_true")
    bench.add_argument("--workload", type=Path, default=project / "data" / "workload.jsonl")
    bench.add_argument("--out", type=Path, default=project / "runs" / "endpoint")
    bench.add_argument("--concurrency", type=int, default=2)
    bench.add_argument("--timeout", type=float, default=30)
    bench.add_argument("--phase", choices=["uncontrolled", "cold", "warm"], default="uncontrolled")
    bench.add_argument("--input-per-million", type=float)
    bench.add_argument("--output-per-million", type=float)
    bench.add_argument("--dedicated-hourly", type=float)
    report = sub.add_parser("report", help="Rebuild a narrative from an existing metrics artifact")
    report.add_argument("--out", type=Path, default=project / "results")
    compare = sub.add_parser("compare", help="Apply declared release gates to compatible artifacts")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        render(args.out)
        return
    if args.command == "compare":
        baseline = json.loads(args.baseline.read_text())
        candidate = json.loads(args.candidate.read_text())
        for obj in (baseline, candidate):
            obj["endpoint_run"]["workload_sha256"] = obj["workload_sha256"]
            obj["endpoint_run"]["execution_kind"] = obj["execution_kind"]
        print(json.dumps(compare_release(baseline["endpoint_run"], candidate["endpoint_run"]), indent=2))
        return
    if args.command in {"doctor", "kv", "demo"}:
        from .kv import doctor as inspect, kv_benchmark
    if args.command == "doctor":
        data = inspect()
        if args.out:
            write_json(args.out, data)
        print(json.dumps(data, indent=2))
        return
    if args.command == "kv":
        data = kv_benchmark(args.device, args.dtype, args.prefix, args.decode, args.repeats)
        write_json(args.out, data)
        print(json.dumps(data, indent=2))
        return
    if args.command == "demo":
        workload = project / "data" / "workload.jsonl"
        cases = load_workload(workload)
        with FixtureServer(cases) as fixture:
            result = run_batch(cases, fixture.endpoint, input_per_million=1, output_per_million=2, dedicated_hourly=1)
        data = {"execution_kind": "loopback_protocol_fixture", "scope": "new_independent_reference_implementation",
                "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
                "phase": "uncontrolled", "cache_reset_or_hit_verified": False,
                "endpoint_run": result, "attention_run": kv_benchmark(), "doctor": inspect()}
        data["doctor"]["vllm_endpoint_status"] = "not_run"
        data["doctor"]["nebius_endpoint_status"] = "not_run"
    else:
        prefix = args.provider.upper()
        default_url = "http://127.0.0.1:8000/v1" if args.provider == "vllm" else "https://api.tokenfactory.nebius.com/v1"
        base_url = os.environ.get(prefix + "_BASE_URL", default_url)
        model = os.environ.get(prefix + "_MODEL", "")
        key = os.environ.get(prefix + "_API_KEY", "")
        if not model or args.provider == "nebius" and not key:
            parser.error("Configure the provider model and required API key in environment variables; values are never printed")
        endpoint = Endpoint(args.provider, base_url, model, key)
        cases = load_workload(args.workload)
        result = run_batch(cases, endpoint, concurrency=args.concurrency, timeout_s=args.timeout,
                           execute_remote=args.execute_remote, input_per_million=args.input_per_million,
                           output_per_million=args.output_per_million, dedicated_hourly=args.dedicated_hourly)
        data = {"execution_kind": "configured_endpoint", "scope": "new_independent_reference_implementation",
                "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
                "phase": args.phase, "cache_reset_or_hit_verified": False,
                "endpoint_run": result, "attention_run": None,
                "doctor": {"hardware": "not_collected_by_client", "provider_run": args.provider}}
    write_json(args.out / "metrics.json", data)
    (args.out / "traces.jsonl").write_bytes("".join(json.dumps(t, sort_keys=True) + "\n" for t in result["traces"]).encode())
    render(args.out)
    print(json.dumps({"execution_kind": data["execution_kind"], "requests": result["requests"],
                      "quality_successes": result["quality_successes"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
