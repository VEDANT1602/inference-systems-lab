"""Bounded closed-loop batches, quality gates and assumption-driven cost math."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

from .protocol import Endpoint, stream_chat


def load_workload(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict) or set(case) != {"case_id", "messages", "expected_label", "prefix_group", "max_tokens"}:
            raise ValueError("Unexpected workload schema")
        if not isinstance(case["case_id"], str) or not 1 <= len(case["case_id"]) <= 128:
            raise ValueError("Invalid case identifier")
        if not isinstance(case["prefix_group"], str) or not 1 <= len(case["prefix_group"]) <= 128:
            raise ValueError("Invalid prefix group")
        if not isinstance(case["expected_label"], str) or case["expected_label"] not in {"billing", "technical", "account"}:
            raise ValueError("Invalid expected label")
        if type(case["max_tokens"]) is not int or not 1 <= case["max_tokens"] <= 1024:
            raise ValueError("Invalid max_tokens")
        if not isinstance(case["messages"], list) or not case["messages"]:
            raise ValueError("Messages are required")
        for message in case["messages"]:
            if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"}:
                raise ValueError("Invalid message schema")
            if not isinstance(message["content"], str) or len(message["content"]) > 50000:
                raise ValueError("Invalid message content")
        cases.append(case)
    if not cases or len(cases) > 1000 or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Workload must contain 1..1000 unique case identifiers")
    return cases


def quantile(values: list[float], q: float) -> float | None:
    if not 0 <= q <= 1:
        raise ValueError("Quantile must be between zero and one")
    if not values:
        return None
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Quantiles require finite observations")
    ordered = sorted(values)
    x = (len(ordered) - 1) * q
    low, high = math.floor(x), math.ceil(x)
    return ordered[low] + (ordered[high] - ordered[low]) * (x - low)


def token_cost(usage: dict | None, input_per_million: float, output_per_million: float) -> float | None:
    if any(not math.isfinite(p) or p < 0 for p in (input_per_million, output_per_million)):
        raise ValueError("Rates must be finite nonnegative assumptions")
    if usage is None:
        return None
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
        raise ValueError("Usage must contain nonnegative integer token counts")
    return (usage["prompt_tokens"] * input_per_million + usage["completion_tokens"] * output_per_million) / 1e6


def economic_break_even(hourly: float | None, mean_request_cost: float | None) -> float | None:
    if hourly is not None and (not math.isfinite(hourly) or hourly < 0):
        raise ValueError("Dedicated hourly cost must be finite and nonnegative")
    if mean_request_cost is not None and (not math.isfinite(mean_request_cost) or mean_request_cost < 0):
        raise ValueError("Mean request cost must be finite and nonnegative")
    if hourly is None or mean_request_cost is None or mean_request_cost == 0:
        return None
    return hourly / (3600 * mean_request_cost)


def run_batch(cases: list[dict], endpoint: Endpoint, *, concurrency: int = 2, timeout_s: float = 5,
              execute_remote: bool = False, fallback: Endpoint | None = None,
              ttft_slo_ms: float = 1000, e2e_slo_ms: float = 5000,
              input_per_million: float | None = None, output_per_million: float | None = None,
              dedicated_hourly: float | None = None, fallback_input_per_million: float | None = None,
              fallback_output_per_million: float | None = None) -> dict:
    if type(concurrency) is not int or not 1 <= concurrency <= 32 or not cases:
        raise ValueError("Use a nonempty workload and concurrency 1..32")
    if (input_per_million is None) != (output_per_million is None):
        raise ValueError("Specify both token rates or neither")
    if input_per_million is not None:
        token_cost(None, input_per_million, output_per_million)
    if (fallback_input_per_million is None) != (fallback_output_per_million is None):
        raise ValueError("Specify both fallback token rates or neither")
    if fallback_input_per_million is not None:
        token_cost(None, fallback_input_per_million, fallback_output_per_million)
    if fallback and input_per_million is not None and fallback_input_per_million is None:
        raise ValueError("Priced fallback runs require separate fallback rates")
    if dedicated_hourly is not None and (not math.isfinite(dedicated_hourly) or dedicated_hourly < 0):
        raise ValueError("Dedicated hourly cost must be finite and nonnegative")
    if any(not math.isfinite(s) or s <= 0 for s in (ttft_slo_ms, e2e_slo_ms)):
        raise ValueError("SLOs must be positive and finite")
    if endpoint.remote and not execute_remote or fallback and fallback.remote and not execute_remote:
        raise ValueError("Remote execution is disabled")
    began = time.perf_counter()

    def one(case):
        worker_start = time.perf_counter()
        attempts = [stream_chat(endpoint, case, timeout_s=timeout_s, execute_remote=execute_remote)]
        if fallback and not attempts[-1]["quality_pass"]:
            attempts.append(stream_chat(fallback, case, timeout_s=timeout_s, execute_remote=execute_remote))
        final = attempts[-1]
        elapsed = (time.perf_counter() - worker_start) * 1000
        known_costs = []
        if input_per_million is not None:
            known_costs = [token_cost(a["usage"], input_per_million if i == 0 else fallback_input_per_million,
                                      output_per_million if i == 0 else fallback_output_per_million)
                           for i, a in enumerate(attempts)]
        first_content = next((sum(a["e2e_ms"] for a in attempts[:i]) + a["ttft_proxy_ms"]
                              for i, a in enumerate(attempts) if a["ttft_proxy_ms"] is not None), None)
        # Accepted-answer timing includes a rejected primary attempt, not its discarded first content.
        accepted_ttft = (sum(a["e2e_ms"] for a in attempts[:-1]) + final["ttft_proxy_ms"]
                         if final["quality_pass"] and final["ttft_proxy_ms"] is not None else None)
        return {"case_id": case["case_id"], "expected_label": case["expected_label"],
                "prompt_sha256": hashlib.sha256(json.dumps(case["messages"], sort_keys=True).encode()).hexdigest(),
                "quality_pass": final["quality_pass"], "transport_success": final["transport_success"],
                "attempts": attempts, "client_queue_ms": (worker_start - began) * 1000,
                "e2e_ms": elapsed, "e2e_including_queue_ms": (time.perf_counter() - began) * 1000,
                "first_content_any_attempt_ms": first_content,
                "accepted_ttft_proxy_ms": accepted_ttft,
                "slo_pass": bool(final["quality_pass"] and accepted_ttft is not None and
                                 accepted_ttft <= ttft_slo_ms and elapsed <= e2e_slo_ms),
                "known_attempt_cost": sum(x for x in known_costs if x is not None) if known_costs else None,
                "all_attempt_usage_known": bool(known_costs) and all(x is not None for x in known_costs)}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        traces = list(executor.map(one, cases))
    wall = time.perf_counter() - began
    good = [t for t in traces if t["quality_pass"]]
    all_cost_known = all(t["all_attempt_usage_known"] for t in traces)
    total_cost = sum(t["known_attempt_cost"] for t in traces) if all_cost_known else None
    mean_cost = total_cost / len(traces) if total_cost is not None else None
    result = {"protocol": "bounded_closed_loop_finite_batch", "requests": len(cases),
              "concurrency": concurrency, "wall_seconds": wall,
              "transport_successes": sum(t["transport_success"] for t in traces),
              "quality_successes": len(good), "quality_success_rate": len(good) / len(cases),
              "slo_successes": sum(t["slo_pass"] for t in traces),
              "quality_goodput_requests_per_second": len(good) / wall,
              "slo_goodput_requests_per_second": sum(t["slo_pass"] for t in traces) / wall,
              "attempts": sum(len(t["attempts"]) for t in traces),
              "errors_by_type": {},
              "ttft_proxy_successes_ms": {"p50": quantile([t["accepted_ttft_proxy_ms"] for t in good], .5),
                                          "p95": quantile([t["accepted_ttft_proxy_ms"] for t in good], .95)},
              "e2e_all_requests_ms": {"p50": quantile([t["e2e_ms"] for t in traces], .5),
                                       "p95": quantile([t["e2e_ms"] for t in traces], .95)},
              "inter_token_latency_ms": None,
              "cost": {"input_per_million_assumption": input_per_million,
                       "output_per_million_assumption": output_per_million,
                       "fallback_input_per_million_assumption": fallback_input_per_million,
                       "fallback_output_per_million_assumption": fallback_output_per_million,
                       "billing_model": "manual_uncached_input_output_rates_not_an_invoice",
                       "dedicated_hourly_assumption": dedicated_hourly,
                       "all_attempt_usage_known": all_cost_known, "total_cost": total_cost,
                       "cost_per_quality_success": total_cost / len(good) if total_cost is not None and good else None,
                       "economic_break_even_requests_per_second": economic_break_even(dedicated_hourly, mean_cost),
                       "capacity_and_slo_feasibility": "not_established"},
              "traces": traces}
    for trace in traces:
        for attempt in trace["attempts"]:
            if attempt["error"]:
                result["errors_by_type"][attempt["error"]] = result["errors_by_type"].get(attempt["error"], 0) + 1
    return result


def compare_release(baseline: dict, candidate: dict, *, minimum_quality: float = .95,
                    maximum_p95_e2e_ratio: float = 1.1) -> dict:
    if not 0 <= minimum_quality <= 1 or not math.isfinite(maximum_p95_e2e_ratio) or maximum_p95_e2e_ratio <= 0:
        raise ValueError("Invalid release gate thresholds")
    if baseline.get("workload_sha256") != candidate.get("workload_sha256") or not baseline.get("workload_sha256"):
        raise ValueError("Release comparison requires the same nonempty workload hash")
    if baseline["requests"] != candidate["requests"]:
        raise ValueError("Release comparison requires identical request denominators")
    if baseline["concurrency"] != candidate["concurrency"]:
        raise ValueError("Release comparison requires identical concurrency")
    if baseline.get("execution_kind") != candidate.get("execution_kind"):
        raise ValueError("Cannot compare fixture and provider executions")
    checks = {"quality": candidate["quality_success_rate"] >= minimum_quality,
              "no_quality_regression": candidate["quality_success_rate"] >= baseline["quality_success_rate"],
              "p95_latency": candidate["e2e_all_requests_ms"]["p95"] <=
              maximum_p95_e2e_ratio * baseline["e2e_all_requests_ms"]["p95"]}
    return {"decision": "candidate_passes_declared_gates" if all(checks.values()) else "hold_candidate",
            "checks": checks, "scope": "fixed workload gate; not proof of population superiority"}
