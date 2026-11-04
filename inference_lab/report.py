"""Generate the narrative from the measured artifact, not from separate arithmetic."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def link(target: Path, directory: Path):
    try:
        return os.path.relpath(target.resolve(), directory.resolve()).replace(os.sep, "/")
    except ValueError:
        return target.resolve().as_uri()


def number(value, places=3):
    return "N/A" if value is None else f"{value:.{places}f}"


def render(output: Path):
    data = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    protocol, kv = data.get("endpoint_run"), data.get("attention_run")
    executed_protocol = protocol is not None
    fixture = data["execution_kind"] == "loopback_protocol_fixture"
    headline = "The measurement contract works; GPU and provider performance remain unmeasured" if fixture else "Inspect quality and failure denominators before selecting a serving configuration"
    paragraph = "The loopback server emits known answers from a lookup table. Its timings exercise the client and cannot rank vLLM, Nebius or GPUs." if fixture else "This run sent the original fictional workload to the explicitly configured endpoint. Results describe this endpoint, model, workload and concurrency; they do not establish cross-provider superiority."
    lines = [f"# {headline}", "", "## The topic: inference performance", "",
             "A fast answer is useful only if it is complete, correct for the task, and delivered within the declared operating limit. I built this lab to make those conditions visible before choosing a hosting or caching strategy.", "",
             "### The narrower question", "",
             "Can I trust the serving measurements enough to decide what to benchmark on GPU next? The implemented boundary covers streamed transport, admission queues, schema/label quality, fallback attempts and cost assumptions.", "",
             "## The ordinary shortcut hides two problems", "",
             "A client can time a whole response and call it latency. It can also count streamed events and treat each event as a generated token. Neither tells me when useful content first arrived or whether one event contained several tokens. Averaging only successful requests additionally hides failures and rejected answers.", "",
             "## My approach: an explicit measurement contract", "",
             "I keep transport completion, task quality and SLO acceptance separate. The fixed batch has a bounded number of active workers; client queue time is recorded separately from post-admission response time. Fallback traces retain every attempt, including time and known usage from a rejected primary answer.", "",
             "The client reports time to first nonempty content as a TTFT proxy and records gaps between content chunks. True inter-token latency stays null because this wire protocol does not guarantee one token per chunk. Provider token usage is a separate billing counter, not a timing trace.", "",
             "## What I executed", "", paragraph, ""]
    if executed_protocol:
        lines.extend(["| Measure | Executed observation |", "| --- | ---: |",
                      f"| Scheduled requests | {protocol['requests']} |",
                      f"| HTTP/stream completions | {protocol['transport_successes']}/{protocol['requests']} |",
                      f"| Correct schema and label | {protocol['quality_successes']}/{protocol['requests']} |",
                      f"| Declared post-admission SLO passed | {protocol['slo_successes']}/{protocol['requests']} |",
                      f"| Total attempts | {protocol['attempts']} |",
                      f"| First-content proxy p50 / p95, ms, accepted answers | {number(protocol['ttft_proxy_successes_ms']['p50'])} / {number(protocol['ttft_proxy_successes_ms']['p95'])} |",
                      f"| E2E p50 / p95, ms, all requests | {number(protocol['e2e_all_requests_ms']['p50'])} / {number(protocol['e2e_all_requests_ms']['p95'])} |",
                      f"| True inter-token latency | Not identifiable from content chunks |", ""])
    if kv and kv.get("status") == "executed":
        lines.extend(["### A separate arithmetic experiment", "",
                      f"I ran an untrained causal-attention layer on **{kv['device_name']}**, using {kv['dtype']}, width {kv['width']}, {kv['heads']} heads, {kv['prefix_positions']} prefix positions and {kv['decode_positions']} decode positions. It compares full-context recomputation with projected key/value reuse, including prefill in the cached path.", "",
                      f"The two paths agreed within tolerance: **{kv['correctness']['allclose']}**; maximum absolute difference was **{kv['correctness']['max_absolute_error']:.8g}**. After {kv['warmup_runs']} warmups, the median across {kv['repeats']} repetitions was **{number(kv['median_full_recompute_ms'])} ms** for full recomputation and **{number(kv['median_kv_cached_ms'])} ms** for the cached path.", "",
                      "This is an arithmetic and measurement check. Random, untrained weights and tiny tensor timings cannot establish vLLM serving speed, LLM answer quality, production cost savings or the benefit of paged attention. The CUDA implementation uses synchronized events and NVTX ranges; inspect the recorded device before calling any observation a GPU result.", ""])
    else:
        lines.extend(["The CUDA attention experiment is not executed in this artifact.", ""])
    if protocol:
        cost = protocol["cost"]
        lines.extend(["## Cost is a declared model, not a provider bill", "",
                      f"Primary input/output assumptions per million tokens: **{number(cost['input_per_million_assumption'])} / {number(cost['output_per_million_assumption'])}**. Dedicated hourly assumption: **{number(cost['dedicated_hourly_assumption'])}**. These are manual assumptions; the fixture uses illustrative values, not published Nebius or NVIDIA prices.", "",
                      f"Every attempt has usage available: **{cost['all_attempt_usage_known']}**. The resulting economic-only break-even is **{number(cost['economic_break_even_requests_per_second'])} requests/second**. It is N/A when required usage or prices are unavailable; a zero or missing bill is not inferred. Cached-input discounts, reserved pricing, idle capacity, networking and staff costs are outside this simple rate model. Real fallback pricing must be supplied separately.", "",
                      "Even a valid economic crossing is insufficient: the dedicated deployment must actually sustain that load while satisfying the same quality and latency gates. That capacity and SLO feasibility is not established here.", ""])
    lines.extend(["## The result changes the next experiment", "",
                  "My recommendation is to use this lab as a measurement gate, then run the declared workload against a pinned vLLM deployment and a separately configured Nebius endpoint. Use a fresh vLLM process for the cold condition, repeat the same prefix for the warm condition, retain cache-hit evidence, and compare multiple repetitions under the same concurrency and output limit. Phase labels alone do not prove a cache reset or cache hit.", "",
                  "Read correctness flags and failure denominators before interpreting any timing. The committed local artifact checks fixture transport and tiny CPU tensor arithmetic. It is insufficient to select a GPU, a provider or an operating break-even for a real workload.", "",
                  "## Inspect and reproduce", "",
                  f"- [Workload and data card]({link(PROJECT / 'data' / 'DATA_CARD.md', output)})",
                  f"- [GPU, vLLM and Nebius runbook]({link(PROJECT / 'docs' / 'gpu-runbook.md', output)})",
                  f"- [Metric definitions]({link(PROJECT / 'docs' / 'measurement-contract.md', output)})",
                  "- [Numerical source](metrics.json) and [raw per-request traces](traces.jsonl)", "",
                  "This is new independent portfolio work by Vedant Thaker. It does not reproduce private employer/client infrastructure or validate historical business, audience or adoption figures.", ""])
    (output / "case-study.md").write_bytes("\n".join(lines).encode("utf-8"))
    if kv and kv.get("status") == "executed":
        values = [kv["median_full_recompute_ms"], kv["median_kv_cached_ms"]]
        maximum = max(values) * 1.15
        title = "Projected KV reuse passes the arithmetic check" if kv["correctness"]["allclose"] else "Arithmetic check failed; do not interpret the timing ratio"
        device = "CUDA GPU" if kv["device"] == "cuda" else "CPU"
        timing = "CUDA event time" if kv["device"] == "cuda" else "CPU wall time"
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="840" height="310" viewBox="0 0 840 310">',
               '<rect width="840" height="310" fill="#f8fafc"/>',
               f'<text x="30" y="34" font-size="22" font-family="sans-serif" fill="#0f172a">{title}</text>',
               f'<text x="30" y="60" font-size="14" font-family="sans-serif">Untrained tiny attention on {device}; this does not measure vLLM serving performance.</text>']
        for i, (name, value) in enumerate(zip(("Full recomputation", "KV reuse + prefill"), values)):
            y = 92 + i * 75
            svg.extend([f'<text x="30" y="{y+23}" font-size="16" font-family="sans-serif">{html.escape(name)}</text>',
                        f'<rect x="225" y="{y}" width="{value/maximum*470:.2f}" height="36" fill="{"#64748b" if i == 0 else "#0369a1"}"/>',
                        f'<text x="{235+value/maximum*470:.2f}" y="{y+24}" font-size="15" font-family="sans-serif">{value:.3f} ms</text>'])
        svg.append(f'<text x="30" y="285" font-size="14" font-family="sans-serif">Median {timing}; same input and weights; warmup excluded; full context vs one-position decode.</text></svg>')
        (output / "attention.svg").write_bytes("\n".join(svg).encode("utf-8"))
