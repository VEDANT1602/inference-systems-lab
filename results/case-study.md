# The measurement contract works; GPU and provider performance remain unmeasured

## The topic: inference performance

A fast answer is useful only if it is complete, correct for the task, and delivered within the declared operating limit. I built this lab to make those conditions visible before choosing a hosting or caching strategy.

### The narrower question

Can I trust the serving measurements enough to decide what to benchmark on GPU next? The implemented boundary covers streamed transport, admission queues, schema/label quality, fallback attempts and cost assumptions.

## The ordinary shortcut hides two problems

A client can time a whole response and call it latency. It can also count streamed events and treat each event as a generated token. Neither tells me when useful content first arrived or whether one event contained several tokens. Averaging only successful requests additionally hides failures and rejected answers.

## My approach: an explicit measurement contract

I keep transport completion, task quality and SLO acceptance separate. The fixed batch has a bounded number of active workers; client queue time is recorded separately from post-admission response time. Fallback traces retain every attempt, including time and known usage from a rejected primary answer.

The client reports time to first nonempty content as a TTFT proxy and records gaps between content chunks. True inter-token latency stays null because this wire protocol does not guarantee one token per chunk. Provider token usage is a separate billing counter, not a timing trace.

## What I executed

The loopback server emits known answers from a lookup table. Its timings exercise the client and cannot rank vLLM, Nebius or GPUs.

| Measure | Executed observation |
| --- | ---: |
| Scheduled requests | 12 |
| HTTP/stream completions | 12/12 |
| Correct schema and label | 12/12 |
| Declared post-admission SLO passed | 12/12 |
| Total attempts | 12 |
| First-content proxy p50 / p95, ms, accepted answers | 6.306 / 8.565 |
| E2E p50 / p95, ms, all requests | 17.040 / 19.717 |
| True inter-token latency | Not identifiable from content chunks |

### A separate arithmetic experiment

I ran an untrained causal-attention layer on **Intel64 Family 6 Model 78 Stepping 3, GenuineIntel**, using float32, width 64, 4 heads, 64 prefix positions and 24 decode positions. It compares full-context recomputation with projected key/value reuse, including prefill in the cached path.

The two paths agreed within tolerance: **True**; maximum absolute difference was **5.2154064e-08**. After 2 warmups, the median across 7 repetitions was **12.064 ms** for full recomputation and **4.107 ms** for the cached path.

This is an arithmetic and measurement check. Random, untrained weights and tiny tensor timings cannot establish vLLM serving speed, LLM answer quality, production cost savings or the benefit of paged attention. The CUDA implementation uses synchronized events and NVTX ranges; inspect the recorded device before calling any observation a GPU result.

## Cost is a declared model, not a provider bill

Primary input/output assumptions per million tokens: **1.000 / 2.000**. Dedicated hourly assumption: **1.000**. These are manual assumptions; the fixture uses illustrative values, not published Nebius or NVIDIA prices.

Every attempt has usage available: **True**. The resulting economic-only break-even is **3.655 requests/second**. It is N/A when required usage or prices are unavailable; a zero or missing bill is not inferred. Cached-input discounts, reserved pricing, idle capacity, networking and staff costs are outside this simple rate model. Real fallback pricing must be supplied separately.

Even a valid economic crossing is insufficient: the dedicated deployment must actually sustain that load while satisfying the same quality and latency gates. That capacity and SLO feasibility is not established here.

## The result changes the next experiment

My recommendation is to use this lab as a measurement gate, then run the declared workload against a pinned vLLM deployment and a separately configured Nebius endpoint. Use a fresh vLLM process for the cold condition, repeat the same prefix for the warm condition, retain cache-hit evidence, and compare multiple repetitions under the same concurrency and output limit. Phase labels alone do not prove a cache reset or cache hit.

Read correctness flags and failure denominators before interpreting any timing. The committed local artifact checks fixture transport and tiny CPU tensor arithmetic. It is insufficient to select a GPU, a provider or an operating break-even for a real workload.

## Inspect and reproduce

- [Workload and data card](../data/DATA_CARD.md)
- [GPU, vLLM and Nebius runbook](../docs/gpu-runbook.md)
- [Metric definitions](../docs/measurement-contract.md)
- [Numerical source](metrics.json) and [raw per-request traces](traces.jsonl)

This is new independent portfolio work by Vedant Thaker. It does not reproduce private employer/client infrastructure or validate historical business, audience or adoption figures.
