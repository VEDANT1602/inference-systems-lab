# Measurement contract

## Boundary before numbers

This is a **bounded closed-loop finite batch**. At most `concurrency` workers run at once; a worker begins another case after completing its current case. The batch is not an open-loop Poisson arrival experiment and does not estimate maximum sustainable production load. It has no claim about tail latency under a fixed external arrival rate.

The timer starts immediately before connecting and sending the HTTP request. After connection setup, a deadline watchdog shuts down the retained socket, bounding both trickled response headers and body reads rather than relying only on an idle socket timeout. Streamed SSE body bytes are capped at 65,536; deeply nested JSON is retained as a protocol or quality failure instead of aborting the batch. DNS resolution and operating-system connection setup can still outlast socket timeouts before the watchdog attaches, so this is not a hard cancellation guarantee for every network failure. Thread scheduling and cleanup can also add a small overrun. The timeout is per attempt, so a primary plus one fallback can take approximately twice the configured attempt timeout plus worker overhead. No automatic retry hides additional usage or failures.

| Field | Exact meaning |
| --- | --- |
| `requests` | All scheduled cases, including HTTP errors, timeouts and wrong/malformed answers |
| `attempts` | Primary plus any fallback requests; each is retained in raw traces |
| `transport_successes` | Final attempts with HTTP 200, valid SSE framing, finish reason, content and DONE marker |
| `quality_successes` | Final answers with normal stop, exact JSON schema and correct known label |
| `ttft_proxy_ms` | Per-attempt client time to first nonempty content delivery; includes connect, transfer and parsing |
| `accepted_ttft_proxy_ms` | All rejected primary time plus the accepted final attempt's first content time |
| `first_content_any_attempt_ms` | Earliest content from any attempt; can belong to a rejected answer and is not used for SLO acceptance |
| `content_chunk_gap_ms` | Consecutive content-delivery gaps; a chunk can carry multiple tokens |
| `inter_token_latency_ms` | Null: true token-level timing is not identifiable from this client protocol |
| `client_queue_ms` | Time from batch start to worker admission; includes finite-batch queueing and submission overhead |
| `e2e_ms` | Worker admission to final result; includes every primary/fallback attempt |
| `e2e_including_queue_ms` | Batch start to final result for each case |
| SLO accepted | Correct final answer, accepted TTFT proxy <= 1,000 ms and post-admission E2E <= 5,000 ms by default |
| Quality/SLO goodput | Accepted case count divided by finite-batch wall seconds, including errors; not capacity proof |

The default SLO intentionally excludes client queue time and says so. To make a user-visible latency claim, inspect `e2e_including_queue_ms` and declare an arrival process. p50/p95 use linear interpolation across the observed sorted values. Accepted first-content percentiles exclude rejected answers, while E2E percentiles include all scheduled cases; the counts remain beside them.

The vLLM server can batch tokens into streamed events. The lab therefore labels first-content timing as a proxy and keeps true ITL unknown. For token timing and larger load tests, independently run vLLM's benchmark tooling and record its exact configuration; do not rename these client chunk gaps as ITL. [vLLM OpenAI server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/), [benchmark CLI](https://docs.vllm.ai/en/latest/cli/bench/serve/)

## Cost and fallback

`attempt_cost = (input_tokens * input_rate + output_tokens * output_rate) / 1,000,000`.

`economic_crossing_rps = dedicated_hourly / (3,600 * mean_cost_per_scheduled_case)`.

All rates are manually supplied assumptions, not current provider prices. Every attempt must have valid usage for total cost to be known. A failed HTTP request with no usage makes full cost unknown; it is not assumed free. Fallback has a separate rate pair. Cost per quality success divides total known attempt cost by accepted cases, including costs of failed/rejected requests.

The crossing is meaningful only with the same workload and accounting scope. It omits cached-input discounts, commitment discounts, networking, staff, idle capacity and billing rounding. It says nothing about a dedicated GPU's ability to sustain that request rate with acceptable quality and tail latency.

## Release decision

The comparison requires identical nonempty workload hash, scheduled denominator, concurrency and execution class. The candidate must meet the declared minimum quality, avoid quality regression and keep p95 post-admission E2E within the allowed ratio. Model, dtype, output limit, prefix phase, hardware and environment must also be reviewed before interpreting a candidate change. The code's gate is a fixed-workload acceptance rule, not a confidence interval or proof of population superiority.

## Tensor experiment

Both paths share an untrained attention layer, seed, dtype and input. Full recomputation projects and attends over the entire context at every decode step. The cached path projects the initial prefix once, stores keys and values, then appends one position each step; that projection prefill is included. Because the experiment is one layer with fixed input embeddings, earlier prefix outputs are unnecessary for the checked last-position outputs. A multi-layer autoregressive model has additional prefill work and cannot inherit this timing ratio. The implementation uses ordinary tensor concatenation and dense attention, not a paged KV allocator or a full transformer decoder.

Correctness is checked before timing. CPU repeats use `perf_counter`; CUDA repeats use events, synchronize before each trial and wait on the ending event. Warmups are excluded. GPU memory is the combined allocator peak across both methods, not a per-method comparison. Small tensors, Python overhead and uncontrolled clock/load effects limit generalization. [PyTorch CUDA timing guidance](https://docs.pytorch.org/docs/main/notes/cuda.html)
