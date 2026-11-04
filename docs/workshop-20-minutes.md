# Workshop: make an inference benchmark earn your trust

**Format:** a new 20-minute technical teaching artifact by Vedant Thaker. It has not been presented to a measured audience. Local CPU execution is sufficient; an optional free Colab GPU extends the arithmetic experiment if available.

## Minute 0–3: define the decision

Start with: “Would I deploy a faster answer if it is wrong?” Ask the learner to name the numerator and denominator in an inference success rate. Show 12 scheduled support cases and declare normal-stop, exact-schema and correct-label gates. Explain why transport completion is a separate event.

The desired outcome is a small decision contract: the same cases and output limit, every failure counted, a stated latency boundary and known price assumptions only.

## Minute 3–7: observe a real stream

Run `python -m inference_lab demo --out runs/workshop`. Open one trace and identify `content_chunks`, `ttft_proxy_ms`, `content_chunk_gap_ms`, usage and finish reason. Show the parser test that splits every UTF-8 byte and CRLF boundary. Explain why one socket read, one SSE event and one generated token are different units.

Ask the learner which quantity cannot be recovered from this trace. Expected answer: true inter-token latency. The honest value is null, not chunk-gap mean renamed as ITL.

## Minute 7–11: break the attractive chart

Run the wrong-label, interrupted-stream and timeout integration tests:

```powershell
python -m unittest discover -s tests -p test_lab.py -v
```

Inspect `test_fallback_retains_cost_and_primary_time`. A correct fallback rescues the case, but the primary attempt still consumed time and possibly tokens. Ask the learner why measuring only the fallback's first chunk would understate accepted-answer latency.

## Minute 11–15: establish arithmetic correctness

Read `TinyAttention.prefill` and `decode`, then run `python -m inference_lab kv --out runs/workshop-attention.json`. Compare the cached and full outputs before inspecting time. State that weights are random and untrained; this checks causal attention and timing discipline, not LLM capability.

If the free Colab notebook has a compatible NVIDIA GPU, run its CUDA cell. Point out event synchronization and NVTX ranges. If no GPU is available, retain `not_run` and discuss the experiment design without inventing a result. Nsight tracing is optional and remains unexecuted until actually collected.

## Minute 15–18: make the next deployment reviewable

Walk through the two migration guides. Local vLLM needs a compatible GPU and pinned runtime/model; Nebius requires an explicitly configured endpoint and spending authorization. Show the cold/warm protocol and why a `warm` label is not cache-hit evidence. On Colab use interactive cells, without an external server/tunnel or bypassing resource limits.

## Minute 18–20: the learner's deliverable

The learner submits one artifact containing the workload hash, failure denominator, quality count, stated latency boundary, raw traces and a decision to hold or continue. For a document task, use the five fictional snippets and source-span gate; an exact quotation is required but does not itself establish legal correctness.

The expected success is an inspectable measurement contract and an honest next experiment. There is no promised speedup, certification, production adoption or audience reach. Primary implementation references are collected in [sources](sources.md).
