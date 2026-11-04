# Migration guide: a chat application to local vLLM

## The integration problem

Replacing an endpoint URL can preserve HTTP shape while changing tokenizer behavior, chat templates, context limits, stop reasons, response quality and batching. I keep the application task fixed and make those differences measurable before switching traffic.

## A small, reviewable migration

1. Export the application's non-sensitive cases with gold labels and exact output schema. This repository starts with 12 fictional cases; production evaluation needs a separate representative dataset.
2. Select an open model, pin its revision, verify its license and ensure the hardware/runtime supports it. The [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) documents the small example used here; choosing it does not establish sufficient task quality.
3. Follow the [GPU runbook](gpu-runbook.md) and verify readiness at loopback. Preserve the actual vLLM image digest, model revision, dtype, context limit and command.
4. Configure `VLLM_BASE_URL` and `VLLM_MODEL`. Keep the same system message, generation limit and concurrency. Run `python -m inference_lab endpoint --provider vllm --out runs/vllm-candidate`.
5. Inspect every failed schema/label and finish reason. The client reports a TTFT proxy, chunk gaps and all-request E2E; do not reinterpret event gaps as token latency.
6. Compare only matching workload hashes and concurrency: `python -m inference_lab compare runs/vllm-baseline/metrics.json runs/vllm-candidate/metrics.json`. A failed gate means hold the candidate and inspect traces. A passing gate means the fixed cases passed the configured rule, not that production rollout is proven safe.

## Fallback without hidden work

The Python API supports one explicit fallback. Every attempted response remains in traces. A rejected primary's elapsed time is included in the accepted answer's first-content timing, and a separate fallback price pair is required when modeling costs:

```python
import os
from pathlib import Path
from inference_lab.benchmark import load_workload, run_batch
from inference_lab.protocol import Endpoint

cases = load_workload(Path('data/workload.jsonl'))
primary = Endpoint('candidate', 'http://127.0.0.1:8000/v1', os.environ['VLLM_MODEL'])
fallback = Endpoint('known-local-baseline', 'http://127.0.0.1:8001/v1', os.environ['BASELINE_MODEL'])
result = run_batch(cases, primary, fallback=fallback, concurrency=2)
```

Both ports must be deliberately configured model deployments. No fallback starts automatically and no remote provider is implied. Before rollout, expand coverage, compare versioned failure sets, and measure a declared arrival process under realistic load. The standard library client is intentionally narrow: it handles one chat choice and text deltas, not multimodal output or tool calls.

Integration reference: [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).
