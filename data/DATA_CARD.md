# Workloads and limits

## Support-routing stream contract

`workload.jsonl` contains 12 original fictional support messages, four each for `billing`, `technical` and `account`. Each case has a shared system policy, one user message, a unique case identifier and a 32-token output limit. The expected response is exactly one JSON object with the field `label`. The scenarios contain no real customers, transactions or employer data.

The shared policy is a seed for a prefix-reuse experiment. Twelve short examples do not characterize production prompt length, input diversity, model quality or cache economics. The case identifier lets the loopback fixture look up a known answer; it is not a learned model. A real endpoint receives both the identifier and the message, but receives no gold label outside the public policy.

The fixture emits synthetic usage of 64 input tokens and six output tokens per request. These are protocol test counters, not tokenization measurements. Its time and usage must not appear in a provider comparison. Actual endpoint runs retain usage if provided and preserve unknown usage as null.

## Fictional document extraction

`contract_extraction.jsonl` contains five invented snippets with two gold fields: `parties` and `notice_period`. One case omits the notice period. One includes a distractor renewal interval. Every output field requires a literal source quote, or explicit null value/null quote for an absent field.

The prompt can be submitted manually to a selected model and its JSON string saved as `{"case_id":"contract-01","output":"..."}` in a candidate JSONL file. Then run:

```powershell
python -m inference_lab.contract_gate runs/contract-candidates.jsonl
```

The gate requires exactly one candidate per case. Invalid JSON, unsupported fields, invented quotes, missing fields and wrong gold values fail. An exact source span is necessary evidence but does not prove semantic entailment. The fixed five-case gold set has no statistical power to rank models. No Claude, Nebius or other model comparison has been run. This task illustrates document extraction mechanics, not legal advice or contract interpretation.

## Rights and reproducibility

Vedant Thaker releases the original fictional workloads under this project's MIT license. The Python loader validates the support workload schema and unique IDs. Every endpoint artifact records the exact workload file's SHA-256, so a release comparison cannot silently substitute examples. No dataset download is required for local validation.
