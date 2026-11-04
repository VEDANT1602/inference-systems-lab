# Migration guide: local vLLM to Nebius Token Factory

## The problem behind a compatible API

An OpenAI-compatible interface reduces client changes; it does not make models, output semantics or billing interchangeable. My migration boundary keeps the same task contract while changing the explicitly selected provider configuration.

## Configuration only; no call has been made

The current base URL is `https://api.tokenfactory.nebius.com/v1`. Resolve a currently available model identifier from your own account rather than assuming the local model is offered. Supply environment variables in a private shell; never put a key in source, a notebook output, screenshots or a URL. The adapter excludes keys and endpoint URLs from traces. [Nebius primary cookbook](https://dev.nebius.com/cookbook/first-agent-on-nebius)

```powershell
$env:NEBIUS_BASE_URL = 'https://api.tokenfactory.nebius.com/v1'
$env:NEBIUS_MODEL = '<available-model-id>'
# Set NEBIUS_API_KEY privately in this process; do not paste a key into this guide.
# Execution is deliberately disabled until the explicit remote switch is supplied.
python -m inference_lab endpoint --provider nebius --out runs/nebius-plan
```

That command refuses remote execution. Once an explicit, verified zero-spend allowance or a separately authorized budget exists, the operator may choose to add `--execute-remote`. This project has not made that request, redeemed credits, created cloud resources or measured Nebius throughput.

## Compare what actually changed

Use the same workload hash, schema, output limit and concurrency. Retain all HTTP failures, timeouts, task errors and usage. A provider may report different token counts because tokenizers and chat templates differ. Never substitute missing usage with locally estimated counts and call it a bill.

Provide actual current rates as manual CLI inputs only after checking the selected model's pricing. The calculation supports an uncached input/output rate pair; provider-specific cached-token discounts are outside this simple model. Keep rates absent for a quality/latency-only evaluation. Economic break-even does not establish dedicated GPU capacity, especially when comparing different models.

## Free options and the stopping rule

The current task budget is zero extra spend. The [official Nebius Builder Program terms](https://nebius.com/builders-terms-and-conditions) describe promotional credits for eligible verified members, expiry and later pay-as-you-go usage. This is not an unlimited free tier. Do not assume eligibility, spend controls or a remaining balance. The lab's local demo and conditional free Colab notebook are the executable default; provider status remains `not_run`.

When a future managed evaluation is authorized, first cap the case count/output length and verify account-level spending controls, then preserve the run's model identifier and rate assumptions. The API key should be removed from the shell afterward. No billing or account change is performed by this repository.
