# GPU execution runbook

The committed results contain no NVIDIA, vLLM or Nebius performance measurements. The checked Windows machine has a CPU-only Torch build. This runbook is a plan for compatible hardware, with the free Colab notebook as the zero-spend option.

## First establish capability

```bash
python -m inference_lab doctor --out runs/doctor.json
python -m inference_lab kv --device cuda --dtype float16 --out runs/cuda-attention.json
```

Record GPU model, driver, CUDA build, Torch/vLLM version, model revision, dtype, context length, memory limit and active processes. The tiny runner records synchronized CUDA-event timings and combined allocator peak only if CUDA actually executes. Do not reuse CPU observations as GPU observations.

Current vLLM NVIDIA requirements include Linux and compute capability at least 7.5; a T4 meets the capability floor but the installed wheel's driver/CUDA compatibility must still pass. Colab assignments and driver versions vary. Fresh environments reduce binary conflicts. [vLLM GPU requirements](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)

## Local Linux vLLM deployment

Only use an already available NVIDIA GPU. No VM creation or billing setup is part of this script. Review `scripts/launch-vllm.sh`, then:

```bash
export MODEL_REVISION=2b01de6d1108f9b2b5e46a726aa678a359b6c03b
bash scripts/launch-vllm.sh
```

The script launches the official image `vllm/vllm-openai:v0.32.0`, binds the published port to host loopback, uses the public Qwen2.5-0.5B-Instruct model with float16 and explicit prefix caching, and caps context/sequence count for a small experiment. This model/release pairing is a configured example, not an executed certification. Record the actual image digest with `docker image inspect vllm/vllm-openai:v0.32.0 --format '{{index .RepoDigests 0}}'`. There is no external service exposure or required provider key. [Official Docker guide](https://docs.vllm.ai/en/latest/deployment/docker/), [pinned model revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/2b01de6d1108f9b2b5e46a726aa678a359b6c03b)

In a second shell, run the client from this directory:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
python -m inference_lab endpoint --provider vllm --phase cold --out runs/vllm-cold --concurrency 2
python -m inference_lab endpoint --provider vllm --phase warm --out runs/vllm-warm --concurrency 2
```

## Prefix-cache protocol

1. Pin model revision, image digest, dtype, hardware, output limit and concurrency. Preserve the workload SHA-256 and all raw trials.
2. Start a fresh vLLM process. Wait for model load/readiness without submitting the benchmark prefix. Save a pre-run metrics snapshot and note the process start; engine/kernel warmup can still confound the first request.
3. Run the cold batch once. Save server cache counters/logs and timing. Within-batch shared prefixes can already warm the cache; interpret per-request order rather than calling the entire batch a pure cold sample.
4. Run the exact batch again against the same process for the warm condition. Retain cache-hit evidence from the server, not just a phase string. The client sets `cache_reset_or_hit_verified=false` until separately reviewed evidence is attached.
5. Repeat fresh-process cold/warm pairs, alternate configuration order where practical, and preserve all failure and quality counts. To isolate cache behavior, also compare prefix caching disabled against enabled using otherwise identical deployments.
6. Review cache counters, prompts and quality before interpreting latency. Small shared prefixes may not produce useful hits; a null effect is a valid result. Do not infer production savings from one pair.

vLLM's automatic prefix caching is a server feature. The toy attention cache in this repository is a different experiment and cannot demonstrate it. [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

## NVIDIA telemetry and profiling

On a compatible local GPU, collect low-frequency utilization/power/allocator context in a separate terminal with `bash scripts/collect-nvidia.sh runs/nvidia-telemetry.csv`, then stop it with Ctrl-C. Sampling telemetry does not directly measure CUDA kernel occupancy or prove utilization efficiency.

If NVIDIA Nsight Systems is installed, profile the actual tiny CUDA experiment:

```bash
nsys profile --trace=cuda,nvtx --sample=none -o runs/kv-profile python -m inference_lab kv --device cuda --dtype float16 --out runs/profile-attention.json
```

Inspect the `full_recompute` and `kv_cached` NVTX ranges, launch count and CUDA execution. Exclude initialization/warmup ranges from comparison and keep unprofiled benchmark trials separately because profiling perturbs time. No Nsight capture is included in the committed result. [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/)

## Managed endpoint

Use the [Nebius migration guide](migrate-to-nebius.md) for configuration. Current user scope is zero extra spend; do not run a managed endpoint just because a key or a promotional balance exists. Free Colab execution does not require a Nebius account and does not prove the Nebius adapter's provider compatibility.
