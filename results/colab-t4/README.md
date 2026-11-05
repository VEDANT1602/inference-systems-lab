# A real CUDA check on a free Colab Tesla T4

The next question was narrower than choosing a serving platform: **does the projected KV path agree with full recomputation when the arithmetic actually runs on NVIDIA CUDA?** I ran the published notebook interactively on a free Colab Tesla T4, with no provider API request or paid compute allocation for this experiment.

The two paths passed the declared correctness check. Across seven measured repetitions, full recomputation had a median of **25.626 ms** and projected KV reuse had a median of **20.790 ms**. These are measurements of one small, untrained attention layer. They do not establish vLLM serving performance, LLM quality or production savings.

## What ran

| Configuration | Recorded value |
| --- | --- |
| Repository commit | [`5780a7c0661c4cf528957cff730de16e3fd447c4`](https://github.com/VEDANT1602/DataScienceProject/commit/5780a7c0661c4cf528957cff730de16e3fd447c4) |
| GPU | NVIDIA Tesla T4; 15,360 MiB reported total memory |
| Driver / Torch CUDA build | 580.82.07 / CUDA 13.0 |
| Runtime | Linux; Python 3.13.16; Torch 2.11.0+cu130 |
| Numerical configuration | float16; seed 42; batch 1; width 64; four attention heads |
| Input | Random embeddings; 128 prefix positions and 64 subsequent positions |
| Warmup / measured trials | Three warmups per path; seven measured trials per path |
| Torch CPU threads | One; the Colab runtime reported two logical CPUs |
| Timing | CUDA events, synchronization before each trial, ending-event synchronization |

The workload is generated tensor arithmetic, with no trained model weights, downloaded evaluation dataset or customer data. Both paths share the same layer and inputs. Full recomputation projects and attends over the growing context at every step. The cached path projects the prefix once, retains its keys and values, and appends one new position at each step. That projection prefill is included in cached timing.

This single-layer setup does not need earlier prefix output vectors to compute the checked last-position outputs. A full autoregressive transformer has additional layers, prefill work, scheduling and memory behavior. Ordinary tensor concatenation here is also distinct from vLLM's paged KV allocator and automatic prefix caching.

## Correctness before interpretation

The checked outputs satisfied `torch.allclose` with **absolute and relative tolerances both 0.02**. The observed maximum absolute difference was **0.00006103515625**. This is agreement within the declared float16 tolerance, rather than bitwise equality or a claim about every input shape.

The combined allocator peak during measurement was **10,620,928 bytes**. It spans both methods and their measurement overhead; it cannot support a per-method memory-savings claim. No Nsight kernel profile was collected in this artifact.

## Raw repetitions

Values below are milliseconds, rounded to six decimal places. [The numerical artifact](cuda-attention.json) retains full precision.

| Repetition | Full recomputation | KV reuse with projection prefill |
| --- | ---: | ---: |
| 1 | 25.625919 | 21.899456 |
| 2 | 28.927616 | 17.887232 |
| 3 | 25.339617 | 18.383200 |
| 4 | 24.276928 | 21.604383 |
| 5 | 24.057728 | 18.703392 |
| 6 | 27.348991 | 20.789761 |
| 7 | 28.106785 | 21.179041 |
| **Median** | **25.625919** | **20.789761** |

The runner measures all full-recomputation trials before all cached trials; rows identify repetition number, not alternating matched pairs. CUDA event intervals can include GPU idle gaps while Python schedules the small operations. Clock state, shared-runtime activity and launch overhead were not isolated. Seven trials provide an inspectable observation, without a confidence interval or a claim of stable service capacity.

## What this changes

This result moves the CUDA attention path from an execution plan to an actual hardware-backed arithmetic and timing check. The next serving experiment still needs a pinned model/runtime, quality and failure denominators, a declared load process and verified cache evidence. This artifact contains no streaming TTFT, inter-token latency, provider-cost comparison or vLLM/Nebius serving measurement. It also makes no CPU-versus-GPU comparison.

Inspect [the capability snapshot](doctor.json) beside [the executed CUDA result](cuda-attention.json). The doctor's `available_not_yet_run` field describes the discovery step before the benchmark; the separate CUDA artifact records `status: executed`. Reproduce the workflow with [the interactive notebook](../../notebooks/free_gpu_lab.ipynb), then keep the emitted configuration and raw trials with any new result.

A separate [post-install capability snapshot](post-install/doctor.json) and [repeat CUDA run](post-install/cuda-attention.json) retain the environment after installing the optional vLLM dependency set and restarting Python. That environment used Torch 2.13.0+cu130; its full/cached medians were 23.859/17.525 ms, again passing the declared tolerance. This was a sequential environment change, not a controlled Torch-version or speed comparison. The original run above remains unchanged. The retained files do not establish vLLM endpoint, prefix-cache or Nebius performance.

This is new independent work by **Vedant Thaker**, using the measurement and reporting approach in this repository. It does not reproduce private employer/client infrastructure or imply an affiliation with NVIDIA, Google, vLLM or Nebius.
