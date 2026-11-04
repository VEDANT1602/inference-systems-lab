# Primary sources and version notes

Reviewed 2026-11-04. Documentation describes what to configure; it is not evidence that this repository executed a GPU or provider run.

| Source | What was verified |
| --- | --- |
| [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/) | Chat endpoint and streaming behavior; token batching prevents reliable client ITL inference |
| [vLLM benchmark serve](https://docs.vllm.ai/en/latest/cli/bench/serve/) | Independent benchmark metric families and SLO-oriented goodput |
| [vLLM prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | Shared-prefix reuse is distinct from reuse inside this tiny attention implementation |
| [vLLM Docker deployment](https://docs.vllm.ai/en/latest/deployment/docker/) | Official image and GPU deployment pattern |
| [vLLM v0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) | Pinned non-RC semver used by the example; not executed here |
| [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) | Linux, supported Python, NVIDIA compute capability >= 7.5; CUDA/PyTorch binary compatibility matters |
| [vLLM supported models](https://docs.vllm.ai/en/stable/models/supported_models/) | Qwen2ForCausalLM architecture support |
| [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Public model, Apache-2.0 license, model-specific vLLM use example |
| [vLLM LLM API](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/llm/) | In-process inference parameters for the optional notebook |
| [NVIDIA Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/) | CUDA/NVTX tracing CLI; profiling is a separate execution |
| [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html) | Asynchronous execution and synchronized timing |
| [Nebius first-agent cookbook](https://dev.nebius.com/cookbook/first-agent-on-nebius) | Current Token Factory OpenAI-compatible base URL |
| [Nebius Builder Program terms](https://nebius.com/builders-terms-and-conditions) | Eligible verified members can receive promotional credits; credits expire and PAYG may follow |
| [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) | Free GPU availability is conditional; interactive notebook use and resource policies |

The release listing also contained an RC-named newer tag when reviewed. This project deliberately pins the checked v0.32.0 example rather than interpreting a newer RC as a stable recommendation. New installations must re-check compatibility and resolve the image digest before publishing comparable results. Model weights are not bundled; the optional notebook resolves and records a concrete Hugging Face revision at execution time.
