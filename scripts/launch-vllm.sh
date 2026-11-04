#!/usr/bin/env bash
set -euo pipefail
# Review and run on an already available compatible local Linux NVIDIA GPU.
# No cloud resources are created. Model weights are downloaded from a public repo.
command -v docker >/dev/null
command -v nvidia-smi >/dev/null
: "${MODEL_REVISION:?Set a reviewed exact model revision before launching}"
docker run --rm --name vedant-inference-lab --gpus all \
  --shm-size 1g \
  -p 127.0.0.1:8000:8000 \
  -v vedant-lab-model-cache:/root/.cache/huggingface \
  -e VLLM_NO_USAGE_STATS=1 \
  vllm/vllm-openai:v0.32.0 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision "$MODEL_REVISION" \
  --tokenizer-revision "$MODEL_REVISION" \
  --dtype half --max-model-len 1024 --max-num-seqs 4 \
  --gpu-memory-utilization 0.70 --enforce-eager --enable-prefix-caching
