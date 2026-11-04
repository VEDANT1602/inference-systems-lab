"""Actual tiny causal-attention arithmetic; this is not a trained language model."""
from __future__ import annotations

import math
import os
import platform
import shutil
import statistics
import subprocess
import time

# Import Torch before other numerical packages on this tested Windows runtime.
import torch


class TinyAttention(torch.nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("Width must be divisible by the number of heads")
        self.width, self.heads, self.dim = width, heads, width // heads
        self.qkv = torch.nn.Linear(width, 3 * width, bias=False)
        self.out = torch.nn.Linear(width, width, bias=False)

    def projections(self, x):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return tuple(t.reshape(batch, length, self.heads, self.dim).transpose(1, 2) for t in (q, k, v))

    def full(self, x):
        q, k, v = self.projections(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dim)
        causal = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))
        joined = (scores.softmax(-1) @ v).transpose(1, 2).reshape(x.shape)
        return self.out(joined)

    def prefill(self, x):
        _, k, v = self.projections(x)
        return k, v

    def decode(self, x, cache):
        if x.shape[1] != 1:
            raise ValueError("Decode takes one new position")
        q, k, v = self.projections(x)
        k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        joined = ((q @ k.transpose(-2, -1) / math.sqrt(self.dim)).softmax(-1) @ v)
        joined = joined.transpose(1, 2).reshape(x.shape)
        return self.out(joined), (k, v)


def doctor():
    available = torch.cuda.is_available()
    executable = shutil.which("nvidia-smi")
    gpu_inventory = []
    if executable:
        result = subprocess.run([executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                                text=True, capture_output=True, timeout=10)
        if result.returncode == 0:
            gpu_inventory = result.stdout.strip().splitlines()
    return {"python": platform.python_version(), "platform": platform.system(), "torch": torch.__version__,
            "cpu_processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
            "cuda_available": available, "torch_cuda_build": torch.version.cuda,
            "nvidia_smi_found": bool(executable), "gpu_inventory": gpu_inventory,
            "cuda_benchmark_status": "available_not_yet_run" if available else "not_run_cuda_unavailable",
            "vllm_endpoint_status": "not_run", "nebius_endpoint_status": "not_run",
            "nsight_profile_status": "not_run"}


def kv_benchmark(device="cpu", dtype_name="float32", prefix=64, decode=24, repeats=7, warmup=2, seed=42):
    if device not in {"cpu", "cuda"} or dtype_name not in {"float32", "float16"}:
        raise ValueError("Unsupported device/dtype")
    if device == "cuda" and not torch.cuda.is_available():
        return {"status": "not_run_cuda_unavailable", "requested_device": "cuda", "scope": "toy_attention_arithmetic"}
    if device == "cpu" and dtype_name == "float16":
        raise ValueError("The CPU reference uses float32")
    if min(prefix, decode, repeats) < 1 or warmup < 0 or prefix + decode > 4096 or repeats > 100:
        raise ValueError("Invalid benchmark size")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype = torch.float32 if dtype_name == "float32" else torch.float16
    model = TinyAttention().to(device=device, dtype=dtype).eval()
    x = torch.randn(1, prefix + decode, 64, device=device, dtype=dtype)

    def reference():
        return torch.cat([model.full(x[:, :prefix+i+1])[:, -1:] for i in range(decode)], dim=1)

    def cached():
        cache = model.prefill(x[:, :prefix])
        outputs = []
        for i in range(decode):
            value, cache = model.decode(x[:, prefix+i:prefix+i+1], cache)
            outputs.append(value)
        return torch.cat(outputs, dim=1)

    def measure(function, label):
        values = []
        for _ in range(repeats):
            if device == "cuda":
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.nvtx.range_push(label)
                start.record()
                function()
                end.record()
                torch.cuda.nvtx.range_pop()
                end.synchronize()
                values.append(start.elapsed_time(end))
            else:
                began = time.perf_counter()
                function()
                values.append((time.perf_counter() - began) * 1000)
        return values

    with torch.inference_mode():
        expected, actual = reference(), cached()
        max_error = (expected - actual).abs().max().item()
        tolerance = 2e-2 if dtype_name == "float16" else 1e-5
        correct = torch.allclose(expected, actual, atol=tolerance, rtol=tolerance)
        for _ in range(warmup):
            reference()
            cached()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        full_ms, cached_ms = measure(reference, "full_recompute"), measure(cached, "kv_cached")
        peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    full_median, cached_median = statistics.median(full_ms), statistics.median(cached_ms)
    return {"status": "executed", "scope": "untrained_tiny_causal_attention_not_vllm_or_llm_quality",
            "device": device, "dtype": dtype_name, "torch": torch.__version__, "python": platform.python_version(),
            "device_name": torch.cuda.get_device_name() if device == "cuda" else (platform.processor() or "CPU"),
            "torch_threads": 1, "seed": seed, "batch": 1, "width": 64, "heads": 4,
            "prefix_positions": prefix, "decode_positions": decode, "repeats": repeats, "warmup_runs": warmup,
            "timing": "CUDA events with synchronization" if device == "cuda" else "CPU perf_counter wall time",
            "correctness": {"allclose": correct, "max_absolute_error": max_error, "atol_rtol": tolerance},
            "full_recompute_ms": full_ms, "kv_cached_ms": cached_ms,
            "median_full_recompute_ms": full_median, "median_kv_cached_ms": cached_median,
            "median_ratio_full_to_cached": full_median / cached_median,
            "cuda_peak_allocated_bytes_combined_measurement": peak,
            "memory_scope": "combined benchmark peak, not separate per-method peaks",
            "kernel_profile_status": "not_run_requires_nsight"}
