"""Rebuild the output-free, interactive Colab notebook from reviewable cell text."""
import json
from pathlib import Path
import textwrap


def md(source):
    return {"cell_type":"markdown", "metadata":{}, "source":textwrap.dedent(source).strip().splitlines(True)}


def code(source):
    return {"cell_type":"code", "execution_count":None, "metadata":{}, "outputs":[],
            "source":textwrap.dedent(source).strip().splitlines(True)}


cells = [
md('''
# Free GPU inference lab: correctness before a serving claim

By Vedant Thaker. This is new independent portfolio work. The committed notebook has **no executed GPU output**. It uses interactive notebook cells, without SSH, tunnels, a hosted web UI, billing setup or provider API requests.

Choose **Runtime → Change runtime type → a free GPU**. Free GPUs are not guaranteed; stop if Colab offers only a paid upgrade. Keep the notebook interactive and respect resource limits. The first experiment is actual tiny PyTorch attention arithmetic. The optional second experiment uses in-process vLLM with a public small model. Neither reproduces private employer work.

Primary references: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [vLLM GPU requirements](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/), [Qwen model/license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [vLLM LLM API](https://docs.vllm.ai/en/v0.32.0/api/vllm/entrypoints/llm/).
'''),
code('''
import os, sys, subprocess, json, hashlib, platform
from pathlib import Path
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
REPO_URL = "https://github.com/VEDANT1602/DataScienceProject.git"
BRANCH = "master"
CHECKOUT = Path("/content/vedant-inference-lab")
if not CHECKOUT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                    "--branch", BRANCH, REPO_URL, str(CHECKOUT)], check=True)
subprocess.run(["git", "sparse-checkout", "set", "projects/inference-systems-lab"], cwd=CHECKOUT, check=True)
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip()
LAB = CHECKOUT / "projects/inference-systems-lab"
assert (LAB / "inference_lab/kv.py").is_file(), "The published lab path is missing."
sys.path.insert(0, str(LAB))
os.chdir(LAB)
OUT = Path("/content/inference-evidence")
OUT.mkdir(exist_ok=True)
print({"repository_commit": COMMIT, "python":platform.python_version(), "path":str(LAB)})
'''),
md('''
## Optional vLLM installation, before importing Torch

Leave this disabled for the tiny CUDA experiment, which uses Colab's installed Torch. If you want the optional vLLM experiment, inspect the pinned installation and enable it once on a compatible free runtime. vLLM wheels bundle compatible Torch dependencies; replacing packages can require a runtime restart. After a restart, rerun the setup cell with this installation switch disabled. If installation or compatibility fails, preserve that failure and skip vLLM; do not purchase compute to continue.
'''),
code('''
INSTALL_VLLM = False
if INSTALL_VLLM:
    assert platform.system() == "Linux" and (3,11) <= sys.version_info[:2] <= (3,14)
    capability_text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True)
    capabilities = [float(x.strip()) for x in capability_text.splitlines() if x.strip()]
    assert capabilities and min(capabilities) >= 7.5, "vLLM example requires NVIDIA capability >=7.5"
    subprocess.run([sys.executable, "-m", "pip", "install", "vllm==0.32.0"], check=True)
    print("Installation complete. Restart the runtime if requested, then rerun setup with INSTALL_VLLM=False.")
else:
    print("vLLM installation skipped; no package changes requested.")
'''),
md('''
## 1. Capability and actual CUDA arithmetic

The model is one untrained causal-attention layer, not an LLM. Both paths use the same input and weights. Compare outputs before timing. CUDA uses synchronized events; warmups are excluded and the cached path includes prefill. The memory number is the combined allocator peak across both methods. If CUDA is missing, the artifact says `not_run_cuda_unavailable`.
'''),
code('''
import torch
from inference_lab.kv import doctor, kv_benchmark
from inference_lab.report import write_json
environment = doctor()
environment["repository_commit"] = COMMIT
write_json(OUT/"doctor.json", environment)
print(json.dumps(environment, indent=2))
cuda_result = kv_benchmark(device="cuda", dtype_name="float16", prefix=128,
                           decode=64, repeats=7, warmup=3)
cuda_result["repository_commit"] = COMMIT
write_json(OUT/"cuda-attention.json", cuda_result)
print(json.dumps(cuda_result, indent=2))
if cuda_result["status"] == "executed":
    assert cuda_result["correctness"]["allclose"], "Correctness failed; do not interpret timing ratios."
'''),
md('''
## 2. Optional in-process vLLM, no external endpoint

Keep `RUN_VLLM=False` until the optional package installation and driver/runtime checks pass. A capability floor is not a guarantee of wheel compatibility or available memory. This conservative example requires at least 6 GiB total VRAM, uses float16 for T4 compatibility, and caps context at 1,024 positions. It downloads the public Apache-2.0 Qwen2.5-0.5B-Instruct model without a token and records the exact resolved revision.

Two sequential batches test the same original support cases. Labels `first_batch` and `repeat_batch` describe order, **not verified cold/warm cache states**. No TTFT/ITL is inferred from offline batch generation. Preserve failures and low quality; a small model may not pass the schema or label gate.
'''),
code('''
RUN_VLLM = False
from importlib.metadata import version, PackageNotFoundError
from urllib.request import urlopen
import re, time
from inference_lab.benchmark import load_workload
from inference_lab.protocol import quality

vllm_result = {"status":"not_run_disabled", "repository_commit":COMMIT,
               "provider_api_requests":0, "execution_kind":"inprocess_vllm_not_streaming_endpoint"}
if RUN_VLLM:
    try:
        installed_vllm = version("vllm")
    except PackageNotFoundError:
        installed_vllm = None
    capable = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7,5)
    enough_memory = capable and torch.cuda.get_device_properties(0).total_memory >= 6*1024**3
    if installed_vllm != "0.32.0" or not capable or not enough_memory:
        vllm_result.update(status="not_run_compatibility_gate", installed_vllm=installed_vllm,
                           capability_gate=bool(capable), memory_gate=bool(enough_memory))
    else:
        from vllm import LLM, SamplingParams
        MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
        with urlopen("https://huggingface.co/api/models/"+MODEL, timeout=20) as response:
            revision = json.load(response)["sha"]
        assert re.fullmatch(r"[0-9a-f]{40}", revision), "Expected exact model commit hash"
        cases = load_workload(LAB/"data/workload.jsonl")
        llm = LLM(model=MODEL, revision=revision, tokenizer_revision=revision,
                  dtype="half", max_model_len=1024, max_num_seqs=4,
                  gpu_memory_utilization=0.65, enforce_eager=True,
                  enable_prefix_caching=True, trust_remote_code=False, seed=42)
        sampling = SamplingParams(temperature=0, max_tokens=32)
        batches = []
        for phase in ("first_batch", "repeat_batch"):
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = llm.chat([case["messages"] for case in cases], sampling_params=sampling, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            observations = []
            for case, item in zip(cases, outputs):
                generated = item.outputs[0]
                passed, reason, label = quality(generated.text, case["expected_label"], generated.finish_reason)
                observations.append({"case_id":case["case_id"], "quality_pass":passed,
                                     "reason":reason, "received_label":label,
                                     "finish_reason":generated.finish_reason,
                                     "prompt_tokens":len(item.prompt_token_ids),
                                     "completion_tokens":len(generated.token_ids),
                                     "response_sha256":hashlib.sha256(generated.text.encode()).hexdigest()})
            assert len(outputs) == len(cases), "Missing generated responses must not vanish from denominator."
            batches.append({"phase":phase, "requests":len(cases), "batch_seconds":elapsed,
                            "quality_successes":sum(o["quality_pass"] for o in observations),
                            "cache_reset_or_hit_verified":False, "ttft_ms":None,
                            "inter_token_latency_ms":None, "observations":observations})
        vllm_result.update(status="executed", model=MODEL, model_revision=revision,
                           vllm=installed_vllm, torch=torch.__version__, dtype="float16",
                           hardware=environment, max_model_len=1024, max_num_seqs=4,
                           workload_sha256=hashlib.sha256((LAB/"data/workload.jsonl").read_bytes()).hexdigest(),
                           batches=batches, scope="fixed fictional workload, offline batch; no provider comparison")
write_json(OUT/"optional-vllm.json", vllm_result)
print(json.dumps(vllm_result, indent=2))
'''),
md('''
## 3. Retain the evidence, then release the runtime

The files record what actually ran, the checked hardware, source commit and model revision when available. Download them if you want a reviewable GPU result. Stop/disconnect the runtime after the experiment. GPU arithmetic alone does not establish vLLM performance, and a vLLM offline batch does not validate the streaming endpoint adapter or Nebius. Managed Nebius inference remains unexecuted and no paid API is invoked.
'''),
code('''
import shutil
archive = shutil.make_archive("/content/inference-evidence", "zip", OUT)
print("Evidence archive:", archive)
from google.colab import files
files.download(archive)
''')]

for index, cell in enumerate(cells):
    cell['id'] = f'cell-{index:02d}'
notebook = {"cells":cells, "metadata":{"kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"},
                                      "language_info":{"name":"python"}, "colab":{"name":"Free GPU Inference Lab"}},
            "nbformat":4, "nbformat_minor":5}
target = Path(__file__).resolve().parents[1]/'notebooks'/'free_gpu_lab.ipynb'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=2)+'\n').encode())
print(target)
