"""Benchmark one small model at different precisions with Ollama: speed, memory and outputs.

For every model tag:
  1. unload everything, read GPU memory used (baseline)
  2. load the model, read Ollama's own report (/api/ps: total size and the part in VRAM)
     and the GPU memory used again (nvidia-smi)
  3. run the same 10 prompts with greedy decoding; Ollama reports exact token counts and
     timings, so tokens/s = eval_count / eval_duration
Then a KV-cache probe: load the 4-bit model with growing context windows and watch memory grow.

Usage:  python benchmark.py            (needs `ollama serve` running and the tags pulled)
"""
import difflib
import json
import statistics
import subprocess
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
API = "http://localhost:11434/api"
MODELS = ["qwen2.5:1.5b-instruct-fp16", "qwen2.5:1.5b-instruct-q8_0", "qwen2.5:1.5b-instruct-q4_K_M"]
OPTIONS = {"temperature": 0, "seed": 0, "num_predict": 256, "num_ctx": 2048}
PROMPTS = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def gpu_used_mib() -> int:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return int(out.decode().strip().splitlines()[0])


def unload_all() -> None:
    for m in requests.get(f"{API}/ps").json().get("models", []):
        requests.post(f"{API}/generate", json={"model": m["name"], "keep_alive": 0})
    for _ in range(30):
        if not requests.get(f"{API}/ps").json().get("models"):
            break
        time.sleep(1)
    time.sleep(2)


def load(model: str, num_ctx: int) -> dict:
    """Load the model with an empty prompt and return Ollama's memory report for it."""
    requests.post(f"{API}/generate", json={"model": model, "prompt": "", "keep_alive": "10m",
                                           "options": {"num_ctx": num_ctx}}).raise_for_status()
    ps = next(m for m in requests.get(f"{API}/ps").json()["models"] if m["name"] == model)
    return {"size_mib": ps["size"] / 2**20, "size_vram_mib": ps["size_vram"] / 2**20}


def generate(model: str, prompt: str) -> dict:
    r = requests.post(f"{API}/chat", timeout=600, json={
        "model": model, "stream": False, "options": OPTIONS,
        "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()


def main() -> None:
    tags = {m["name"]: m for m in requests.get(f"{API}/tags").json()["models"]}
    summary, outputs = {}, {}
    for model in MODELS:
        unload_all()
        base = gpu_used_mib()
        mem = load(model, OPTIONS["num_ctx"])
        loaded = gpu_used_mib()
        generate(model, "Say hi.")  # warm-up, not timed
        runs = []
        for p in PROMPTS:
            r = generate(model, p["prompt"])
            runs.append({
                "id": p["id"], "output": r["message"]["content"],
                "gen_tokens": r["eval_count"], "gen_tok_s": r["eval_count"] / (r["eval_duration"] / 1e9),
                "prompt_tokens": r["prompt_eval_count"],
                "prompt_tok_s": r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9),
            })
            print(f"{model}  prompt {p['id']:>2}: {runs[-1]['gen_tokens']:>3} tokens at "
                  f"{runs[-1]['gen_tok_s']:.1f} tok/s", flush=True)
        outputs[model] = runs
        total_tokens = sum(x["gen_tokens"] for x in runs)
        total_time = sum(x["gen_tokens"] / x["gen_tok_s"] for x in runs)
        summary[model] = {
            "file_size_mib": tags[model]["size"] / 2**20,
            "ollama_loaded_size_mib": mem["size_mib"],
            "ollama_in_vram_mib": mem["size_vram_mib"],
            "gpu_mem_increase_mib": loaded - base,
            "gen_tok_s_overall": total_tokens / total_time,
            "gen_tok_s_median": statistics.median(x["gen_tok_s"] for x in runs),
            "prompt_tok_s_median": statistics.median(x["prompt_tok_s"] for x in runs),
            "total_generated_tokens": total_tokens,
        }
        print(json.dumps({model: summary[model]}, indent=2), flush=True)

    # how similar are the quantised outputs to the fp16 outputs (same prompt, greedy decoding)?
    ref = outputs[MODELS[0]]
    for model in MODELS[1:]:
        sims = [difflib.SequenceMatcher(None, a["output"], b["output"]).ratio() for a, b in zip(ref, outputs[model])]
        summary[model]["identical_to_fp16"] = sum(a["output"] == b["output"] for a, b in zip(ref, outputs[model]))
        summary[model]["mean_char_similarity_to_fp16"] = statistics.mean(sims)

    # KV-cache probe: same 4-bit weights, bigger context window -> bigger KV cache allocation
    kv = {}
    for ctx in [2048, 8192, 32768]:
        unload_all()
        kv[ctx] = load(MODELS[-1], ctx)
        print(f"num_ctx {ctx:>6}: Ollama reports {kv[ctx]['size_mib']:.0f} MiB", flush=True)
    unload_all()

    (OUT / "benchmark_summary.json").write_text(json.dumps({"summary": summary, "kv_probe": kv}, indent=2),
                                                encoding="utf-8")
    (OUT / "outputs.json").write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
