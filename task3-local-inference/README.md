# Task 3 – Running a model locally and optimisation

**Qwen2.5-1.5B-Instruct** in [Ollama](https://ollama.com) (llama.cpp backend) at three precisions:
`fp16`, `q8_0` (8-bit) and `q4_K_M` (4-bit). Same 10 prompts (`prompts.json`), greedy decoding
(temperature 0, seed 0, max 256 new tokens, 2k context).

`benchmark.py` for each model: unloads everything, loads the model, reads Ollama's memory report
(`/api/ps`) and the GPU memory rise (`nvidia-smi`), then runs the 10 prompts and takes tokens/s from
Ollama's own counters (`eval_count / eval_duration`). It ends with a KV-cache probe that loads the 4-bit
model with a 2k / 8k / 32k context window.

## Results (RTX 3050 Laptop 4 GB, Ryzen 5 5600H)

| Setting | File | Loaded | In VRAM | Gen. tok/s (GPU) | Gen. tok/s (CPU only) | Prompt tok/s (GPU) | Correct /10 |
|---|---|---|---|---|---|---|---|
| fp16 | 2950 MiB | 3544 MiB | 2263 MiB (64%) | 9.7 | 3.9 | 263 | 10 |
| q8_0 | 1570 MiB | 1687 MiB | 1687 MiB (100%) | 77.4 | 7.8 | 1528 | 10 |
| q4_K_M | 940 MiB | 1057 MiB | 1057 MiB (100%) | 99.4 | 13.9 | 1579 | 10 |

* fp16 does not fit next to the desktop's own GPU usage, so Ollama splits it: 64% on GPU, the rest on
  CPU, and the CPU part sets the pace.
* Quality: all three answer all 10 prompts correctly (`results/quality_grades.json`); q8 reproduces
  the fp16 text exactly on 7/10 prompts, q4 on 6/10. The differences are wording and length only.
* KV-cache probe (4-bit model): 1057 / 1304 / 2024 MiB at 2k / 8k / 32k context. That is +967 MiB for
  30,720 tokens; the formula 2 × 28 layers × 2 KV heads × 128 × 2 bytes = 28 KiB/token predicts 840 MiB.

Files: `results/benchmark_summary_{gpu,cpu}.json`, `results/outputs_{gpu,cpu}.json`.

## Run

```bash
ollama pull qwen2.5:1.5b-instruct-fp16
ollama pull qwen2.5:1.5b-instruct-q8_0
ollama pull qwen2.5:1.5b-instruct-q4_K_M
RESULTS_TAG=_gpu python benchmark.py   # with `ollama serve` running
```

For the CPU-only numbers, start `ollama serve` with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) and run with
`RESULTS_TAG=_cpu`. (In this run the CPU-only pass happened because Ollama started while the GPU was full from
QLoRA training and found no usable GPU; the model, prompts and settings were otherwise identical.)
