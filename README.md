# small-llm-lab

Three hands-on tasks with small open-weight language models, all run on a laptop RTX 3050 (4 GB).
The report is [`report/report.pdf`](report/report.pdf) (3 pages).

| Folder | Task | Headline result |
|---|---|---|
| [`task1-lora-finetuning/`](task1-lora-finetuning/) | LoRA / QLoRA fine-tuning of Qwen2.5-0.5B-Instruct: voice-assistant request → fixed JSON (MASSIVE dataset) | Exact match on 998 held-out requests: base 4.9% → LoRA 70.7%, QLoRA 69.2% |
| [`task2-rag-from-scratch/`](task2-rag-from-scratch/) | RAG over the 48-page NIST AI RMF with bge-small + FAISS + hand-written BM25, no frameworks | recall@5 = 1.00 (80-word chunks, hybrid + re-rank) on 15 hand-written questions |
| [`task3-local-inference/`](task3-local-inference/) | Qwen2.5-1.5B in Ollama at fp16 / q8_0 / q4_K_M | 9.7 → 77.4 → 99.4 tok/s on GPU; 10/10 prompts correct at every precision |
| [`report/`](report/) | PDF report, built from the result files by `build_report.py` | |

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu124   # or your CUDA version
.venv/Scripts/pip install -r requirements.txt
```

Each task folder has its own README with the data download and run commands.
