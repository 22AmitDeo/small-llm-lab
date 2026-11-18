# Task 2 – Retrieval (RAG) from scratch

Question answering over the **NIST AI Risk Management Framework (AI RMF 1.0, NIST AI 100-1)**:
48 pages, ~15,700 words, public domain. No LangChain / LlamaIndex: just an embedding model,
FAISS, a hand-written BM25 and a local LLM through Ollama's HTTP API.

## Pipeline (`rag.py`)

1. **Read + normalise** the PDF with PyMuPDF (re-join words hyphenated across lines, collapse whitespace).
2. **Chunk** into fixed windows of N words with 20 % overlap; every chunk keeps its character offsets and page.
3. **Dense index**: `BAAI/bge-small-en-v1.5` → [CLS] vector → L2-normalise → FAISS `IndexFlatIP`
   (inner product of unit vectors = cosine similarity). Queries get bge's retrieval instruction prefix.
4. **BM25** (Okapi, k1 = 1.5, b = 0.75) written in NumPy over a term-frequency matrix.
5. **Hybrid**: reciprocal rank fusion of the dense and BM25 top-20 lists (rank-based, so the two score scales don't matter).
6. **Re-rank**: `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (question, chunk) pair; keep the top 5.
7. **Answer**: `qwen2.5:1.5b-instruct-q4_K_M` (Ollama) answers from the 5 numbered chunks and cites them.

## Test set (`questions.json`)

15 questions written by hand, each with the **verbatim source passage** and its PDF page. They mix
definitions (Q2, Q4), exact identifiers (Q9 "GOVERN 1.7"), and paraphrases that share few words with
the source (Q5, Q13, Q14). `evaluate_retrieval.py` checks every passage is found verbatim in the text.

A retrieved chunk is a **hit** if it covers ≥ 50 % of the gold passage's characters.

## Results (`results/retrieval_results.json`)

| Chunk size | Method | recall@1 | recall@5 | MRR |
|---|---|---|---|---|
| 80 words (245 chunks) | dense | 0.67 | 0.93 | 0.79 |
| | BM25 | 0.87 | 0.87 | 0.87 |
| | hybrid (RRF) | 0.80 | 0.87 | 0.83 |
| | **hybrid + re-rank** | **0.87** | **1.00** | **0.93** |
| 300 words (66 chunks) | dense | 0.73 | 0.93 | 0.80 |
| | BM25 | 0.73 | 0.93 | 0.82 |
| | hybrid (RRF) | 0.87 | 0.93 | 0.88 |
| | hybrid + re-rank | 0.73 | 1.00 | 0.84 |

End-to-end, with the best setting (80 words, hybrid + re-rank) feeding the 1.5B model, **12/15 answers
were correct** (graded by hand, `results/answer_grades.json`). All 3 wrong answers had the gold passage
in the retrieved context: the small generator, not retrieval, was the bottleneck.

## Run

```bash
curl -L -o data/NIST.AI.100-1.pdf https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
python evaluate_retrieval.py                     # recall@1/@5 and MRR for both chunk sizes
ollama pull qwen2.5:1.5b-instruct-q4_K_M
python ask.py "What are the characteristics of trustworthy AI?"
python ask.py --all                              # answers for the 15 test questions
```
