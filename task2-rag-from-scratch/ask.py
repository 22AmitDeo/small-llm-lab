"""End-to-end QA: retrieve the top-5 chunks and let a local LLM (via Ollama) answer with citations.

Usage:
    python ask.py "What are the characteristics of trustworthy AI?"
    python ask.py --all          # answer all 15 test questions, save to results/answers.jsonl
"""
import json
import sys



from rag import (HERE, BM25, CrossEncoder, DenseIndex, Embedder, answer, chunk_document, load_document,
                 load_questions, rerank, rrf)

CHUNK_SIZE = 80  # best setting in evaluate_retrieval.py (hybrid + rerank: recall@5 1.00, MRR 0.93)

text, page_starts = load_document()
chunks = chunk_document(text, page_starts, CHUNK_SIZE)
dense, bm25 = DenseIndex(chunks, Embedder()), BM25(chunks)
cross = CrossEncoder()


def retrieve(q: str, k: int = 5):
    cands = rrf([dense.search(q, 20), bm25.search(q, 20)], 20)
    return [chunks[cid] for cid, _ in rerank(q, cands, chunks, cross, k)]


if sys.argv[1:] == ["--all"]:
    out = (HERE / "results" / "answers.jsonl").open("w", encoding="utf-8")
    for q in load_questions():
        passages = retrieve(q["question"])
        a = answer(q["question"], passages)
        print(f"\nQ{q['id']}: {q['question']}\nA: {a}")
        out.write(json.dumps({"id": q["id"], "question": q["question"], "answer": a,
                              "retrieved_pages": [c.page for c in passages], "gold_page": q["page"]}) + "\n")
else:
    question = " ".join(sys.argv[1:])
    passages = retrieve(question)
    print(answer(question, passages))
    print("\nsources:", ", ".join(f"p.{c.page}" for c in passages))
