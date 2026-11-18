"""Measure retrieval accuracy for two chunk sizes and four retrieval methods.

A retrieved chunk counts as a hit if it covers at least half of the characters of the
question's gold source passage (passages can straddle chunk boundaries).

Metrics: recall@1, recall@5 (gold passage in the top 1 / top 5) and MRR.

Usage:  python evaluate_retrieval.py
"""
import json
import re

import numpy as np


from rag import (HERE, BM25, CrossEncoder, DenseIndex, Embedder, chunk_document, load_document,
                 load_questions, normalise, rerank, rrf)

CHUNK_SIZES = [80, 300]  # words per chunk (20% overlap)
K = 5

text, page_starts = load_document()
print(f"document: {len(page_starts)} pages, {len(text.split()):,} words")
questions = load_questions()
for q in questions:  # locate each gold passage in the document text (it may appear more than once)
    gold = normalise(q["source_passage"])
    q["spans"] = [(m.start(), m.start() + len(gold)) for m in re.finditer(re.escape(gold), text)]
    assert q["spans"], f"gold passage for {q['id']} not found verbatim in the document"

embedder = Embedder()
cross = CrossEncoder()


def is_hit(chunk, spans) -> bool:
    return any(min(chunk.end, e) - max(chunk.start, s) >= 0.5 * (e - s) for s, e in spans)


results, failures = {}, {}
for size in CHUNK_SIZES:
    chunks = chunk_document(text, page_starts, size)
    dense, bm25 = DenseIndex(chunks, embedder), BM25(chunks)
    methods = {
        "dense": lambda q: dense.search(q, K),
        "bm25": lambda q: bm25.search(q, K),
        "hybrid (RRF)": lambda q: rrf([dense.search(q, 20), bm25.search(q, 20)], K),
        "hybrid + rerank": lambda q: rerank(q, rrf([dense.search(q, 20), bm25.search(q, 20)], 20), chunks, cross, K),
    }
    for name, fn in methods.items():
        ranks = []
        for q in questions:
            got = fn(q["question"])
            rank = next((r + 1 for r, (cid, _) in enumerate(got) if is_hit(chunks[cid], q["spans"])), None)
            ranks.append(rank)
            if name == "dense" and (rank is None or rank > 1):
                failures.setdefault(size, []).append({
                    "id": q["id"], "question": q["question"], "gold_page": q["page"], "rank": rank,
                    "top1_page": chunks[got[0][0]].page, "top1_text": chunks[got[0][0]].text[:300]})
        r1 = np.mean([r == 1 for r in ranks])
        r5 = np.mean([r is not None for r in ranks])
        mrr = np.mean([1 / r if r else 0 for r in ranks])
        results[f"{size}w | {name}"] = {"chunks": len(chunks), "recall@1": r1, "recall@5": r5, "MRR": mrr,
                                       "ranks": ranks}
        print(f"chunk {size:>3} words ({len(chunks):>3} chunks) | {name:<16} "
              f"recall@1 {r1:.2f}  recall@5 {r5:.2f}  MRR {mrr:.2f}   ranks {ranks}")

out = HERE / "results"
out.mkdir(exist_ok=True)
(out / "retrieval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
(out / "dense_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
