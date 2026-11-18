"""Retrieval-augmented QA from scratch: PDF -> chunks -> embeddings -> FAISS / BM25 -> LLM.

No LangChain / LlamaIndex. The only pieces used are:
  * PyMuPDF                  to read the PDF text
  * transformers             to run the embedding model and the cross-encoder re-ranker
                             (pooling + normalisation written out below)
  * numpy + faiss            for vector search
  * BM25                     implemented below in ~30 lines of numpy
  * Ollama HTTP API          for the final answer (any LLM would do)
"""
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pymupdf
import requests
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).parent
PDF = HERE / "data" / "NIST.AI.100-1.pdf"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge's query instruction
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = os.environ.get("RAG_DEVICE", "cpu")  # the corpus is small; CPU is fast enough
TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------- document + chunking
def normalise(s: str) -> str:
    s = s.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = re.sub(r"-\s*\n\s*(?=[a-z])", "", s)  # re-join words hyphenated across lines
    return re.sub(r"\s+", " ", s).strip()


def load_document() -> tuple[str, list[int]]:
    """Whole PDF as one normalised string + the char offset where each page starts."""
    text, page_starts = "", []
    for page in pymupdf.open(PDF):
        page_starts.append(len(text))
        text += normalise(page.get_text()) + " "
    return text, page_starts


@dataclass
class Chunk:
    id: int
    start: int  # char offsets into the full document text
    end: int
    page: int
    text: str


def chunk_document(text: str, page_starts: list[int], size: int, overlap: float = 0.2) -> list[Chunk]:
    """Fixed-size windows of `size` words that overlap by `overlap` of a window."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    step = max(1, int(size * (1 - overlap)))
    chunks = []
    for i in range(0, len(words), step):
        win = words[i : i + size]
        s, e = win[0][0], win[-1][1]
        page = int(np.searchsorted(page_starts, s, side="right"))  # 1-based page number
        chunks.append(Chunk(len(chunks), s, e, page, text[s:e]))
        if i + size >= len(words):
            break
    return chunks


# ---------------------------------------------------------------- embedding model
class Embedder:
    """bge-small: run the encoder, take the [CLS] vector, L2-normalise it."""

    def __init__(self, name: str = EMBED_MODEL, device: str = DEVICE):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i : i + batch_size], padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.device)
            cls = self.model(**enc).last_hidden_state[:, 0]          # [CLS] token = sentence vector
            out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


class CrossEncoder:
    """Re-ranker: reads query and passage together, outputs one relevance logit."""

    def __init__(self, name: str = RERANK_MODEL, device: str = DEVICE):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        enc = self.tok([q for q, _ in pairs], [p for _, p in pairs], padding=True, truncation=True,
                       max_length=512, return_tensors="pt").to(self.device)
        return self.model(**enc).logits[:, 0].cpu().numpy()


# ---------------------------------------------------------------- dense retrieval
class DenseIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.embedder = embedder
        emb = embedder.encode([c.text for c in chunks])
        self.index = faiss.IndexFlatIP(emb.shape[1])  # inner product of unit vectors = cosine similarity
        self.index.add(emb)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embedder.encode([QUERY_PREFIX + query])
        scores, ids = self.index.search(q, k)
        return list(zip(ids[0].tolist(), scores[0].tolist()))


# ---------------------------------------------------------------- BM25 (from scratch)
class BM25:
    """Okapi BM25: score(q, d) = sum over query terms t of
         idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avg|d|))"""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        docs = [TOKEN_RE.findall(c.text.lower()) for c in chunks]
        self.vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
        self.tf = np.zeros((len(docs), len(self.vocab)), dtype=np.float32)  # term-frequency matrix
        for i, d in enumerate(docs):
            for t in d:
                self.tf[i, self.vocab[t]] += 1
        dl = self.tf.sum(1)
        df = (self.tf > 0).sum(0)
        n = len(docs)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        self.norm = k1 * (1 - b + b * dl / dl.mean())  # per-document length normalisation
        self.k1 = k1

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        cols = [self.vocab[t] for t in TOKEN_RE.findall(query.lower()) if t in self.vocab]
        if not cols:
            return []
        tf = self.tf[:, cols]
        scores = (self.idf[cols] * tf * (self.k1 + 1) / (tf + self.norm[:, None])).sum(1)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


# ---------------------------------------------------------------- fusion + re-ranking
def rrf(result_lists: list[list[tuple[int, float]]], k: int, c: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: each list votes 1/(c + rank) for its items. Uses ranks, not raw scores,
    so BM25 scores and cosine similarities (different scales) can be combined safely."""
    fused: dict[int, float] = {}
    for results in result_lists:
        for rank, (cid, _) in enumerate(results):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (c + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]


def rerank(query: str, candidates: list[tuple[int, float]], chunks: list[Chunk], cross_encoder, k: int):
    """A cross-encoder reads (query, passage) together and outputs one relevance score."""
    if not candidates:
        return []
    scores = cross_encoder.predict([(query, chunks[cid].text) for cid, _ in candidates])
    order = np.argsort(-np.asarray(scores))[:k]
    return [(candidates[i][0], float(scores[i])) for i in order]


# ---------------------------------------------------------------- answer generation
def answer(question: str, passages: list[Chunk], model: str = "qwen2.5:1.5b-instruct-q4_K_M") -> str:
    context = "\n\n".join(f"[{i + 1}] (page {c.page}) {c.text}" for i, c in enumerate(passages))
    prompt = (
        "Answer the question using only the numbered passages below from the NIST AI Risk Management "
        "Framework. Cite the passages you used like [1]. If the passages do not contain the answer, "
        "say you don't know.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )
    r = requests.post("http://localhost:11434/api/generate", timeout=600, json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 300,
                    # OLLAMA_NUM_GPU=0 runs the answer model on CPU (e.g. while the GPU is busy training)
                    **({"num_gpu": int(os.environ["OLLAMA_NUM_GPU"])} if "OLLAMA_NUM_GPU" in os.environ else {})},
    })
    r.raise_for_status()
    return r.json()["response"].strip()


def load_questions() -> list[dict]:
    return json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
