"""Build report/report.pdf (max 4 pages). Every number is read from the results/ files of the three tasks.

Usage:  python report/build_report.py
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
T1, T2, T3 = ROOT / "task1-lora-finetuning", ROOT / "task2-rag-from-scratch", ROOT / "task3-local-inference"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ styles
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=8.2, leading=10.2,
                      alignment=TA_LEFT, spaceAfter=3)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.4, leading=9)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=11.5, leading=14,
                    spaceBefore=4, spaceAfter=3, textColor=colors.HexColor("#1f3a5f"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=8.8, leading=11,
                    spaceBefore=4, spaceAfter=2)
TITLE = ParagraphStyle("title", parent=H1, fontSize=14, leading=17, spaceAfter=1)
MONO = ParagraphStyle("mono", parent=BODY, fontName="Courier", fontSize=7.2, leading=8.6)


def P(text, style=BODY):
    return Paragraph(text, style)


def table(rows, widths, header=True, bold_rows=(), font=7.4):
    data = [[c if isinstance(c, Paragraph) else Paragraph(str(c), ParagraphStyle(
        "cell", parent=SMALL, fontSize=font, leading=font + 1.6)) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c2cc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5"))]
    for r in bold_rows:
        style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff6dd"))]
    t.setStyle(TableStyle(style))
    return t


pct = lambda x: f"{100 * x:.1f}%"
story = []

# ================================================================== header
story += [
    P("Small open-weight LLMs: LoRA fine-tuning, RAG from scratch, local quantised inference", TITLE),
    P("Code: github.com/22AmitDeo/small-llm-lab (one folder per task). Hardware for every run: laptop "
      "RTX 3050 4 GB, Ryzen 5 5600H, 14 GB RAM. All numbers below come from the result files in the repo.", SMALL),
]

# ================================================================== TASK 1
m = {s: load(T1 / "results" / f"metrics_{s}.json") for s in ["base_zeroshot", "base_3shot", "lora", "qlora"]}
tl = {s: load(T1 / "results" / f"train_log_{s}.json") for s in ["lora", "qlora"]}
story.append(P("Task 1 - Fine-tuning Qwen2.5-0.5B-Instruct with LoRA / QLoRA", H1))
story.append(P(
    "<b>Task.</b> Turn a voice-assistant request into a fixed JSON object "
    '<font face="Courier">{"intent": ..., "slots": {type: words}}</font>. Data: Amazon MASSIVE (en-US, real human '
    "requests with intent + slot labels), 5 scenarios (alarm, calendar, smart-home, weather, transport) = 20 intents, "
    "34 slot types. Official splits: 3,991 train / 698 dev / <b>998 held-out test</b>. Every setting gets the same "
    "system prompt listing the allowed intents and slot types; greedy decoding. <b>Metric: exact match</b> = intent "
    "correct AND the full set of (slot type, value) pairs correct. Also reported: intent accuracy, slot micro-F1, "
    "valid-JSON rate."))
rows = [["Model (998 test requests)", "Exact match", "Intent acc.", "Slot F1", "Valid JSON"]]
names = {"base_zeroshot": "Base, instructions only", "base_3shot": "Base, + 3 solved examples in prompt",
         "lora": "LoRA (bf16 base)", "qlora": "QLoRA (4-bit NF4 base)"}
for s, n in names.items():
    rows.append([n, pct(m[s]["exact_match"]), pct(m[s]["intent_acc"]), f"{m[s]['slot_f1']:.3f}", pct(m[s]["valid_json"])])
story.append(table(rows, [6.2, 2.4, 2.4, 2.0, 2.2], bold_rows=(3, 4)))
story.append(Spacer(1, 3))
story.append(P(
    f"The base model already writes valid JSON almost every time, but it guesses labels (e.g. \"wake me up at five "
    f"am\" &rarr; transport_query) and invents slot types. Few-shot prompting roughly doubles exact match but stays "
    f"under 10%. Fine-tuning moves exact match from {pct(m['base_zeroshot']['exact_match'])} to "
    f"{pct(m['lora']['exact_match'])}. Training took {tl['lora']['train_seconds'] / 60:.0f} min (LoRA) and "
    f"{tl['qlora']['train_seconds'] / 60:.0f} min (QLoRA); dev loss fell from {tl['lora']['history'][0]['dev_loss']:.3f} "
    f"to {tl['lora']['history'][-1]['dev_loss']:.3f} (LoRA). The 4-bit base starts worse "
    f"(dev loss {tl['qlora']['history'][0]['dev_loss']:.3f}) but the adapter recovers almost all of the gap."))

a = tl["lora"]["args"]
story.append(P("LoRA settings and why", H2))
story.append(table([
    ["Setting", "Value", "Reason"],
    ["Rank r / alpha", f"{a['rank']} / {a['alpha']}", "The task is narrow (a fixed format + label set), so a low rank is enough; "
     "alpha = 2r keeps the update scale (alpha/r = 2) in the usual range so the default LR works."],
    ["Target modules", "q, k, v, o, gate, up, down", "Adapting the MLP as well as attention lets the model learn new "
     "label vocabulary, not only where to attend. Cost: "
     f"{tl['lora']['trainable_params'] / 1e6:.1f}M trainable params ({100 * tl['lora']['trainable_params'] / tl['lora']['total_params']:.2f}% of the model)."],
    ["Learning rate", f"{a['lr']:.0e}, cosine, 5% warm-up", "LoRA tolerates ~10x higher LR than full fine-tuning because "
     "only small fresh matrices move; cosine decay settles the end of training."],
    ["Epochs / batch", f"{a['epochs']} / {a['micro_batch'] * a['grad_accum']} "
     f"({a['micro_batch']} x {a['grad_accum']} accumulation)", "Dev loss was still falling after epoch 1 and only slightly "
     "after epoch 2, so 2 epochs. Micro-batch 2: Qwen's 152k-token vocabulary makes the logits large on a 4 GB GPU."],
    ["Dropout / loss", f"{a['dropout']} / answer tokens only", "Light regularisation; prompt tokens are masked "
     "(label -100) so the model learns to produce the JSON, not to copy the instructions."],
], [2.6, 3.4, 9.6]))

story.append(P("Three test requests the fine-tuned model still gets wrong", H2))
story.append(table([
    ["Request", "Gold vs. prediction", "Likely reason"],
    ["should i bring an umbrella today", 'gold slots {date: today}; pred adds weather_descriptor: "umbrella"',
     "Label noise, not model error: in training, 'umbrella' is tagged weather_descriptor in 23 of 23 requests. The "
     "model learned the training convention; this test label breaks it. Exact match punishes it anyway."],
    ["my set alarms", "gold alarm_query; pred alarm_set",
     "Surface-word shortcut. 'set' appears in 55% of alarm_query and 59% of alarm_set training requests, so the word "
     "itself is not informative; here it is an adjective ('alarms that are set') and there is no verb to disambiguate."],
    ["please turn off the light of my son's room", 'gold house_place: "my son\'s room"; pred relation: "son\'s room"',
     "Overlapping slot types: 'son' is a typical relation word, and the phrase is a place described through a person. "
     "The model picks the wrong reading and the wrong span boundary."],
], [4.1, 4.6, 6.9]))
story.append(P(
    "Error profile of LoRA on the full test set (292 errors): missed slot 47%, wrong intent 15%, wrong slot type 13%, "
    "right type but wrong span 13%, extra slot 12%. Intent confusions are mostly calendar_query vs calendar_set. "
    "Several 'errors' are inconsistent gold labels (e.g. 'week' is tagged date 61 times but time 6 times in training).",
    SMALL))

story.append(P("GPU memory to fine-tune a 7B model (working in task1-lora-finetuning/memory_estimate.py)", H2))
story.append(P(
    "Shapes of Llama-2-7B: 6.74B params, h = 4096, MLP 11008, 32 layers. AdamW, mixed precision, batch 1, 512 tokens, "
    "gradient checkpointing (activations ~0.3 GB) + ~1 GB CUDA overhead. GB = 2<super>30</super> bytes."))
story.append(table([
    ["", "Per-parameter cost", "Working", "Total"],
    ["(a) Full FT", "bf16 weights 2 + bf16 grads 2 + fp32 master copy 4 + Adam m, v 8 = 16 bytes",
     "6.74B x 16 B = 100.4 GB, + 1.3 GB", "<b>~102 GB</b> (2 x 80 GB GPUs, or ZeRO/FSDP)"],
    ["(b) LoRA r=16", "frozen bf16 weights 2 bytes; only adapters train",
     "adapters = 32 x [4 x 16 x (4096+4096) + 3 x 16 x (4096+11008)] = 40.0M (0.59%); 40M x 16 B = 0.6 GB; "
     "6.74B x 2 B = 12.6 GB; + 1.3 GB", "<b>~14 GB</b> (one 16-24 GB GPU)"],
    ["(c) QLoRA", "frozen NF4 weights 0.5 byte + ~0.127 bit/param quantisation constants",
     "6.74B x 0.5 B = 3.1 GB; constants 0.1 GB; adapters 0.6 GB; + 1.3 GB", "<b>~5 GB</b> (8 GB card / free T4)"],
], [1.9, 4.6, 6.4, 2.7]))
story.append(P(
    f"Sanity check on the 0.5B run: measured peak {tl['lora']['peak_gpu_mem_gb']:.2f} GB (LoRA) vs "
    f"{tl['qlora']['peak_gpu_mem_gb']:.2f} GB (QLoRA). At 0.5B the weights are small, so the logits of a 152k vocabulary "
    "and activations dominate and the 4-bit saving is modest; at 7B the weights dominate and the saving is ~3x.", SMALL))

# ================================================================== TASK 2
r = load(T2 / "results" / "retrieval_results.json")
grades = load(T2 / "results" / "answer_grades.json")
story.append(P("Task 2 - Retrieval-augmented QA from scratch", H1))
story.append(P(
    "<b>Corpus:</b> NIST AI Risk Management Framework 1.0 (48 pages, 15.7k words). <b>Pipeline:</b> PyMuPDF text &rarr; "
    "fixed word windows with 20% overlap &rarr; bge-small-en-v1.5 [CLS] embeddings, L2-normalised, FAISS inner-product "
    "index; BM25 written in NumPy; hybrid = reciprocal-rank fusion of both top-20 lists; optional cross-encoder re-rank "
    "(ms-marco-MiniLM-L-6) &rarr; top 5 chunks &rarr; Qwen2.5-1.5B (4-bit, Ollama) answers with citations. "
    "<b>Test set:</b> 15 hand-written questions, each with its verbatim source passage (questions.json). A chunk is a hit "
    "if it covers &ge;50% of the passage."))
rows = [["Chunk size", "Method", "recall@1", "recall@5", "MRR"]]
for key, v in r.items():
    size, meth = key.split(" | ")
    rows.append([size.replace("w", " words") + f" ({v['chunks']} chunks)", meth, f"{v['recall@1']:.2f}",
                 f"{v['recall@5']:.2f}", f"{v['MRR']:.2f}"])
story.append(table(rows, [4.2, 3.6, 2.0, 2.0, 2.0], bold_rows=(4,)))
story.append(Spacer(1, 3))
story.append(P(
    "<b>Which chunk size worked better.</b> With dense search alone the two sizes tie on recall@5 (0.93); 300-word chunks "
    "are slightly better at rank 1 because a bigger chunk carries more context for the embedding. But the comparison is "
    "not equal: five 300-word chunks cover ~10% of the document while five 80-word chunks cover ~2.5%, so big chunks get "
    "recall partly for free and hand the LLM far more irrelevant text. The best full pipeline was <b>80-word chunks + "
    "hybrid + re-rank (recall@5 1.00, MRR 0.93)</b>: small chunks hold one idea each, so the cross-encoder can judge them "
    "precisely; inside a 300-word chunk the answer is diluted by neighbouring content and re-ranking helped less (MRR 0.84). "
    f"End-to-end with that setting, {grades['correct']}/{grades['total']} answers were correct (hand-graded); all 3 wrong "
    "answers had the right passage in context - the 1.5B generator misread a table, used a neighbouring sentence, or "
    "returned only citation markers."))
story.append(P("Two retrieval failures and fixes", H2))
story.append(table([
    ["Question (setting)", "What happened", "Fix"],
    ["Q9 'Which GOVERN subcategory covers retiring AI systems safely?' (dense, 80 words: not in top 5)",
     "The source says 'decommissioning and phasing out', not 'retiring'. All rows of the GOVERN table embed to nearly the "
     "same vector, so generic table headers ranked first.",
     "Keyword signal: BM25 ranks it #1 ('GOVERN', 'safely' match exactly). Hybrid + re-rank gets it to #2. Chunking table "
     "rows individually with their IDs would also help."],
    ["Q2 'How does the AI RMF define risk?' (BM25, both sizes: not in top 5)",
     "Every query word ('AI', 'RMF', 'risk') is everywhere, so IDF is low and BM25 rewards chunks that repeat 'AI RMF'. "
     "The source says 'refers to', not 'define'.",
     "Semantic signal: dense search finds it at rank 1-2; hybrid + re-rank puts it at #1 (80 words). Query rewriting "
     "('risk refers to', 'definition of risk') would help BM25."],
], [4.6, 5.7, 5.3]))
story.append(P("What BM25 adds, and what a re-ranker does", H2))
story.append(P(
    "<b>BM25</b> scores a chunk by the query words it contains: rare words count more (IDF), repeats count with diminishing "
    "returns (k1), and long chunks are penalised (b). Embedding search compresses a chunk into one vector of 'meaning', "
    "which handles paraphrase but blurs exact tokens - identifiers like 'GOVERN 1.7', names, numbers and rare jargon. BM25 "
    "keeps those exact matches, so the two fail on different questions (Q9 vs Q2 above) and fusing them is more robust. "
    "<b>A re-ranker</b> is a cross-encoder: instead of comparing two separately computed vectors, it reads the question "
    "and one candidate passage together through a transformer and outputs a relevance score, so every word of the "
    "question can attend to every word of the passage. That is far more accurate but costs a full forward pass per pair, "
    "so it only re-orders a short list (here the top 20 from hybrid search) rather than searching the whole corpus."))

# ================================================================== TASK 3
b = load(T3 / "results" / "benchmark_summary.json")
q = load(T3 / "results" / "quality_grades.json")
S, kv = b["summary"], b["kv_probe"]
story.append(P("Task 3 - Running Qwen2.5-1.5B-Instruct locally with Ollama at three precisions", H1))
story.append(P(
    "Same model, same 10 prompts, greedy decoding (temperature 0, seed 0, max 256 new tokens, 2k context). Speed comes "
    "from Ollama's own token counts and timers; memory from Ollama's loaded size (/api/ps) and the rise in GPU memory "
    "(nvidia-smi) when the model is loaded."))
rows = [["Setting", "File size", "Loaded (Ollama)", "of which in VRAM", "GPU mem rise", "Generation tok/s",
         "Prompt tok/s", "Correct (of 10)"]]
for name, v in S.items():
    tag = name.split("-")[-1]
    rows.append([tag, f"{v['file_size_mib']:.0f} MiB", f"{v['ollama_loaded_size_mib']:.0f} MiB",
                 f"{v['ollama_in_vram_mib']:.0f} MiB", f"{v['gpu_mem_increase_mib']:.0f} MiB",
                 f"{v['gen_tok_s_overall']:.1f}", f"{v['prompt_tok_s_median']:.0f}", f"{q['score'][name]}"])
story.append(table(rows, [1.6, 1.7, 2.1, 2.1, 1.9, 2.2, 1.8, 2.0]))
story.append(Spacer(1, 3))
story.append(P(q["observations"]))
story.append(P(q["prompt_table_note"], SMALL))

story.append(P("KV cache and why quantisation saves memory", H2))
k = {int(c): v["size_mib"] for c, v in kv.items()}
per_tok = 2 * 28 * 2 * 128 * 2
story.append(P(
    "<b>KV cache.</b> When a transformer generates token t, attention needs the key and value vectors of every earlier "
    "token in every layer. Recomputing them for each new token would make generation quadratic, so they are computed "
    "once and stored: that store is the KV cache. It grows linearly with context length and with the number of "
    "concurrent sequences. For Qwen2.5-1.5B (28 layers, 2 KV heads x 128 dims thanks to grouped-query attention, fp16) "
    f"it is 2 (K,V) x 28 x 2 x 128 x 2 bytes = {per_tok // 1024} KiB per token. Measured: loading the same 4-bit model "
    f"with a 2k / 8k / 32k context window took {k[2048]:.0f} / {k[8192]:.0f} / {k[32768]:.0f} MiB - "
    f"+{k[32768] - k[2048]:.0f} MiB for 30k extra tokens, close to the {per_tok * 30720 / 2**20:.0f} MiB the formula gives. "
    "<b>Quantisation</b> stores each weight in fewer bits: 16 bits &rarr; ~8.5 (q8_0) &rarr; ~4.8 (q4_K_M, blocks of "
    "weights share a scale so 4-bit codes can be mapped back to real values). Memory scales with bits per weight, and "
    "because token-by-token decoding is limited by how fast weights are read from memory, fewer bytes also means more "
    "tokens per second. The cost is rounding error, which shows up first in brittle tasks (exact arithmetic, counting)."))

story.append(P("Serving 20 concurrent users on one 24 GB GPU (back-of-envelope)", H2))
story.append(P(
    "<b>Assumptions:</b> model Qwen2.5-7B-Instruct (28 layers, 4 KV heads x 128, GQA); each user needs up to 4k tokens "
    "of context (prompt + answer); vLLM with PagedAttention and continuous batching; ~2 GB for activations, CUDA "
    "graphs and fragmentation. <b>KV per token</b> = 2 x 28 x 4 x 128 x 2 B = 56 KiB (fp16). "
    "<b>Worst case</b>: 20 users x 4,096 tokens x 56 KiB = 4.4 GiB."))
story.append(table([
    ["Plan", "Weights", "KV for 20 x 4k", "Overhead", "Total", "Verdict"],
    ["7B in fp16", "15.2 GB", "4.4 GB", "~2 GB", "~21.6 GB", "Fits, little headroom for longer contexts"],
    ["<b>7B AWQ/GPTQ 4-bit (chosen)</b>", "~5.5 GB", "4.4 GB (fp16) or 2.2 GB (fp8 KV)", "~2 GB", "~12 GB",
     "Fits with ~10 GB spare: room for 8k contexts or ~40 users"],
], [3.6, 1.6, 3.4, 1.6, 1.5, 4.0]))
story.append(P(
    "Why this works: vLLM allocates KV memory in small pages only as tokens are produced, so 20 users rarely need the "
    "worst case at once, and continuous batching runs all active requests in the same forward pass, so 20 users cost "
    "about the same weight reads as one. Decode speed is memory-bound: each step reads the ~5.5 GB of weights plus the "
    "active KV cache (up to ~4.4 GB), so on a ~1 TB/s card (RTX 3090/4090 class) a step takes ~10 ms and yields one token "
    "for every active user - on the order of 50 tokens/s per user at full load, well above reading speed (a ~300 GB/s "
    "card such as an L4 would give roughly a third of that). Safeguards: cap max context (max_model_len), limit concurrent sequences "
    "(max_num_seqs = 20-32) and enable prefix caching for a shared system prompt.", SMALL))

doc = SimpleDocTemplate(str(ROOT / "report" / "report.pdf"), pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                        topMargin=1.1 * cm, bottomMargin=1.1 * cm, title="Small LLM lab report")
doc.build(story)
import pymupdf  # noqa: E402  (only to report the page count)
print("pages:", pymupdf.open(ROOT / "report" / "report.pdf").page_count)
