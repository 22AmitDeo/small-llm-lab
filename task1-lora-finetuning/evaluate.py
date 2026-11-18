"""Evaluate base vs fine-tuned Qwen2.5-0.5B-Instruct on the held-out MASSIVE test split.

Settings compared (same system prompt, greedy decoding):
    base_zeroshot   base model, instructions only
    base_3shot      base model, instructions + 3 solved training examples
    lora / qlora    base model + trained adapter, instructions only

Usage:
    python evaluate.py base_zeroshot
    python evaluate.py base_3shot
    python evaluate.py lora
    python evaluate.py qlora
"""
import json
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import DATA, HERE, MODEL_ID, build_messages, load_jsonl, load_labels, parse_output, score

setting = sys.argv[1]
BATCH = 32

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"  # decoder-only models must be left-padded for batched generation
if setting == "qlora":
    # evaluate the QLoRA adapter on the same 4-bit base it was trained against
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0})
    model = PeftModel.from_pretrained(model, HERE / "outputs" / "qlora" / "adapter")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).cuda()
if setting == "lora":
    model = PeftModel.from_pretrained(model, HERE / "outputs" / "lora" / "adapter")
    model = model.merge_and_unload()  # fold B@A into the weights: no extra latency at inference
model.eval()

labels = load_labels()
test = load_jsonl(DATA / "test.jsonl")
train = load_jsonl(DATA / "train.jsonl")
# 3 fixed demonstrations: one with no slots, one with one slot, one with several slots
shots = []
if setting == "base_3shot":
    by_n = {}
    for ex in train:
        by_n.setdefault(min(len(ex["target"]["slots"]), 2), ex)
    shots = [by_n[0], by_n[1], by_n[2]]

stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
raws, t0 = [], time.time()
for i in range(0, len(test), BATCH):
    chunk = test[i : i + BATCH]
    prompts = [tok.apply_chat_template(build_messages(ex["text"], labels, shots), tokenize=False,
                                       add_generation_prompt=True) for ex in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=96, do_sample=False, eos_token_id=stop_ids,
                             pad_token_id=tok.pad_token_id)
    raws += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"{min(i + BATCH, len(test))}/{len(test)}", flush=True)

preds = [parse_output(r) for r in raws]
metrics = score(preds, [ex["target"] for ex in test])
metrics["setting"] = setting
metrics["seconds"] = round(time.time() - t0, 1)
print(json.dumps(metrics, indent=2))

res_dir = HERE / "results"
res_dir.mkdir(exist_ok=True)
(res_dir / f"metrics_{setting}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
with (res_dir / f"predictions_{setting}.jsonl").open("w", encoding="utf-8") as f:
    for ex, raw, pr in zip(test, raws, preds):
        f.write(json.dumps({"id": ex["id"], "text": ex["text"], "gold": ex["target"], "raw": raw, "pred": pr},
                           ensure_ascii=False) + "\n")
