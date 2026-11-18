"""LoRA / QLoRA fine-tuning of Qwen2.5-0.5B-Instruct on the request -> JSON task.

A plain PyTorch training loop (no Trainer) so every step is visible:
  * prompt tokens are masked out of the loss (label -100), so the model only learns to
    produce the JSON answer, not to repeat the instructions;
  * LoRA adapters sit on all linear layers of attention and MLP blocks;
  * --qlora loads the frozen base weights in 4-bit NF4 instead of bf16.

Usage:
    python train_lora.py                 # LoRA,  base weights in bf16
    python train_lora.py --qlora         # QLoRA, base weights in 4-bit NF4
"""
import argparse
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from common import DATA, HERE, MODEL_ID, build_messages, load_jsonl, load_labels, target_text

p = argparse.ArgumentParser()
p.add_argument("--qlora", action="store_true")
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--dropout", type=float, default=0.05)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--micro_batch", type=int, default=2)
p.add_argument("--grad_accum", type=int, default=8)
p.add_argument("--max_len", type=int, default=384)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
run_name = "qlora" if args.qlora else "lora"
out_dir = HERE / "outputs" / run_name
out_dir.mkdir(parents=True, exist_ok=True)

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ------------------------------------------------------------------ model
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if args.qlora:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0},
                                                 torch_dtype=torch.bfloat16)
    # Note: peft's prepare_model_for_kbit_training() is NOT used. It upcasts the non-quantised layers
    # (incl. the 136M-param tied embedding / output matrix) to fp32, which ran out of memory on a 4 GB GPU.
    # The LoRA adapters only sit on Linear layers, so gradients reach them without it.
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).cuda()

lora_cfg = LoraConfig(
    r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
    target_modules=TARGETS, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
for n, prm in model.named_parameters():  # keep the trainable adapter weights in fp32
    if prm.requires_grad:
        prm.data = prm.data.float()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ------------------------------------------------------------------ data
labels = load_labels()


def encode(ex: dict) -> dict:
    prompt = tok.apply_chat_template(build_messages(ex["text"], labels), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(target_text(ex["target"]) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[: args.max_len]
    lab = ([-100] * len(p_ids) + a_ids)[: args.max_len]
    return {"input_ids": ids, "labels": lab}


train = [encode(ex) for ex in load_jsonl(DATA / "train.jsonl")]
dev = [encode(ex) for ex in load_jsonl(DATA / "dev.jsonl")][:300]
print(f"train {len(train)}  dev {len(dev)}  max train len {max(len(x['input_ids']) for x in train)}")


def collate(batch: list[dict]) -> dict:
    L = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id)
    lab = torch.full((len(batch), L), -100)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"])
        lab[i, :n] = torch.tensor(x["labels"])
        att[i, :n] = 1
    return {"input_ids": ids.cuda(), "labels": lab.cuda(), "attention_mask": att.cuda()}


def batches(data: list[dict], bs: int, shuffle: bool):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield collate([data[j] for j in idx[i : i + bs]])


@torch.no_grad()
def dev_loss() -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for b in batches(dev, 4, False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**b)
        n_tok = (b["labels"][:, 1:] != -100).sum().item()
        tot += out.loss.item() * n_tok
        cnt += n_tok
    model.train()
    return tot / cnt


# ------------------------------------------------------------------ train
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(train) / (args.micro_batch * args.grad_accum))
total_steps = steps_per_epoch * args.epochs
sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

torch.cuda.reset_peak_memory_stats()
log = {"args": vars(args), "trainable_params": trainable, "total_params": total, "history": []}
log["history"].append({"step": 0, "dev_loss": dev_loss()})
print(f"step 0  dev_loss {log['history'][-1]['dev_loss']:.4f}")

model.train()
step, t0 = 0, time.time()
for epoch in range(args.epochs):
    micro = 0
    running = 0.0
    for b in batches(train, args.micro_batch, True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**b).loss / args.grad_accum
        loss.backward()
        running += loss.item()
        micro += 1
        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:  # running holds the sum of 25 optimiser steps' losses
                print(f"epoch {epoch} step {step}/{total_steps}  train_loss {running / 25:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
                log["history"].append({"step": step, "train_loss": running / 25})
                running = 0.0
    d = dev_loss()
    log["history"].append({"step": step, "epoch": epoch + 1, "dev_loss": d})
    print(f"end of epoch {epoch + 1}: dev_loss {d:.4f}", flush=True)

log["train_seconds"] = time.time() - t0
log["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
print(f"done in {log['train_seconds']:.0f}s, peak GPU memory {log['peak_gpu_mem_gb']:.2f} GB")
model.save_pretrained(out_dir / "adapter")
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / f"train_log_{run_name}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
