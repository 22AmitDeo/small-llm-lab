"""Back-of-envelope GPU memory needed to fine-tune a 7B model: full FT vs LoRA vs QLoRA.

Uses the real shapes of Llama-2-7B (a standard "7B" model) so every number can be traced:
    hidden h = 4096, MLP size = 11008, layers = 32, heads = 32, vocab = 32000, params = 6.74B

Assumptions (stated so they can be challenged):
  * mixed-precision training with AdamW
  * batch 1, sequence 512 tokens, gradient checkpointing ON for all three settings
  * "overhead" = CUDA context + allocator fragmentation, taken as ~1 GB
  * GB here means GiB (2^30 bytes)
"""
GB = 1024**3
P = 6.74e9                       # parameters
h, ffn, L, heads, V = 4096, 11008, 32, 32, 32000
s, b = 512, 1                    # sequence length, batch size
OVERHEAD = 1.0                   # GB


def activations_gb(checkpointing: bool) -> float:
    # Per layer, a transformer block keeps about s*b*h*(34 + 5*heads*s/h) bytes of activations in
    # 16-bit (Korthikanti et al., 2022). With checkpointing we keep only each layer's input
    # (2*s*b*h bytes) and recompute one layer at a time during the backward pass.
    per_layer = s * b * h * (34 + 5 * heads * s / h)
    logits = s * b * V * 4  # fp32 logits for the loss
    if checkpointing:
        return (L * 2 * s * b * h + per_layer + logits) / GB
    return (L * per_layer + logits) / GB


def lora_params(r: int) -> int:
    # LoRA on one Linear(d_in -> d_out) adds A (r x d_in) and B (d_out x r) = r*(d_in + d_out) params
    attn = 4 * r * (h + h)                   # q, k, v, o are all 4096 -> 4096
    mlp = 3 * r * (h + ffn)                  # gate, up: 4096 -> 11008 ; down: 11008 -> 4096
    return L * (attn + mlp)


act = activations_gb(checkpointing=True)
print(f"activations (batch {b}, seq {s}, checkpointing): {act:.2f} GB\n")

# (a) full fine-tuning: every weight is trained
w16 = P * 2 / GB          # bf16 weights used in forward/backward
g16 = P * 2 / GB          # bf16 gradients
master = P * 4 / GB       # fp32 master copy of the weights
adam = P * 8 / GB         # AdamW first + second moment, fp32 each
full = w16 + g16 + master + adam + act + OVERHEAD
print("(a) FULL FINE-TUNING  (16 bytes per parameter + activations)")
print(f"    weights bf16   {P/1e9:.2f}B x 2 B = {w16:6.1f} GB")
print(f"    gradients bf16 {P/1e9:.2f}B x 2 B = {g16:6.1f} GB")
print(f"    fp32 master    {P/1e9:.2f}B x 4 B = {master:6.1f} GB")
print(f"    Adam m and v   {P/1e9:.2f}B x 8 B = {adam:6.1f} GB")
print(f"    activations + overhead        = {act + OVERHEAD:6.1f} GB")
print(f"    TOTAL                         ~ {full:6.1f} GB   -> multi-GPU (e.g. 2x80 GB, or ZeRO/FSDP)\n")

# (b) LoRA: frozen bf16 base, only adapters get gradients + optimizer state
r = 16
lp = lora_params(r)
lora_train = lp * 16 / GB  # adapter in fp32 (4) + grad (4) + Adam m, v (8)
lora = w16 + lora_train + act + OVERHEAD
print(f"(b) LoRA  (r={r} on q,k,v,o,gate,up,down)")
print(f"    adapter params = 32 x [4 x 16 x (4096+4096) + 3 x 16 x (4096+11008)] = {lp/1e6:.1f}M "
      f"({100*lp/P:.2f}% of the model)")
print(f"    frozen weights bf16 {P/1e9:.2f}B x 2 B         = {w16:6.2f} GB")
print(f"    adapter weights+grads+Adam {lp/1e6:.0f}M x 16 B    = {lora_train:6.2f} GB")
print(f"    activations + overhead                    = {act + OVERHEAD:6.2f} GB")
print(f"    TOTAL                                     ~ {lora:6.1f} GB   -> fits a 24 GB GPU\n")

# (c) QLoRA: frozen base stored in 4-bit NF4 (+ quantisation constants), adapters as in LoRA
w4 = P * 0.5 / GB
# NF4 with double quantisation: one constant per 64 weights, itself quantised to 8 bit,
# plus a second-level fp32 constant per 256 constants -> ~0.127 bits per parameter
consts = P * 0.127 / 8 / GB
qlora = w4 + consts + lora_train + act + OVERHEAD
print("(c) QLoRA  (base in 4-bit NF4, same adapters)")
print(f"    frozen weights 4-bit {P/1e9:.2f}B x 0.5 B       = {w4:6.2f} GB")
print(f"    quantisation constants (~0.127 bit/param)  = {consts:6.2f} GB")
print(f"    adapter weights+grads+Adam                 = {lora_train:6.2f} GB")
print(f"    activations + overhead                     = {act + OVERHEAD:6.2f} GB")
print(f"    TOTAL                                      ~ {qlora:6.1f} GB   -> fits an 8 GB card or a free 16 GB T4\n")

print(f"summary: full ~{full:.0f} GB | LoRA ~{lora:.0f} GB | QLoRA ~{qlora:.0f} GB")
