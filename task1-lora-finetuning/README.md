# Task 1 – Fine-tuning with LoRA / QLoRA

Fine-tune **Qwen2.5-0.5B-Instruct** to turn a plain-English voice-assistant request into a fixed JSON format:

```json
{"intent": "alarm_set", "slots": {"time": "five am", "date": "this week"}}
```

## Data

[Amazon MASSIVE](https://github.com/alexa/massive) (CC BY 4.0), en-US: real human-written requests with
intent labels and inline slot annotations. Kept 5 scenarios (alarm, calendar, smart-home/iot, weather,
transport) → 20 intents, 34 slot types. Official splits: **3,991 train / 698 dev / 998 test**.
`prepare_data.py` converts `wake me up at [time : five am] [date : this week]` into the JSON target.

## Method

* Same system prompt for every setting (lists the allowed intents and slot types), greedy decoding.
* Loss only on the answer tokens (prompt tokens masked with -100), plain PyTorch loop in `train_lora.py`.
* **Metric: exact match** = intent correct AND the full set of (slot type, value) pairs correct.
  Also intent accuracy, slot micro-F1 and valid-JSON rate (`common.py`).

## Results on the 998 held-out test requests

| Model | Exact match | Intent acc. | Slot F1 | Valid JSON |
|---|---|---|---|---|
| Base, instructions only | 4.9% | 30.5% | 0.156 | 96.3% |
| Base, + 3 solved examples | 9.7% | 40.6% | 0.128 | 99.7% |
| **LoRA** (bf16 base) | **70.7%** | 95.7% | 0.812 | 100% |
| **QLoRA** (4-bit NF4 base) | **69.2%** | 95.9% | 0.802 | 99.9% |

| Run | Dev loss before → after | Train time | Peak GPU memory |
|---|---|---|---|
| LoRA | 0.485 → 0.043 | 33 min | 3.71 GB |
| QLoRA | 0.730 → 0.044 | 39 min | 3.20 GB |

(RTX 3050 Laptop, 4 GB.) The 4-bit base starts worse, but the adapter recovers nearly all of the gap.

## LoRA settings

| Setting | Value |
|---|---|
| rank r / alpha | 16 / 32 |
| target modules | q, k, v, o, gate, up, down proj |
| dropout | 0.05 |
| learning rate | 2e-4, cosine, 5% warm-up, AdamW, no weight decay |
| epochs | 2 |
| batch | 16 effective (micro-batch 2 × 8 accumulation) |
| trainable params | 8.8M (1.75% of 503M) |

Reasons are in the report.

## Error analysis (`results/predictions_lora.jsonl`)

292 errors: missed slot 47%, wrong intent 15%, wrong slot type 13%, right type but wrong span 13%,
extra slot 12%. Some "errors" are inconsistent gold labels (e.g. "umbrella" is tagged `weather_descriptor`
in 23/23 training requests but not in the test request "should i bring an umbrella today").

## 7B memory estimate

`python memory_estimate.py` prints the full working: full FT ≈ 102 GB, LoRA ≈ 14 GB, QLoRA ≈ 5 GB.

## Run

```bash
# data (40 MB archive; only en-US is needed)
mkdir -p data/raw && cd data/raw
curl -LO https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
tar -xzf amazon-massive-dataset-1.1.tar.gz 1.1/data/en-US.jsonl && cd ../..
python prepare_data.py

python train_lora.py            # LoRA  -> outputs/lora/adapter
python train_lora.py --qlora    # QLoRA -> outputs/qlora/adapter
python evaluate.py base_zeroshot
python evaluate.py base_3shot
python evaluate.py lora
python evaluate.py qlora
python memory_estimate.py
```
