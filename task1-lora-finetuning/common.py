"""Shared pieces for Task 1: prompt building, JSON parsing and the evaluation metric."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def load_labels() -> dict:
    return json.loads((DATA / "labels.json").read_text(encoding="utf-8"))


def system_prompt(labels: dict) -> str:
    return (
        "You convert a smart-assistant request into JSON. Reply with JSON only, in exactly this format:\n"
        '{"intent": "<intent>", "slots": {"<slot_type>": "<value>"}}\n'
        f"<intent> must be one of: {', '.join(labels['intents'])}.\n"
        f"<slot_type> must be one of: {', '.join(labels['slot_types'])}.\n"
        "Slot values must be copied word for word from the request. "
        'Use "slots": {} when the request has no slot.'
    )


def target_text(target: dict) -> str:
    """The exact string the model is trained to produce."""
    return json.dumps(target, ensure_ascii=False)


def build_messages(text: str, labels: dict, shots: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt(labels)}]
    for ex in shots or []:
        msgs.append({"role": "user", "content": ex["text"]})
        msgs.append({"role": "assistant", "content": target_text(ex["target"])})
    msgs.append({"role": "user", "content": text})
    return msgs


# ---------------------------------------------------------------- parsing + metric
def parse_output(raw: str) -> dict | None:
    """Return the predicted dict, or None if the output is not valid JSON in our format."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)  # tolerate markdown code fences
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("intent"), str) or not isinstance(obj.get("slots", {}), dict):
        return None
    obj.setdefault("slots", {})
    return obj


def _norm(v) -> str:
    return re.sub(r"\s+", " ", str(v).lower().strip(" .,!?'\""))


def slot_pairs(slots: dict) -> set[tuple[str, str]]:
    pairs = set()
    for k, v in slots.items():
        for item in v if isinstance(v, list) else [v]:
            pairs.add((k, _norm(item)))
    return pairs


def score(preds: list[dict | None], golds: list[dict]) -> dict:
    """Exact match = intent right AND the full set of (slot type, value) pairs right."""
    n = len(golds)
    valid = intent_ok = exact = tp = fp = fn = 0
    for p, g in zip(preds, golds):
        gp = slot_pairs(g["slots"])
        if p is None:
            fn += len(gp)
            continue
        valid += 1
        pp = slot_pairs(p["slots"])
        i_ok = p["intent"] == g["intent"]
        intent_ok += i_ok
        exact += i_ok and pp == gp
        tp += len(pp & gp)
        fp += len(pp - gp)
        fn += len(gp - pp)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "exact_match": exact / n,
        "intent_acc": intent_ok / n,
        "slot_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "slot_precision": prec,
        "slot_recall": rec,
        "valid_json": valid / n,
    }
