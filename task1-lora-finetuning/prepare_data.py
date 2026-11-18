"""Build the train / dev / test files for Task 1 from the MASSIVE dataset (en-US).

Task: turn a plain-English smart-assistant request into a fixed JSON format
    {"intent": "<intent label>", "slots": {"<slot type>": "<words from the request>"}}

MASSIVE (Amazon, CC BY 4.0) gives human-written utterances with an intent label and
inline slot annotations, e.g.  "wake me up at [time : five am] [date : this week]".
We keep five scenarios so the task stays narrow: alarm, calendar, iot, weather, transport.

Usage:  python prepare_data.py
Needs:  data/raw/1.1/data/en-US.jsonl  (see README for the download command)
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw" / "1.1" / "data" / "en-US.jsonl"
OUT = HERE / "data"
SCENARIOS = {"alarm", "calendar", "iot", "weather", "transport"}
SLOT_RE = re.compile(r"\[(\w+) : ([^\]]+)\]")


def to_target(annot_utt: str, intent: str) -> dict:
    """Turn MASSIVE's bracket annotation into our JSON target."""
    slots: dict = {}
    for slot_type, value in SLOT_RE.findall(annot_utt):
        value = value.strip()
        if slot_type in slots:  # same slot type twice -> keep both values as a list
            prev = slots[slot_type]
            slots[slot_type] = (prev if isinstance(prev, list) else [prev]) + [value]
        else:
            slots[slot_type] = value
    return {"intent": intent, "slots": slots}


def main() -> None:
    rows = [json.loads(line) for line in RAW.open(encoding="utf-8")]
    rows = [r for r in rows if r["scenario"] in SCENARIOS]

    splits: dict[str, list] = {"train": [], "dev": [], "test": []}
    for r in rows:
        splits[r["partition"]].append(
            {"id": r["id"], "text": r["utt"], "target": to_target(r["annot_utt"], r["intent"])}
        )

    # The label inventory comes from the training split only.
    intents = sorted({ex["target"]["intent"] for ex in splits["train"]})
    slot_types = sorted({k for ex in splits["train"] for k in ex["target"]["slots"]})

    random.seed(0)
    random.shuffle(splits["train"])
    for name, data in splits.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    labels = {"intents": intents, "slot_types": slot_types}
    (OUT / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    print({k: len(v) for k, v in splits.items()})
    print(f"{len(intents)} intents, {len(slot_types)} slot types")
    print("test intent distribution:", Counter(ex["target"]["intent"] for ex in splits["test"]).most_common())


if __name__ == "__main__":
    main()
