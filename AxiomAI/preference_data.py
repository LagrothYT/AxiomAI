import json
import os
import uuid
from datetime import datetime, timezone


PREFERENCE_DIR = os.path.join("data", "preferences")
PREFERENCE_PATH = os.path.join(PREFERENCE_DIR, "preferences.jsonl")
FEEDBACK_PATH = os.path.join(PREFERENCE_DIR, "feedback_events.jsonl")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_preference_dir():
    os.makedirs(PREFERENCE_DIR, exist_ok=True)
    return PREFERENCE_DIR


def append_jsonl(path, payload):
    ensure_preference_dir()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def append_preference_pair(prompt, chosen, rejected, chosen_label, rejected_label, metadata=None):
    record = {
        "id": str(uuid.uuid4()),
        "created_at": utc_now(),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_label": chosen_label,
        "rejected_label": rejected_label,
        "metadata": metadata or {},
    }
    append_jsonl(PREFERENCE_PATH, record)
    append_feedback_event(
        event_type="preference_pair",
        prompt=prompt,
        candidates={chosen_label: chosen, rejected_label: rejected},
        decision={"chosen": chosen_label, "rejected": rejected_label},
        metadata=metadata,
        preference_id=record["id"],
    )
    return record


def append_feedback_event(event_type, prompt, candidates=None, decision=None, metadata=None, preference_id=None):
    record = {
        "id": str(uuid.uuid4()),
        "created_at": utc_now(),
        "event_type": event_type,
        "prompt": prompt,
        "candidates": candidates or {},
        "decision": decision or {},
        "metadata": metadata or {},
    }
    if preference_id:
        record["preference_id"] = preference_id
    append_jsonl(FEEDBACK_PATH, record)
    return record


def count_jsonl(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def preference_stats():
    ensure_preference_dir()
    return {
        "preference_dir": PREFERENCE_DIR,
        "preferences_path": PREFERENCE_PATH,
        "feedback_path": FEEDBACK_PATH,
        "pairs": count_jsonl(PREFERENCE_PATH),
        "events": count_jsonl(FEEDBACK_PATH),
    }
