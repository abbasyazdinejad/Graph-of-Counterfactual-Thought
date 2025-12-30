from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

V1 = Path("data/cycb_v1.jsonl")
CAND = Path("data/cycb_candidates.jsonl")
OUT = Path("data/cycb_v50.jsonl")

def read_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    if not V1.exists():
        raise SystemExit("Missing data/cycb_v1.jsonl")
    if not CAND.exists():
        raise SystemExit("Missing data/cycb_candidates.jsonl")

    v1 = read_jsonl(V1)
    cand = read_jsonl(CAND)

    # Deduplicate by sample_id (keep v1 if collision)
    seen = {x["sample_id"] for x in v1}
    merged = list(v1)
    skipped = 0
    for x in cand:
        sid = x.get("sample_id")
        if sid in seen:
            skipped += 1
            continue
        merged.append(x)
        seen.add(sid)

    # Quick summary
    cats = Counter([x["category"] for x in merged])
    labels = Counter([x["label"] for x in merged])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for x in merged:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f" v1: {len(v1)}")
    print(f" candidates: {len(cand)}")
    print(f" merged: {len(merged)}  (skipped duplicates: {skipped})")
    print("Category counts:", dict(cats))
    print("Label counts:", dict(labels))
    print(f" wrote -> {OUT}")

if __name__ == "__main__":
    main()
