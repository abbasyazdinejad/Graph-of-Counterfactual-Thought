from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

from cycb.io import load_jsonl

CAND = Path("data/cycb_candidates.jsonl")

def main():
    if not CAND.exists():
        raise SystemExit("No candidates file found. Run step5_expand_with_openai.py first.")

    instances = load_jsonl(str(CAND))  # uses your Pydantic schema
    print(f" Validated {len(instances)} candidates")

    cats = Counter([x.category for x in instances])
    print("Category counts:", dict(cats))

    # quick checks for required perturbations + evidence
    bad = 0
    for x in instances:
        if len(x.evidence) != 5:
            bad += 1
        for k in ["remove_persistence", "mask_encryption", "modify_c2", "suppress_exfiltration"]:
            if k not in x.counterfactual_labels:
                bad += 1
                break
    print(f"Basic schema/content checks flagged: {bad} issues (0 is ideal)")

if __name__ == "__main__":
    main()
