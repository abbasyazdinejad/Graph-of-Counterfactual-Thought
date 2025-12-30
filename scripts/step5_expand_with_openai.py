from __future__ import annotations
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from cycb.llm_openai import OpenAIChatLLM
from cycb.dataset_gen import SYSTEM, GenSpec, build_user_prompt, ALLOWED_CATEGORIES

OUT_PATH = Path("data/cycb_candidates.jsonl")

def next_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:03d}"

def prefix_for_category(cat: str) -> str:
    return {
        "Ransomware": "RANSOM",
        "Infostealer": "INFO",
        "Backdoor": "BD",
        "Benign Software": "BENIGN",
        "Generic Malware": "MAL",
    }[cat]

def load_existing_ids(paths: List[Path]) -> set[str]:
    ids = set()
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                ids.add(obj["sample_id"])
    return ids

def main():
    load_dotenv()

    llm = OpenAIChatLLM(temperature=0.2)  # small creativity but stable
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_ids([Path("data/cycb_v1.jsonl"), OUT_PATH])

    # Target: generate 40 candidates (so v1=10 → ~50 after review)
    per_cat = 8  # 5 categories * 8 = 40

    to_generate = []
    for cat in ALLOWED_CATEGORIES:
        prefix = prefix_for_category(cat)
        # find next available index
        i = 1
        generated = 0
        while generated < per_cat:
            sid = next_id(prefix, i)
            i += 1
            if sid in existing:
                continue

            # label mapping: benign category uses "Benign"
            label = "Benign" if cat == "Benign Software" else cat

            to_generate.append(GenSpec(sample_id=sid, category=cat, label=label, n_evidence=5, n_perturbations=3))
            generated += 1

    print(f"Generating {len(to_generate)} candidates -> {OUT_PATH}")

    with OUT_PATH.open("a", encoding="utf-8") as out:
        for spec in to_generate:
            user = build_user_prompt(spec)
            r = llm.predict(SYSTEM, user, method="GEN")

            raw = r.decision  # predict() returns decision as text
            try:
                obj = json.loads(raw)
            except Exception:
                # fallback: try to extract JSON if model wrapped text
                start = raw.find("{")
                end = raw.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    print(f"[SKIP] {spec.sample_id}: could not parse JSON")
                    continue
                obj = json.loads(raw[start:end+1])

            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            print(f"  {obj.get('sample_id')}")

    print("Done.")

if __name__ == "__main__":
    main()
