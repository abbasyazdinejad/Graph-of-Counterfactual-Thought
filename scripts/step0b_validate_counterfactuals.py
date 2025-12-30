#!/usr/bin/env python3
"""Sanity check for CyCB synthetic JSONL.

This catches the two most common causes of weird CF metrics:
- expected_label contains an out-of-taxonomy string (e.g., "Generic Malware")
- a perturbation is effectively a no-op (counterfactual text == original text)

Run:
  python -m scripts.step0b_validate_counterfactuals --path data/cycb_synth.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {"Ransomware", "Infostealer", "Backdoor", "Benign"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="data/cycb_synth.jsonl")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    n = 0
    bad_label = 0
    noop = 0
    missing = 0

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            ex = json.loads(line)
            root_id = ex.get("id", "<no-id>")
            root_text = (ex.get("text") or "").strip()
            root_label = ex.get("label")
            if root_label not in LABELS:
                bad_label += 1
                print(f"[BAD root label] {root_id}: {root_label}")

            cfs = ex.get("counterfactuals")
            if not isinstance(cfs, list) or len(cfs) != 3:
                missing += 1
                print(f"[BAD counterfactuals count] {root_id}: {0 if cfs is None else len(cfs)}")
                continue

            for cf in cfs:
                pert = cf.get("perturbation")
                exp = cf.get("expected_label")
                cf_text = (cf.get("text") or "").strip()

                if exp not in LABELS:
                    bad_label += 1
                    print(f"[BAD expected_label] {root_id}/{pert}: {exp}")

                if cf_text == root_text:
                    noop += 1
                    print(f"[NO-OP perturbation] {root_id}/{pert}: counterfactual text unchanged")

    print("\n[Sanity Summary]")
    print(f"  Instances: {n}")
    print(f"  Bad labels: {bad_label}")
    print(f"  No-op counterfactuals: {noop}")
    print(f"  Missing/invalid counterfactual lists: {missing}")

    return 0 if (bad_label == 0 and noop == 0 and missing == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
