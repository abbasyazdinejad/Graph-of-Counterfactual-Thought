from __future__ import annotations
import json
from pathlib import Path
from typing import List
from .schema import CyCBInstance

def load_jsonl(path: str | Path) -> List[CyCBInstance]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    instances: List[CyCBInstance] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                inst = CyCBInstance(**obj)
                #print("DEBUG EVIDENCE:", inst.evidence_text())
                instances.append(inst)
            except Exception as e:
                raise ValueError(f"Failed parsing JSONL at line {i}: {e}") from e
    return instances

