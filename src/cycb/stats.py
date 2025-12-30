from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List
from .schema import CyCBInstance

@dataclass
class CategoryStats:
    instances: int
    avg_evidence_items: float
    avg_counterfactuals: float

def dataset_stats(instances: List[CyCBInstance]) -> Dict[str, CategoryStats]:
    grouped: Dict[str, List[CyCBInstance]] = defaultdict(list)
    for inst in instances:
        grouped[inst.category].append(inst)

    stats: Dict[str, CategoryStats] = {}
    for cat, items in grouped.items():
        n = len(items)
        avg_e = sum(len(x.evidence) for x in items) / n
        avg_cf = sum(len(x.counterfactual_labels) for x in items) / n
        stats[cat] = CategoryStats(instances=n, avg_evidence_items=avg_e, avg_counterfactuals=avg_cf)
    return stats
