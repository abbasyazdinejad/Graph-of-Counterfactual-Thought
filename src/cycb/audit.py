from __future__ import annotations
import re
from typing import List
from .schema import CyCBInstance

EID_PATTERN = re.compile(r"\[E\d+\]")

def evidence_id_coverage(reasoning: str, inst: CyCBInstance) -> float:
    """
    Coverage = fraction of evidence IDs that are explicitly referenced in reasoning.
    If reasoning is empty, coverage is 0.
    """
    if not reasoning.strip():
        return 0.0
    mentioned = set(EID_PATTERN.findall(reasoning))
    if not inst.evidence:
        return 1.0
    total = {f"[{ev.eid}]" for ev in inst.evidence}
    hit = len(total.intersection(mentioned))
    return hit / max(len(total), 1)

def contradiction_penalty(reasoning: str) -> float:
    """
    Very light heuristic:
    Penalize if reasoning claims strong certainty while mentioning missing/unknown evidence.
    """
    r = reasoning.lower()
    if not r.strip():
        return 1.0
    penalty = 0.0
    if "no evidence" in r or "not provided" in r or "unknown" in r:
        penalty += 0.25
    if "certain" in r and ("unknown" in r or "not provided" in r):
        penalty += 0.25
    return min(penalty, 0.5)

def compute_atd(reasoning: str, method: str, num_nodes: int = 1) -> float:
    """
    Audit Trace Depth (ATD):
    - For CVA: depth approx = number of nodes in G-CoT (root + counterfactuals)
    - For CoT/ToT: depth approx = number of non-empty lines in reasoning
    - MultiAgent: depth = number of agents (implied) or lines
    - Direct: depth = 0
    """
    if method == "Direct":
        return 0.0
    if method == "CVA":
        return float(num_nodes)

    # For text-based traces, approximate depth by line count
    lines = [ln for ln in reasoning.splitlines() if ln.strip()]
    return float(len(lines)) if lines else 1.0
