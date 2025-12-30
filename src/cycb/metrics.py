from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from .schema import CyCBInstance
from .agents import AgentResult, CVAResult
from .audit import evidence_id_coverage, contradiction_penalty, compute_atd

@dataclass
class MethodAverages:
    avg_css: Optional[float]  # only meaningful for CVA (and later for others when we add CF-eval)
    avg_dsi: float
    avg_atd: float
    orig_acc: float
    cf_acc: Optional[float]

def css_from_cva(cva: CVAResult) -> float:
    if not cva.counterfactuals:
        return 1.0
    root_dec = cva.root.decision
    same = sum(1 for _, r in cva.counterfactuals if r.decision == root_dec)
    return same / len(cva.counterfactuals)

def alignment_accuracy(
    inst: CyCBInstance,
    root_decision: str,
    cf_decisions: List[Tuple[str, str]],
) -> Tuple[float, float]:
    acc_orig = 1.0 if root_decision == inst.label else 0.0

    if not inst.counterfactual_labels:
        return acc_orig, 1.0

    correct = 0
    total = 0
    gt = inst.counterfactual_labels
    for pname, dec in cf_decisions:
        if pname in gt:
            total += 1
            if dec == gt[pname]:
                correct += 1
    acc_cf = (correct / total) if total > 0 else 1.0
    return acc_orig, acc_cf

def dsi_for_instance(inst: CyCBInstance, decision: str, reasoning: str, css: Optional[float]) -> float:
    """
    Minimal DSI consistent with your paper:
    DSI = stability component (CSS if available else 0.5)
          * evidence coverage
          * (1 - contradiction_penalty)
    """
    stability = css if css is not None else 0.5
    cov = evidence_id_coverage(reasoning, inst)
    pen = contradiction_penalty(reasoning)
    dsi = stability * max(cov, 0.1) * (1.0 - pen)  # floor cov to avoid zeros early
    return max(0.0, min(1.0, dsi))

def aggregate_metrics(
    instances: List[CyCBInstance],
    results: Dict[str, List[dict]],
) -> Dict[str, MethodAverages]:
    """
    results[method] is a list of per-sample dicts with keys:
      - decision
      - reasoning
      - css (optional)
      - atd
      - orig_acc
      - cf_acc (optional)
    """
    out: Dict[str, MethodAverages] = {}
    for method, rows in results.items():
        n = len(rows)
        avg_css = None
        if any(r.get("css") is not None for r in rows):
            css_vals = [r["css"] for r in rows if r.get("css") is not None]
            avg_css = sum(css_vals) / max(len(css_vals), 1)

        avg_dsi = sum(r["dsi"] for r in rows) / n
        avg_atd = sum(r["atd"] for r in rows) / n
        orig_acc = sum(r["orig_acc"] for r in rows) / n

        cf_acc = None
        if any(r.get("cf_acc") is not None for r in rows):
            cf_vals = [r["cf_acc"] for r in rows if r.get("cf_acc") is not None]
            cf_acc = sum(cf_vals) / max(len(cf_vals), 1)

        out[method] = MethodAverages(
            avg_css=avg_css,
            avg_dsi=avg_dsi,
            avg_atd=avg_atd,
            orig_acc=orig_acc,
            cf_acc=cf_acc,
        )
    return out
