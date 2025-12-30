from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .schema import CyCBInstance
from .perturbations import apply_perturbation
from .prompts import DEFAULT_PROMPTS

LABELS = ["Ransomware", "Infostealer", "Backdoor", "Benign", "Generic Malware"]

@dataclass
class AgentResult:
    decision: str
    reasoning: str
    method: str

@dataclass
class CVAResult:
    root: AgentResult
    counterfactuals: List[Tuple[str, AgentResult]]  # (perturb_name, result) it is ok 
    g_cot: Dict  # json-serializable structured artifact # done

class MockLLM:
    """
    Deterministic 'LLM-like' classifier for plumbing tests.
    Uses simple evidence keywords to output labels + short rationale.
    """

    def predict(self, evidence_text: str) -> AgentResult:
        t = evidence_text.lower()

        # very simple rules (good enough to test the pipeline)
        if "cryptencrypt" in t or "file encryption" in t or "encrypt" in t and "many" in t:
            label = "Ransomware"
            why = "Encryption behavior across multiple files strongly indicates ransomware."
        elif "credential" in t or "login data" in t or "browser credential" in t:
            label = "Infostealer"
            why = "Credential store access + collection suggests infostealer behavior."
        elif "command-and-control" in t or "beacon" in t or "c2" in t:
            label = "Backdoor"
            why = "C2-like signaling indicates backdoor/remote control behavior."
        elif "well-known cloud" in t or "normal https" in t or "configuration files" in t:
            label = "Benign"
            why = "Benign-looking network + config activity; no clear malicious indicators."
        else:
            label = "Generic Malware"
            why = "Some suspicious activity but insufficient evidence for a specific family."

        #return AgentResult(decision=label, reasoning=why, method="MockLLM") # done
        # include a simple "Evidence used" line so audit can score coverage # done
        reasoning = f"{why}\nEvidence used: [E1]"
        return AgentResult(decision=label, reasoning=reasoning, method="MockLLM")


def run_direct(inst: CyCBInstance, llm: MockLLM) -> AgentResult:
    # "Direct Agent" baseline: decision only (reasoning minimal)
    res = llm.predict(inst.evidence_text())
    return AgentResult(decision=res.decision, reasoning="", method="Direct")

def run_cot(inst: CyCBInstance, llm: MockLLM) -> AgentResult:
    res = llm.predict(inst.evidence_text())
    reasoning = f"Reasoning: {res.reasoning}\nLABEL: {res.decision}"
    return AgentResult(decision=res.decision, reasoning=reasoning, method="CoT")

def run_tot(inst: CyCBInstance, llm: MockLLM) -> AgentResult:
    # Simulate ToT by proposing candidates then choosing one
    base = llm.predict(inst.evidence_text())
    candidates = [
        (base.decision, base.reasoning),
        ("Generic Malware", "Fallback if evidence is incomplete or ambiguous."),
        ("Benign", "If indicators match normal software behavior."),
    ]
    reasoning_lines = ["Candidates:"]
    for lab, why in candidates:
        reasoning_lines.append(f"- {lab}: {why}")
    reasoning_lines.append(f"LABEL: {base.decision}")
    return AgentResult(decision=base.decision, reasoning="\n".join(reasoning_lines), method="ToT")

def run_multi_agent(inst: CyCBInstance, llm: MockLLM, n_agents: int = 3) -> AgentResult:
    # Simulate multi-agent by repeated same model + voting
    votes: List[str] = []
    reasons: List[str] = []
    for _ in range(n_agents):
        r = llm.predict(inst.evidence_text())
        votes.append(r.decision)
        reasons.append(r.reasoning)

    # majority vote
    decision = max(set(votes), key=votes.count)
    reasoning = "Votes: " + ", ".join(votes) + "\n" + " | ".join(reasons)
    return AgentResult(decision=decision, reasoning=reasoning, method="MultiAgent")

def run_cva(inst: CyCBInstance, llm: MockLLM, perturbations: Optional[List[str]] = None) -> CVAResult:
    if perturbations is None:
        perturbations = list(inst.counterfactual_labels.keys())

    root_base = llm.predict(inst.evidence_text())
    root = AgentResult(
        decision=root_base.decision,
        reasoning=root_base.reasoning,
        method="CVA",
    )

    cfs: List[Tuple[str, AgentResult]] = []
    for pname in perturbations:
        inst_cf = apply_perturbation(inst, pname)
        r_cf = llm.predict(inst_cf.evidence_text())
        cfs.append((pname, AgentResult(decision=r_cf.decision, reasoning=r_cf.reasoning, method="CVA")))

    g_cot = {
        "root": {"sample_id": inst.sample_id, "decision": root.decision},
        "counterfactuals": [
            {"perturbation": pname, "decision": r.decision} for pname, r in cfs
        ],
    }
    return CVAResult(root=root, counterfactuals=cfs, g_cot=g_cot)

def run_llm_with_prompt(inst: CyCBInstance, llm, mode: str) -> AgentResult:
    from .prompts import DEFAULT_PROMPTS
    p = DEFAULT_PROMPTS
    if mode == "Direct":
        user = p.direct.format(evidence_text=inst.evidence_text())
        return llm.predict(p.system, user, method="Direct")
    if mode == "CoT":
        user = p.cot.format(evidence_text=inst.evidence_text())
        return llm.predict(p.system, user, method="CoT")
    if mode == "ToT":
        user = p.tot.format(evidence_text=inst.evidence_text())
        return llm.predict(p.system, user, method="ToT")
    raise ValueError(f"Unknown mode: {mode}")


