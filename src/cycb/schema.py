from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

EvidenceType = Literal[
    "encryption",
    "persistence",
    "c2",
    "privilege",
    "credential_access",
    "exfiltration",
    "filesystem",
    "other",
]

class EvidenceItem(BaseModel):
    eid: str = Field(..., description="Evidence ID (unique within sample)")
    etype: EvidenceType = Field(..., description="Evidence semantic type")
    text: str = Field(..., description="Human-readable evidence description")

class CyCBInstance(BaseModel):
    sample_id: str
    category: str
    evidence: List[EvidenceItem]
    label: str = Field(..., description="Ground-truth label under original evidence")

    # maps perturbation_name -> ground-truth label under that counterfactual variant
    counterfactual_labels: Dict[str, str] = Field(default_factory=dict)

    def evidence_text(self) -> str:
        """Canonical textual prompt input (we’ll use this later for LLM calls)."""
        lines = [f"Sample: {self.sample_id} | Category: {self.category}"]
        lines.append("Evidence:")
        for ev in self.evidence:
            lines.append(f"- [{ev.eid}] ({ev.etype}) {ev.text}")
        return "\n".join(lines)
