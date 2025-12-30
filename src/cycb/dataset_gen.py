from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

ALLOWED_CATEGORIES = ["Ransomware", "Infostealer", "Backdoor", "Benign Software", "Generic Malware"]
ALLOWED_LABELS = ["Ransomware", "Infostealer", "Backdoor", "Benign", "Generic Malware"]
ALLOWED_ETYPES = [
    "encryption", "persistence", "c2", "privilege",
    "credential_access", "exfiltration", "filesystem", "other"
]
ALLOWED_PERTURBATIONS = ["remove_persistence", "mask_encryption", "modify_c2", "suppress_exfiltration"]

SYSTEM = """You are generating benchmark instances for a cybersecurity decision benchmark (CyCB).
You must output STRICT JSON only (no markdown, no commentary).
The instance must be internally consistent, realistic, and labelable from evidence."""

@dataclass
class GenSpec:
    sample_id: str
    category: str
    label: str
    n_evidence: int = 5
    n_perturbations: int = 3

def build_user_prompt(spec: GenSpec) -> str:
    return f"""
Generate ONE CyCB instance as STRICT JSON with the schema:

{{
  "sample_id": "{spec.sample_id}",
  "category": "{spec.category}",
  "label": "{spec.label}",
  "evidence": [
    {{"eid":"E1","etype":"<one of {ALLOWED_ETYPES}>","text":"[E1] ..."}},
    ...
  ],
  "counterfactual_labels": {{
    "remove_persistence": "<one of {ALLOWED_LABELS}>",
    "mask_encryption": "<one of {ALLOWED_LABELS}>",
    "modify_c2": "<one of {ALLOWED_LABELS}>",
    "suppress_exfiltration": "<one of {ALLOWED_LABELS}>"
  }}
}}

Constraints:
- category MUST be one of: {ALLOWED_CATEGORIES}
- label MUST be one of: {ALLOWED_LABELS}
- evidence length: exactly {spec.n_evidence}
- Use diverse etype values across evidence (not all "other").
- Evidence must be plausible and discriminative for the label.
- counterfactual_labels must include ALL 4 perturbations (even if unchanged).
- Under "mask_encryption": if label is Ransomware, consider whether removing encryption evidence could shift to Generic Malware.
- Under "suppress_exfiltration": if label is Infostealer, decide if it still counts as Infostealer or becomes Generic Malware (be consistent).
- Keep texts short (1 sentence each).

Return STRICT JSON only.
""".strip()
