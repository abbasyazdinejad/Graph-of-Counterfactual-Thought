from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DecisionSpace = Literal[
    "Ransomware",
    "Infostealer",
    "Backdoor",
    "Benign",
    "Generic Malware",
]

@dataclass(frozen=True)
class PromptPack:
    system: str
    direct: str
    cot: str
    tot: str



LABEL_DEFINITIONS = (
    "Label definitions:\n"
    "- Ransomware: Primary behavior is file encryption with ransom or coercive impact.\n"
    "- Infostealer: Primary behavior is credential or sensitive data collection and exfiltration.\n"
    "- Backdoor: Primary behavior is covert persistent access without direct monetization.\n"
    "- Benign: Legitimate software behavior without malicious intent.\n"
    "- Generic Malware: Malicious behavior that does not clearly fit the above categories.\n"
)

LABEL_LIST_TEXT = (
    "Ransomware\n"
    "Infostealer\n"
    "Backdoor\n"
    "Benign\n"
    "Generic Malware"
)

DEFAULT_SYSTEM = (
    "You are a cybersecurity malware analysis expert. "
    "You are always given complete structured behavioral evidence. "
    "You must NEVER ask for more evidence. "
    "Your task is to classify the sample into exactly ONE label from this set: "
    "Ransomware, Infostealer, Backdoor, Benign, Generic Malware. "
    "Output must follow the user instruction exactly."
)

DIRECT_TEMPLATE = (
    "Task: Classify the sample using the evidence below.\n\n"
    + LABEL_DEFINITIONS +
    "\nYou MUST output exactly ONE label from:\n"
    + LABEL_LIST_TEXT +
    "\nDo NOT output anything else.\n\n"
    "{evidence_text}\n\n"
    "FINAL LABEL:"
)

COT_TEMPLATE = (
    "Task: Classify the sample using the evidence below.\n\n"
    + LABEL_DEFINITIONS +
    "\nProvide brief reasoning grounded ONLY in the evidence.\n"
    "Then output the final decision EXACTLY as:\n"
    "FINAL LABEL: <label>\n\n"
    "{evidence_text}\n"
)

TOT_TEMPLATE = (
    "Task: Classify the sample using the evidence below.\n\n"
    + LABEL_DEFINITIONS +
    "\nGenerate 3 candidate labels with 1-2 evidence-grounded bullets each.\n"
    "Then select the best one and output EXACTLY as:\n"
    "FINAL LABEL: <label>\n\n"
    "{evidence_text}\n"
)

DEFAULT_PROMPTS = PromptPack(
    system=DEFAULT_SYSTEM,
    direct=DIRECT_TEMPLATE,
    cot=COT_TEMPLATE,
    tot=TOT_TEMPLATE,
)
