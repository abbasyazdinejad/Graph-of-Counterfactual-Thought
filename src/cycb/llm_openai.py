from __future__ import annotations

import os
import re
from dataclasses import dataclass
from openai import OpenAI

from .agents import AgentResult

VALID_LABELS = ["Ransomware", "Infostealer", "Backdoor", "Benign", "Generic Malware"]

# 1) FINAL LABEL: <label>
FINAL_LABEL_RE = re.compile(
    r"FINAL LABEL:\s*(Ransomware|Infostealer|Backdoor|Benign|Generic\s+Malware)\b",
    re.IGNORECASE,
)

# 2) Bare label anywhere (fallback)
BARE_LABEL_RE = re.compile(
    r"\b(Ransomware|Infostealer|Backdoor|Benign|Generic\s+Malware)\b",
    re.IGNORECASE,
)

def _canonicalize(label: str) -> str:
    # normalize spacing + case
    lab = " ".join(label.strip().split())
    lab_low = lab.lower()
    mapping = {
        "ransomware": "Ransomware",
        "infostealer": "Infostealer",
        "backdoor": "Backdoor",
        "benign": "Benign",
        "generic malware": "Generic Malware",
    }
    return mapping.get(lab_low, lab)


@dataclass
class OpenAIChatLLM:
    model: str = "gpt-4o"
    temperature: float = 0.0

    def __post_init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in environment. Put it in .env and load via dotenv."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = os.environ.get("OPENAI_MODEL", self.model)

    def predict(
        self,
        system_prompt: str,
        user_prompt: str,
        method: str | None = None,
    ) -> AgentResult:
        if method is None:
            method = "direct"

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = (resp.choices[0].message.content or "").strip()

        # First try strict FINAL LABEL
        m = FINAL_LABEL_RE.search(text)
        if m:
            decision = _canonicalize(m.group(1))
            reasoning = text[: m.start()].strip()
            return AgentResult(decision=decision, reasoning=reasoning, method=method)

        # Fallback: if the whole output is just a label (common for Direct)
        compact = re.sub(r"[*_`]", "", text).strip()
        compact = compact.replace(".", "").strip()
        if compact.lower() in {x.lower() for x in VALID_LABELS}:
            decision = _canonicalize(compact)
            return AgentResult(decision=decision, reasoning="", method=method)

        # Last resort: find any label mention (take last occurrence)
        all_matches = list(BARE_LABEL_RE.finditer(text))
        if all_matches:
            last = all_matches[-1]
            decision = _canonicalize(last.group(1))
            reasoning = text[: last.start()].strip()
            return AgentResult(decision=decision, reasoning=reasoning, method=method)

        raise ValueError(
            "Could not parse a valid label from model output.\n"
            f"Model output:\n{text}"
        )
