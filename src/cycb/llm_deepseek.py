from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from cycb.labels import normalize_label


@dataclass
class AgentResult:
    decision: str
    reasoning: str


_LABEL_RE = re.compile(
    r"(?:FINAL\s+LABEL|LABEL)\s*:\s*(.+)$",
    flags=re.IGNORECASE | re.MULTILINE,
)


class DeepSeekChatLLM:
    """
    OpenAIChatLLM-compatible wrapper for DeepSeek.
    Exposes: predict(system_prompt, user_prompt, method)
    """

    def __init__(self, model: str = "deepseek-reasoner", temperature: float = 0.0):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY")

        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def _parse_output(self, output: str) -> AgentResult:
        output = (output or "").strip()

        # Try FINAL LABEL / LABEL format
        m = _LABEL_RE.search(output)
        if m:
            decision_raw = m.group(1).strip().splitlines()[0].strip()
            reasoning = output[: m.start()].strip()
        else:
            # fallback: treat entire output as "decision"
            decision_raw = output.strip()
            reasoning = ""

        decision = normalize_label(decision_raw.replace(".", "").strip())
        return AgentResult(decision=decision, reasoning=reasoning)

    def predict(self, system_prompt: str, user_prompt: str, method: Optional[str] = None) -> AgentResult:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        output = (resp.choices[0].message.content or "").strip()
        return self._parse_output(output)
