from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic


@dataclass
class AgentResult:
    decision: str
    reasoning: str


def _extract_label(output: str) -> tuple[str, str]:
    """
    Supports both:
      - "FINAL LABEL: <label>"
      - "LABEL: <label>"
    Returns (decision, reasoning).
    """
    out = (output or "").strip()

    for tag in ("FINAL LABEL:", "LABEL:", "Final Label:", "Final label:", "label:"):
        if tag in out:
            before, after = out.split(tag, 1)
            reasoning = before.strip()
            decision = after.strip().splitlines()[0].strip()
            return decision.rstrip(".").strip(), reasoning

    # If model outputs only a label, keep first line
    first = out.splitlines()[0].strip() if out else ""
    return first.rstrip(".").strip(), ""


class ClaudeChatLLM:
    """
    Drop-in for your OpenAI step4 code:
      predict(system_prompt=..., user_prompt=..., method=...)
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0, max_tokens: int = 1024):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
        self.temperature = float(os.getenv("CLAUDE_TEMPERATURE", str(temperature)))
        self.max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", str(max_tokens)))

    def predict(self, system_prompt: str, user_prompt: str, method: str = "") -> AgentResult:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # msg.content is a list of blocks; take all text blocks safely
        chunks = []
        for b in (msg.content or []):
            t = getattr(b, "text", None)
            if t:
                chunks.append(t)
        output = "\n".join(chunks).strip()

        decision, reasoning = _extract_label(output)
        return AgentResult(decision=decision, reasoning=reasoning)
