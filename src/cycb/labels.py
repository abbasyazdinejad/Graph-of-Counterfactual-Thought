"""Label parsing / normalization helpers.

Use this to make scoring robust to near-synonyms and legacy labels.
# done
"""

from __future__ import annotations

import re
from typing import Optional

LABELS = ("Ransomware", "Infostealer", "Backdoor", "Benign")

# Keep patterns ordered from most specific to least.
_PATTERNS = [
    (re.compile(r"\bransom\w*\b", re.I), "Ransomware"),
    (re.compile(r"\binfost\w*|\bsteal\w*\b.*\bcred\w*|\bcredential\w*\b", re.I), "Infostealer"),
    (re.compile(r"\bbackdoor\b|\bc2\b|command\s*and\s*control|remote\s*access|reverse\s*shell", re.I), "Backdoor"),
    (re.compile(r"\bbenign\b|legit\w*|normal\s*activity", re.I), "Benign"),
    (re.compile(r"generic\s+malware|\bmalware\b|trojan", re.I), "Backdoor"),  # legacy
]


def normalize_label(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Quick exact match
    for l in LABELS:
        if s.lower() == l.lower():
            return l

    for pat, lab in _PATTERNS:
        if pat.search(s):
            return lab

    return None
