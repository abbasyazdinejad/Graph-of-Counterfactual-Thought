#!/usr/bin/env python3
"""CyCB synthetic generator (v2.3) — counterfactuals that actually change the evidence.

Drop-in intent:
- Generate a JSONL file at data/cycb_synth.jsonl.
- Each instance has exactly 3 counterfactuals.

Why this version fixes your current outputs:
1) No out-of-taxonomy expected counterfactual labels.
   Your run showed expected_cf="Generic Malware" in remove_c2, but your decision space
   is effectively 4 classes. This makes CF_Acc artificially 0.0 for that perturbation.
2) Perturbations are guaranteed to *remove/alter the right indicators* in the text.
   Your run showed remove_credential_access / remove_encryption_and_coercion never flipping,
   which is usually a sign those perturbations didn't actually remove the core signals.

Decision space (4-way): Ransomware, Infostealer, Backdoor, Benign.

Schema (per line):
{
  "id": "RANSOMWARE_001",
  "label": "Ransomware",
  "evidence": {...},
  "text": "...",
  "counterfactuals": [
    {"cf_id": "...", "perturbation": "remove_encryption_and_coercion", "expected_label": "Benign", "evidence": {...}, "text": "..."},
    ...
  ]
}

Run:
  python -m scripts.step0_generate_synth_cycb --n-per-class 25 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

LABELS = ["Ransomware", "Infostealer", "Backdoor", "Benign"]


@dataclass
class Profile:
    core: List[str]
    optional: List[str]


PROFILES: Dict[str, Profile] = {
    # Core signals are the *only* things you need to decide the label.
    "Ransomware": Profile(
        core=["encrypt_files", "ransom_note"],
        optional=["file_extension_change", "shadow_copy_delete"],
    ),
    "Infostealer": Profile(
        core=["credential_dumping", "browser_credential_theft", "data_exfiltration"],
        optional=["clipboard_monitoring"],
    ),
    "Backdoor": Profile(
        core=["c2_beaconing", "remote_shell", "persistence_run_key"],
        optional=["dns_tunneling"],
    ),
    "Benign": Profile(
        core=["legitimate_admin_activity", "normal_file_access"],
        optional=["software_update", "scheduled_backup"],
    ),
}


def render_text(evidence: Dict[str, bool]) -> str:
    """Render evidence as an LLM-friendly bullet list without label leakage."""
    bullets: List[str] = []

    # Ransomware-like
    if evidence.get("encrypt_files"):
        bullets.append("Rapid encryption of user documents observed (files become unreadable).")
    if evidence.get("ransom_note"):
        bullets.append("A note appears demanding payment for a decryption key.")
    if evidence.get("file_extension_change"):
        bullets.append("Files are renamed with a new, unusual extension.")
    if evidence.get("shadow_copy_delete"):
        bullets.append("System restore/shadow copies are deleted to hinder recovery.")

    # Infostealer-like
    if evidence.get("credential_dumping"):
        bullets.append("Process accesses LSASS / credential stores to extract secrets.")
    if evidence.get("browser_credential_theft"):
        bullets.append("Browser password stores/cookies are accessed and copied.")
    if evidence.get("clipboard_monitoring"):
        bullets.append("Clipboard contents are polled frequently.")
    if evidence.get("data_exfiltration"):
        bullets.append("Outbound transfer of credential-like data to an external endpoint.")

    # Backdoor-like
    if evidence.get("c2_beaconing"):
        bullets.append("Periodic network beacons to a remote server (C2-style keepalive).")
    if evidence.get("dns_tunneling"):
        bullets.append("Suspicious DNS queries consistent with tunneling.")
    if evidence.get("remote_shell"):
        bullets.append("A remote interactive shell session is opened from the host.")
    if evidence.get("persistence_run_key"):
        bullets.append("Registry Run key / startup persistence is established.")

    # Benign-like
    if evidence.get("legitimate_admin_activity"):
        bullets.append("Admin tools are used in a typical maintenance workflow.")
    if evidence.get("normal_file_access"):
        bullets.append("Routine document access and editing by a user process.")
    if evidence.get("software_update"):
        bullets.append("A signed updater downloads and installs patches.")
    if evidence.get("scheduled_backup"):
        bullets.append("Backup agent archives files to a known backup destination.")

    # Always include a short context header
    header = (
        "You are given endpoint telemetry summarized as discrete evidence items. "
        "Classify the activity into exactly one category.\n\n"
        "Evidence:\n"
    )
    return header + "\n".join(f"- {b}" for b in bullets)


def make_base_evidence(label: str, rng: random.Random) -> Dict[str, bool]:
    prof = PROFILES[label]
    ev = {k: True for k in prof.core}

    # optional signals: keep small to avoid ambiguous multi-label behavior
    for k in prof.optional:
        ev[k] = rng.random() < 0.35

    # Ensure no cross-class core leakage
    for other_label, other_prof in PROFILES.items():
        if other_label == label:
            continue
        for k in other_prof.core:
            ev.setdefault(k, False)

    # Ensure a boolean for every known key
    all_keys = {k for p in PROFILES.values() for k in (p.core + p.optional)}
    for k in all_keys:
        ev.setdefault(k, False)

    return ev


def apply_perturbation(label: str, ev: Dict[str, bool], perturbation: str) -> Tuple[Dict[str, bool], str]:
    """Return (new_evidence, expected_label)."""
    new = dict(ev)

    if perturbation == "modify_c2":
        # Modify but do not remove C2 evidence.
        if new.get("c2_beaconing"):
            new["dns_tunneling"] = True  # treat as a "changed" C2 modality
        return new, label

    if perturbation == "remove_persistence":
        new["persistence_run_key"] = False
        # Backdoor without persistence is still a backdoor if C2+remote shell remain.
        return new, label

    if perturbation == "remove_c2":
        # Remove C2 + remote shell together to make it plausibly benign.
        new["c2_beaconing"] = False
        new["dns_tunneling"] = False
        new["remote_shell"] = False
        # If anything malicious remains, map to Backdoor; otherwise benign.
        if new.get("persistence_run_key"):
            # keep benign expectation by also removing persistence in this perturbation
            new["persistence_run_key"] = False
        return new, "Benign"

    if perturbation == "remove_credential_access":
        new["credential_dumping"] = False
        new["browser_credential_theft"] = False
        new["clipboard_monitoring"] = False
        # Also remove exfil to avoid leaving a "still malicious" artifact.
        new["data_exfiltration"] = False
        return new, "Benign"

    if perturbation == "remove_encryption_and_coercion":
        new["encrypt_files"] = False
        new["ransom_note"] = False
        new["file_extension_change"] = False
        new["shadow_copy_delete"] = False
        return new, "Benign"

    if perturbation == "inject_c2_backdoor":
        # Turn a benign sample into a backdoor-like one.
        new["c2_beaconing"] = True
        new["remote_shell"] = True
        new["persistence_run_key"] = True
        return new, "Backdoor"

    # Fallback: no-op
    return new, label


def make_counterfactuals(label: str, ev: Dict[str, bool], rng: random.Random) -> List[Dict]:
    # Always 3 CFs, but pick perturbations that make sense for the label.
    if label == "Ransomware":
        perturbs = ["remove_encryption_and_coercion", "modify_c2", "remove_persistence"]
        # Ransomware typically doesn't have C2 in our profiles; modify_c2 becomes a harmless no-op.
    elif label == "Infostealer":
        perturbs = ["remove_credential_access", "modify_c2", "remove_persistence"]
    elif label == "Backdoor":
        perturbs = ["remove_c2", "modify_c2", "remove_persistence"]
    else:  # Benign
        perturbs = ["inject_c2_backdoor", "modify_c2", "remove_persistence"]

    cfs: List[Dict] = []
    for j, p in enumerate(perturbs, start=1):
        new_ev, expected = apply_perturbation(label, ev, p)
        cfs.append(
            {
                "cf_id": f"{p}_{j}",
                "perturbation": p,
                "expected_label": expected,
                "evidence": new_ev,
                "text": render_text(new_ev),
            }
        )
    return cfs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/cycb_synth.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    counts: Dict[str, int] = {k: 0 for k in LABELS}

    for label in LABELS:
        for i in range(1, args.n_per_class + 1):
            counts[label] += 1
            inst_id = f"{label.upper()}_{i:03d}"
            ev = make_base_evidence(label, rng)
            row = {
                "id": inst_id,
                "label": label,
                "evidence": ev,
                "text": render_text(ev),
                "counterfactuals": make_counterfactuals(label, ev, rng),
                "version": "2.3",
            }
            rows.append(row)

    # Write jsonl
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Generated {len(rows)} CyCB synthetic instances (v2.3)")
    print(f"[OK] Saved to {out_path.resolve()}")
    print("[Summary]")
    for k in LABELS:
        print(f"  {k}: {counts[k]}")
    # Sanity: exactly 3 CFs
    uniq_cf_counts = sorted({len(r["counterfactuals"]) for r in rows})
    print(f"[Sanity] Unique CF counts per instance: {uniq_cf_counts} (expected: [3])")


if __name__ == "__main__":
    main()
