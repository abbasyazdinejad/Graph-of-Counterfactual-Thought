from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from cycb.schema import CyCBInstance, EvidenceItem


@dataclass(frozen=True)
class Perturbation:
    name: str
    description: str
    op: Callable[[List[EvidenceItem]], List[EvidenceItem]]


def _remove_by_type(target: str) -> Callable[[List[EvidenceItem]], List[EvidenceItem]]:
    def inner(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
        return [ev for ev in evidence if ev.etype != target]
    return inner


def _rewrite_type(target: str, new_text: str) -> Callable[[List[EvidenceItem]], List[EvidenceItem]]:
    def inner(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
        out: List[EvidenceItem] = []
        for ev in evidence:
            if ev.etype == target:
                out.append(ev.model_copy(update={"text": new_text}))
            else:
                out.append(ev)
        return out
    return inner


def _mask_encryption(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
    """
    Non-semantic/noise: keep the fact of encrypted comms but remove specifics.
    (Should NOT flip benign->malicious; should mostly preserve label.)
    """
    out: List[EvidenceItem] = []
    for ev in evidence:
        if ev.etype == "encryption":
            out.append(ev.model_copy(update={"text": "(MASKED) Encrypted traffic observed (details unavailable)."}))
        else:
            out.append(ev)
    return out


def _modify_c2(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
    """
    Non-semantic/noise: alter C2 endpoint string while preserving semantics.
    If there is no c2 evidence, this becomes a no-op.
    """
    out: List[EvidenceItem] = []
    for ev in evidence:
        if ev.etype == "c2":
            out.append(
                ev.model_copy(
                    update={"text": "C2 indicators altered: endpoint/IOC changed but behavior remains consistent with remote comms."}
                )
            )
        else:
            out.append(ev)
    return out


def _remove_encryption_and_coercion(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
    """
    Semantic flip for Ransomware -> Backdoor:
      - remove encryption evidence
      - remove coercive text (ransom/payment/decrypt)
      - add explicit backdoor-like evidence (remote command execution)
    """
    out: List[EvidenceItem] = []
    for ev in evidence:
        # remove encryption
        if ev.etype == "encryption":
            continue

        # remove coercion/ransom artifacts if present in text
        t = (ev.text or "").lower()
        if any(k in t for k in ["ransom", "decrypt", "payment", "decryption key", "pay "]):
            continue

        out.append(ev)

    # add a strong backdoor/operator tasking signal
    out.append(
        EvidenceItem(
            eid="ECF1",
            etype="other",
            text="Executed remote commands (cmd /c) received from the network channel; behavior consistent with operator tasking.",
        )
    )
    return out


def _remove_credential_access_flip_to_benign(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
    """
    Semantic flip for Infostealer -> Benign:
      - remove credential_access evidence
      - rewrite exfiltration as benign diagnostic upload (still 'exfiltration' type, but benign semantics)
    """
    out: List[EvidenceItem] = []
    for ev in evidence:
        if ev.etype == "credential_access":
            continue
        if ev.etype == "exfiltration":
            out.append(
                ev.model_copy(
                    update={"text": "Uploaded diagnostic logs to a vendor support endpoint as part of normal telemetry."}
                )
            )
        else:
            out.append(ev)
    return out


def _inject_c2_backdoor(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
    """
    Semantic flip for Benign -> Backdoor:
      - inject explicit C2 + persistence evidence
    """
    out = list(evidence)

    # inject C2 if not already present
    if not any(ev.etype == "c2" for ev in out):
        out.append(
            EvidenceItem(
                eid="ECF_C2",
                etype="c2",
                text="Established periodic beaconing to an unknown remote endpoint and awaited command tasking.",
            )
        )

    # inject persistence if not already present
    if not any(ev.etype == "persistence" for ev in out):
        out.append(
            EvidenceItem(
                eid="ECF_P",
                etype="persistence",
                text=r"Created registry Run key HKCU\Software\Microsoft\Windows\CurrentVersion\Run\svchost-helper.",
            )
        )

    return out


# Registered perturbations (must match CF_RULES keys)
PERTURBATIONS: Dict[str, Perturbation] = {
    # shared primitives
    "remove_persistence": Perturbation(
        name="remove_persistence",
        description="Remove persistence evidence items (Run keys, scheduled tasks, services).",
        op=_remove_by_type("persistence"),
    ),
    "remove_c2": Perturbation(
        name="remove_c2",
        description="Remove command-and-control evidence items.",
        op=_remove_by_type("c2"),
    ),
    "suppress_exfiltration": Perturbation(
        name="suppress_exfiltration",
        description="Remove exfiltration evidence items.",
        op=_remove_by_type("exfiltration"),
    ),
    "remove_credential_access": Perturbation(
        name="remove_credential_access",
        description="Remove credential access evidence and rewrite any exfiltration as benign telemetry (Infostealer->Benign).",
        op=_remove_credential_access_flip_to_benign,
    ),
    "mask_encryption": Perturbation(
        name="mask_encryption",
        description="Mask encryption indicators (non-semantic/noise).",
        op=_mask_encryption,
    ),
    "modify_c2": Perturbation(
        name="modify_c2",
        description="Modify C2 IOCs while preserving semantics (non-semantic/noise).",
        op=_modify_c2,
    ),

    # class-specific semantic flips
    "remove_encryption_and_coercion": Perturbation(
        name="remove_encryption_and_coercion",
        description="Remove ransomware encryption+coercion and inject backdoor-like tasking evidence (Ransomware->Backdoor).",
        op=_remove_encryption_and_coercion,
    ),
    "inject_c2_backdoor": Perturbation(
        name="inject_c2_backdoor",
        description="Inject C2 + persistence signals into otherwise benign evidence (Benign->Backdoor).",
        op=_inject_c2_backdoor,
    ),
}


def apply_perturbation(inst: CyCBInstance, perturbation_name: str) -> CyCBInstance:
    if perturbation_name not in PERTURBATIONS:
        raise KeyError(
            f"Unknown perturbation: {perturbation_name}. "
            f"Available: {sorted(list(PERTURBATIONS.keys()))}"
        )
    p = PERTURBATIONS[perturbation_name]
    new_evidence = p.op(inst.evidence)

    return CyCBInstance(
        sample_id=inst.sample_id,
        category=inst.category,
        evidence=new_evidence,
        label=inst.label,
        counterfactual_labels=inst.counterfactual_labels,
    )
