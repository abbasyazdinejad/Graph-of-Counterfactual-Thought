from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

OUT_PATH = Path("data/cycb_v1.jsonl")

def inst(
    sample_id: str,
    category: str,
    label: str,
    evidence: List[Dict],
    counterfactual_labels: Dict[str, str],
):
    return {
        "sample_id": sample_id,
        "category": category,
        "label": label,
        "evidence": evidence,  # list of {eid, text}
        "counterfactual_labels": counterfactual_labels,  # pname -> label
    }

#def E(eid: str, text: str):
    # Store with explicit [E#] in the text so coverage checks work later
   # return {"eid": eid, "text": f"[{eid}] {text}"}

#def E(eid: str, text: str, etype: str = "behavior"):
def E(eid: str, text: str, etype: str = "other"):

    # etype is required by your schema
    return {"eid": eid, "etype": etype, "text": f"[{eid}] {text}"}

def main():
    # NOTE: labels must match your agent label space
    # If your schema uses different names, keep them consistent everywhere.
    data = []

    # -------------------------
    # RANSOMWARE 
    # -------------------------
    data.append(inst(
        "RANSOM_001",
        "Ransomware",
        "Ransomware",
        [
            E("E1", "Creates file enumeration loop across user directories and reads many documents."),
            E("E2", "Writes encrypted copies with new extensions; originals become unreadable."),
            E("E3", "Drops ransom note in each directory and changes wallpaper to payment instructions."),
            E("E4", "Disables shadow copies / restore points using system commands."),
            E("E5", "Spawns process to terminate backup services and database services."),
        ],
        {
            # If encryption is masked, ransomware evidence weakens -> could drop to Generic Malware
            "mask_encryption": "Generic Malware",
            # Remove persistence doesn’t change ransomware
            "remove_persistence": "Ransomware",
            # Modify C2 might not matter (many ransomware families don’t need C2)
            "modify_c2": "Ransomware",
        }
    ))

    data.append(inst(
        "RANSOM_002",
        "Ransomware",
        "Ransomware",
        [
            E("E1", "Mass renames files after rewriting contents with high-entropy bytes."),
            E("E2", "Creates mutex to avoid multiple runs and enumerates network shares."),
            E("E3", "Attempts to delete backups and disables recovery features."),
            E("E4", "Ransom note indicates timed payment and decryption key escrow."),
        ],
        {
            "mask_encryption": "Generic Malware",
            "suppress_exfiltration": "Ransomware",
            "modify_c2": "Ransomware",
        }
    ))

    # -------------------------
    # INFOSTEALER 
    # -------------------------
    data.append(inst(
        "INFO_001",
        "Infostealer",
        "Infostealer",
        [
            E("E1", "Reads browser profile databases and extracts saved login records."),
            E("E2", "Accesses OS credential store / keychain APIs."),
            E("E3", "Compresses stolen data into a single archive for staging."),
            E("E4", "Makes outbound POST requests with the archive to a remote endpoint."),
        ],
        {
            # If exfiltration is suppressed, could look like credential harvesting tool -> still infostealer (attempted theft exists)
            "suppress_exfiltration": "Infostealer",
            # If persistence removed, still infostealer (core behavior unchanged)
            "remove_persistence": "Infostealer",
            # If encryption masked (irrelevant), still infostealer
            "mask_encryption": "Infostealer",
        }
    ))

    data.append(inst(
        "INFO_002",
        "Infostealer",
        "Infostealer",
        [
            E("E1", "Enumerates installed browsers and reads cookie / session stores."),
            E("E2", "Collects autofill data and email addresses from local profiles."),
            E("E3", "Targets crypto wallet extensions and copies key material."),
            E("E4", "Performs periodic beacon to fetch exfil destination and sends staged bundle."),
        ],
        {
            "suppress_exfiltration": "Infostealer",
            "modify_c2": "Infostealer",
            "remove_persistence": "Infostealer",
        }
    ))

    # -------------------------
    # BACKDOOR 
    # -------------------------
    data.append(inst(
        "BD_001",
        "Backdoor",
        "Backdoor",
        [
            E("E1", "Maintains periodic beacon to command endpoint and sleeps with jitter."),
            E("E2", "Downloads tasking and executes shell commands received from remote server."),
            E("E3", "Creates persistence via scheduled task to relaunch at startup."),
            E("E4", "Opens reverse shell when specific command is received."),
        ],
        {
            # If C2 modified to benign domain but still beacons, still backdoor
            "modify_c2": "Backdoor",
            # Remove persistence: still backdoor (runtime C2 behavior remains)
            "remove_persistence": "Backdoor",
            # Suppress exfiltration doesn't change remote control nature
            "suppress_exfiltration": "Backdoor",
        }
    ))

    data.append(inst(
        "BD_002",
        "Backdoor",
        "Backdoor",
        [
            E("E1", "Registers as a service and maintains long-lived encrypted channel."),
            E("E2", "Receives remote commands and uploads system inventory results."),
            E("E3", "Implements keep-alive and retries with fallback domains."),
        ],
        {
            "modify_c2": "Backdoor",
            "remove_persistence": "Backdoor",
            "mask_encryption": "Backdoor",
        }
    ))

    # -------------------------
    # BENIGN (2)
    # -------------------------
    data.append(inst(
        "BENIGN_001",
        "Benign Software",
        "Benign",
        [
            E("E1", "Reads configuration files and user preferences for an installed application."),
            E("E2", "Connects over HTTPS to well-known vendor update servers."),
            E("E3", "Downloads signed update package and verifies signature before install."),
            E("E4", "Writes logs to application directory; no unusual process injection."),
        ],
        {
            # Masking encryption irrelevant (benign already)
            "mask_encryption": "Benign",
            # Modify c2 -> for benign, still benign (endpoint change shouldn’t make it malware)
            "modify_c2": "Benign",
            # Remove persistence -> benign apps can have startup entries; removing doesn’t change benign
            "remove_persistence": "Benign",
        }
    ))

    data.append(inst(
        "BENIGN_002",
        "Benign Software",
        "Benign",
        [
            E("E1", "Performs normal file I/O on user-selected documents; no mass rewrite."),
            E("E2", "Uses OS APIs for UI events and stores recent file list."),
            E("E3", "Periodic telemetry to vendor with opt-in setting enabled."),
        ],
        {
            "suppress_exfiltration": "Benign",
            "modify_c2": "Benign",
            "remove_persistence": "Benign",
        }
    ))

    # -------------------------
    # GENERIC MALWARE 
    # -------------------------
    data.append(inst(
        "MAL_001",
        "Generic Malware",
        "Generic Malware",
        [
            E("E1", "Downloads a second-stage payload and executes it in memory."),
            E("E2", "Attempts privilege escalation via known exploit pattern."),
            E("E3", "Enumerates processes and injects into a common system process."),
        ],
        {
            "remove_persistence": "Generic Malware",
            "modify_c2": "Generic Malware",
            "suppress_exfiltration": "Generic Malware",
        }
    ))

    data.append(inst(
        "MAL_002",
        "Generic Malware",
        "Generic Malware",
        [
            E("E1", "Creates autorun registry entry and hides executable in user profile."),
            E("E2", "Connects to unknown domain and fetches encrypted blob."),
            E("E3", "Performs environment checks and anti-analysis delays."),
        ],
        {
            "remove_persistence": "Generic Malware",
            "modify_c2": "Generic Malware",
            "mask_encryption": "Generic Malware",
        }
    ))

    # Write JSONL
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f" Wrote {len(data)} instances -> {OUT_PATH}")

if __name__ == "__main__":
    main()
