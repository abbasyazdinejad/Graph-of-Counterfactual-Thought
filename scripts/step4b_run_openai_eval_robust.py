from __future__ import annotations

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from cycb.io import load_jsonl
from cycb.perturbations import apply_perturbation
from cycb.prompts import DEFAULT_PROMPTS  # PromptPack dataclass (attribute access)



MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))


def parse_label_and_reasoning(text: str) -> Tuple[str, str]:
    """
    Robustly parse decision label. Supports formats like:
      - "LABEL: Benign"
      - "Reasoning...\nLABEL: Ransomware"
      - just "Benign"
    Returns: (decision, reasoning)
    """
    text = (text or "").strip()
    if not text:
        return "", ""

    if "LABEL:" in text:
        before, after = text.split("LABEL:", 1)
        reasoning = before.strip()
        decision = after.strip().splitlines()[0].strip()
    else:
        reasoning = ""
        decision = text.splitlines()[0].strip()

    decision = decision.replace(".", "").strip()
    return decision, reasoning


def css_consistency(root_decision: str, cf_decisions: List[str]) -> float:
    """
    Simple, stable CSS: fraction of counterfactual decisions matching root decision.
    If no counterfactuals, CSS = 1.0.
    """
    if not cf_decisions:
        return 1.0
    if not root_decision:
        return 0.0
    m = sum(1 for d in cf_decisions if (d or "").strip() == root_decision)
    return m / float(len(cf_decisions))


def safe_get_instance_id(inst: Any) -> str:
    for k in ["sample_id", "instance_uid", "id", "uid"]:
        if hasattr(inst, k):
            v = getattr(inst, k)
            if v is not None:
                return str(v)
    return "UNKNOWN"


def safe_get_category(inst: Any) -> str:
    for k in ["category", "type", "family"]:
        if hasattr(inst, k):
            v = getattr(inst, k)
            if v is not None:
                return str(v)
    return ""


def safe_get_label(inst: Any) -> str:
    return str(getattr(inst, "label", "") or "")


def chat_completion(client: OpenAI, system: str, user: str) -> Dict[str, str]:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    decision, reasoning = parse_label_and_reasoning(text)
    return {"decision": decision, "reasoning": reasoning, "raw": text}


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env and run again.")

    client = OpenAI(api_key=api_key)

    repo_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text())

    dataset_path = repo_root / cfg["dataset_path"]
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"openai_eval_log_{MODEL_NAME}_{stamp}.jsonl"
    csv_path = out_dir / f"table_metrics_openai_{MODEL_NAME}_{stamp}.csv"

    instances = load_jsonl(str(dataset_path))

    # perturbations can come from config; if missing, use keys from instance.counterfactual_labels
    cfg_perturbations = cfg.get("perturbations", None)

    # CSV rows: one row per (instance, method, perturbation)
    csv_rows: List[Dict[str, Any]] = []

    system_prompt = DEFAULT_PROMPTS.system  # PromptPack: attribute, not dict

    with open(log_path, "w", encoding="utf-8") as f_log:
        for inst in instances:
            sid = safe_get_instance_id(inst)
            label = safe_get_label(inst)
            cat = safe_get_category(inst)

            print(f"== {sid} ({cat}) ==")

            evidence = inst.evidence_text() if hasattr(inst, "evidence_text") else ""

            # ---------- DIRECT / COT / TOT ----------
            per_method_outputs: Dict[str, Dict[str, str]] = {}

            user_direct = DEFAULT_PROMPTS.direct.format(evidence_text=evidence)
            user_cot = DEFAULT_PROMPTS.cot.format(evidence_text=evidence)
            user_tot = DEFAULT_PROMPTS.tot.format(evidence_text=evidence)

            per_method_outputs["direct"] = chat_completion(client, system_prompt, user_direct)
            per_method_outputs["cot"] = chat_completion(client, system_prompt, user_cot)
            per_method_outputs["tot"] = chat_completion(client, system_prompt, user_tot)

            # ---------- CVA (robust): root + perturbations ----------
            # Decide perturbation list
            inst_cf_labels = getattr(inst, "counterfactual_labels", {}) or {}
            if cfg_perturbations is None:
                perturb_list = list(inst_cf_labels.keys())
            else:
                perturb_list = list(cfg_perturbations)

            # root
            root_out = chat_completion(client, system_prompt, user_cot)
            root_dec = root_out["decision"]

            cva_rows: List[Dict[str, Any]] = []
            cva_rows.append(
                {
                    "perturbation": "orig",
                    "decision": root_out["decision"],
                    "reasoning": root_out["reasoning"],
                    "raw": root_out["raw"],
                }
            )

            # counterfactuals
            cf_decisions: List[str] = []
            for p_name in perturb_list:
                try:
                    inst_p = apply_perturbation(inst, p_name)
                    ev_p = inst_p.evidence_text() if hasattr(inst_p, "evidence_text") else evidence
                    user_p = DEFAULT_PROMPTS.cot.format(evidence_text=ev_p)

                    out_p = chat_completion(client, system_prompt, user_p)
                    cf_decisions.append(out_p["decision"])

                    cva_rows.append(
                        {
                            "perturbation": p_name,
                            "decision": out_p["decision"],
                            "reasoning": out_p["reasoning"],
                            "raw": out_p["raw"],
                        }
                    )
                except Exception as e:
                    # keep going; log failure as a row
                    cva_rows.append(
                        {
                            "perturbation": p_name,
                            "decision": "",
                            "reasoning": "",
                            "raw": f"[ERROR] {type(e).__name__}: {e}",
                        }
                    )

            css = css_consistency(root_dec, cf_decisions)

            # accuracies
            def acc(decision: str, gold: str) -> int:
                return int((decision or "").strip() == (gold or "").strip())

            direct_acc = acc(per_method_outputs["direct"]["decision"], label)
            cot_acc = acc(per_method_outputs["cot"]["decision"], label)
            tot_acc = acc(per_method_outputs["tot"]["decision"], label)

            # counterfactual accuracy: compare each perturbation to its gold label if available
            cf_correct = 0
            cf_total = 0
            for row in cva_rows:
                p = row["perturbation"]
                if p == "orig":
                    continue
                gold_cf = inst_cf_labels.get(p, None)
                if gold_cf is None:
                    continue
                cf_total += 1
                cf_correct += acc(row["decision"], gold_cf)

            cf_acc = (cf_correct / cf_total) if cf_total > 0 else ""

            # Write CSV rows
            for method in ["direct", "cot", "tot"]:
                d = per_method_outputs[method]
                csv_rows.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "model": MODEL_NAME,
                        "sample_id": sid,
                        "category": cat,
                        "label": label,
                        "method": method,
                        "perturbation": "orig",
                        "decision": d["decision"],
                        "reasoning": d["reasoning"],
                        "css": "",
                        "orig_acc": direct_acc if method == "direct" else (cot_acc if method == "cot" else tot_acc),
                        "cf_acc": "",
                    }
                )

            for row in cva_rows:
                gold_cf = inst_cf_labels.get(row["perturbation"], "")
                this_cf_acc = ""
                if row["perturbation"] != "orig" and gold_cf:
                    this_cf_acc = acc(row["decision"], gold_cf)
                csv_rows.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "model": MODEL_NAME,
                        "sample_id": sid,
                        "category": cat,
                        "label": label,
                        "method": "cva",
                        "perturbation": row["perturbation"],
                        "decision": row["decision"],
                        "reasoning": row["reasoning"],
                        "css": css if row["perturbation"] == "orig" else "",
                        "orig_acc": acc(root_dec, label) if row["perturbation"] == "orig" else "",
                        "cf_acc": this_cf_acc,
                    }
                )

            # JSONL record (per instance)
            f_log.write(
                json.dumps(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "model": MODEL_NAME,
                        "sample_id": sid,
                        "category": cat,
                        "label": label,
                        "direct": per_method_outputs["direct"],
                        "cot": per_method_outputs["cot"],
                        "tot": per_method_outputs["tot"],
                        "cva": {
                            "css": css,
                            "rows": cva_rows,
                            "cf_acc": cf_acc,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    print(f"[green]Wrote JSONL → {log_path}[/green]")
    print(f"[green]Wrote CSV  → {csv_path}[/green]")


if __name__ == "__main__":
    main()
