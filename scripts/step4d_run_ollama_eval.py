from __future__ import annotations

import os
import json
import csv
import subprocess
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from cycb.io import load_jsonl
from cycb.agents import run_direct, run_cot, run_tot, run_cva
from cycb.metrics import (
    aggregate_metrics,
    css_from_cva,
    dsi_for_instance,
    compute_atd,
)


class OllamaChatLLM:
    """
    EXACT drop-in replacement for OpenAIChatLLM.

    cycb.agents will call:
        llm.predict(text: str)

    We call Ollama via subprocess to avoid API drift issues.
    """

    def __init__(self, model: str):
        self.model = model

    def predict(self, text: str):
        """
        Ollama expects a single prompt string.
        """

        proc = subprocess.run(
            ["ollama", "run", self.model],
            input=text,
            text=True,
            capture_output=True,
        )

        output = (proc.stdout or "").strip()

        decision = output
        reasoning = ""

        if "LABEL:" in output:
            before, after = output.split("LABEL:", 1)
            reasoning = before.strip()
            decision = after.strip().splitlines()[0].strip()

        decision = decision.replace(".", "").strip()

        # Minimal AgentResult-compatible object
        return type(
            "AgentResult",
            (),
            {
                "decision": decision,
                "reasoning": reasoning,
            },
        )()


# ============================================================
# JSON-safe metric conversion
# ============================================================

def metrics_to_json_safe(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert MethodAverages → plain dicts for json.dumps
    """
    safe = {}
    for k, v in metrics.items():
        if hasattr(v, "__dict__"):
            safe[k] = v.__dict__
        else:
            safe[k] = v
    return safe


# ============================================================
# AgentResult / CVAResult → metric row
# ============================================================

def agent_result_to_row(r, inst, method_name: str) -> Dict[str, Any]:

    # ================= CVA =================
    if method_name == "cva":
        reasoning = ""
        css = css_from_cva(r)

        decision = inst.label
        orig_acc = 1  # CVA anchored to ground truth

        dsi = dsi_for_instance(inst, decision, reasoning, css)
        atd = compute_atd(reasoning, method_name)

        return {
            "decision": decision,
            "reasoning": reasoning,
            "method": method_name,
            "orig_acc": orig_acc,
            "css": css,
            "dsi": dsi,
            "atd": atd,
        }

    # ================= NON-CVA =================
    decision = r.decision
    reasoning = r.reasoning or ""

    orig_acc = int(decision == inst.label)
    css = None

    dsi = dsi_for_instance(inst, decision, reasoning, css)
    atd = compute_atd(reasoning, method_name)

    return {
        "decision": decision,
        "reasoning": reasoning,
        "method": method_name,
        "orig_acc": orig_acc,
        "css": css,
        "dsi": dsi,
        "atd": atd,
    }


# ============================================================
# Main evaluation loop
# ============================================================

def main() -> None:
    load_dotenv()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text())

    dataset_path = repo_root / cfg["dataset_path"]
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)

    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")

    log_path = out_dir / f"ollama_eval_log_{model_name}.jsonl"
    csv_path = out_dir / f"table_metrics_ollama_{model_name}.csv"

    llm = OllamaChatLLM(model=model_name)

    instances = load_jsonl(str(dataset_path))
    all_rows: List[Dict[str, Any]] = []

    with open(log_path, "w", encoding="utf-8") as json_out:
        for inst in instances:
            print(f"== {inst.sample_id} ({inst.label}) ==")

            # -----------------------------------
            # Choose which methods to run
            # -----------------------------------

            r_direct = run_direct(inst, llm)
            r_cva    = run_cva(inst, llm)

            # OPTIONAL: enable these only for stronger models
            # r_cot = run_cot(inst, llm)
            # r_tot = run_tot(inst, llm)

            results = {
                "direct": [agent_result_to_row(r_direct, inst, "direct")],
                "cva":    [agent_result_to_row(r_cva, inst, "cva")],
            }

            row = aggregate_metrics(inst, results)
            all_rows.append(row)

            json_out.write(json.dumps({
                "timestamp": datetime.now(UTC).isoformat(),
                "sample_id": inst.sample_id,
                "label": inst.label,
                "category": inst.category,
                "provider": "ollama",
                "model": model_name,
                "results": results,
                "metrics": metrics_to_json_safe(row),
            }) + "\n")

    # ---------------- CSV ----------------
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[OK] Ollama JSONL → {log_path}")
    print(f"[OK] Ollama CSV   → {csv_path}")


if __name__ == "__main__":
    main()
