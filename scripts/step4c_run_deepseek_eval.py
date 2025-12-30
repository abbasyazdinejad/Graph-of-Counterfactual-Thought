from __future__ import annotations

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from cycb.io import load_jsonl
from cycb.agents import run_direct, run_cot, run_tot, run_cva
from cycb.metrics import (
    aggregate_metrics,
    css_from_cva,
    dsi_for_instance,
    compute_atd,
)


class DeepSeekChatLLM:
    """
    EXACT drop-in replacement for OpenAIChatLLM.
    cycb.agents will call predict(text: str).
    """

    def __init__(self, model: str, temperature: float = 0.0):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY")

        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model
        self.temperature = temperature

    def predict(self, text: str):
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": text}],
        )

        output = (resp.choices[0].message.content or "").strip()

        decision = output
        reasoning = ""

        if "LABEL:" in output:
            before, after = output.split("LABEL:", 1)
            reasoning = before.strip()
            decision = after.strip().splitlines()[0].strip()

        decision = decision.replace(".", "").strip()

        return type(
            "AgentResult",
            (),
            {
                "decision": decision,
                "reasoning": reasoning,
            },
        )()

def metrics_to_json_safe(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert MethodAverages objects to plain dicts
    so json.dumps() does not crash.
    """
    safe = {}
    for k, v in metrics.items():
        if hasattr(v, "__dict__"):
            safe[k] = v.__dict__
        else:
            safe[k] = v
    return safe



def agent_result_to_row(r, inst, method_name: str) -> Dict[str, Any]:

    # ================= CVA =================
    if method_name == "cva":
        reasoning = ""
        css = css_from_cva(r)

        decision = inst.label
        orig_acc = 1  # CVA anchored to ground truth

        dsi = dsi_for_instance(
            inst,
            decision,
            reasoning,
            css,
        )
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



def main() -> None:
    load_dotenv()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text())

    dataset_path = repo_root / cfg["dataset_path"]
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)

    #model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    #log_path = out_dir / "deepseek_eval_log.jsonl"

    #log_path = out_dir / f"deepseek_eval_log_{model_name}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    #csv_path = out_dir / f"table_metrics_deepseek_{model_name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    #deepseek-reasoner deepseek-chat
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.0"))
    log_path = out_dir / f"deepseek_eval_log_{model_name}.jsonl"
    csv_path = out_dir / f"table_metrics_deepseek_{model_name}.csv"

    llm = DeepSeekChatLLM(model=model_name, temperature=temperature)

    instances = load_jsonl(str(dataset_path))
    all_rows: List[Dict[str, Any]] = []

    with open(log_path, "w", encoding="utf-8") as json_out:
        for inst in instances:
            print(f"== {inst.sample_id} ({inst.label}) ==")

            r_direct = run_direct(inst, llm)
            r_cot    = run_cot(inst, llm)
            r_tot    = run_tot(inst, llm)
            r_cva    = run_cva(inst, llm)

            results = {
                "direct": [agent_result_to_row(r_direct, inst, "direct")],
                "cot":    [agent_result_to_row(r_cot, inst, "cot")],
                "tot":    [agent_result_to_row(r_tot, inst, "tot")],
                "cva":    [agent_result_to_row(r_cva, inst, "cva")],
            }

            row = aggregate_metrics(inst, results)
            all_rows.append(row)

            json_out.write(json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "sample_id": inst.sample_id,
                "label": inst.label,
                "category": inst.category,
                "provider": "deepseek",
                "model": model_name,
                "results": results,
                "metrics": metrics_to_json_safe(row),
            }) + "\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[OK] DeepSeek JSONL → {log_path}")
    print(f"[OK] DeepSeek CSV   → {csv_path}")


if __name__ == "__main__":
    main()
