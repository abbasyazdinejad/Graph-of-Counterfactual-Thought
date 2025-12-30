from __future__ import annotations

import json, csv
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import yaml
from dotenv import load_dotenv
from rich import print
from cycb.audit import compute_atd
from cycb.metrics import dsi_for_instance, css_from_cva, aggregate_metrics

from cycb.io import load_jsonl
from cycb.agents import run_direct, run_cot, run_tot, run_cva
from cycb.llm_openai import OpenAIChatLLM

from cycb.metrics import (
    aggregate_metrics,
    dsi_for_instance,
    compute_atd,
    css_from_cva,
)


MODEL_NAME = "gpt-4o"


def main() -> None:
    load_dotenv()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text())

    dataset_path = repo_root / cfg["dataset_path"]
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)

    log_path = out_dir / f"openai_eval_log_{MODEL_NAME}.jsonl"
    csv_path = out_dir / f"table_metrics_openai_{MODEL_NAME}.csv"

    llm = OpenAIChatLLM(model=MODEL_NAME)

    instances = load_jsonl(str(dataset_path))
    all_rows = []

    with open(log_path, "w", encoding="utf-8") as json_out:
        for inst in instances:
            print(f"[bold cyan]== {inst.sample_id} ({inst.label}) ==[/bold cyan]")

       
            raw_results = {
                "direct": [run_direct(inst, llm)],
                "cot":    [run_cot(inst, llm)],
                "tot":    [run_tot(inst, llm)],
                "cva":    [run_cva(inst, llm)],
            }

   
            results = {}

            for method, runs in raw_results.items():
                rows = []
                for r in runs:
                    css = css_from_cva(inst, r) if method == "cva" else None
                    row = {
                        "decision": r.decision,
                        "reasoning": r.reasoning,
                        "method": r.method,
                        "css": css,
                        "dsi": dsi_for_instance(inst, r.decision, r.reasoning, css),
                        "atd": compute_atd((r.reasoning or "")),
                    }

       
                    
                    rows.append(row)

                results[method] = rows


            agg_row = aggregate_metrics(inst, results)
            all_rows.append(agg_row)


            json_out.write(json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "sample_id": inst.sample_id,
                "label": inst.label,
                "category": inst.category,
                "model": MODEL_NAME,
                "results": results,
                "metrics": agg_row
            }) + "\n")


    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[green]Wrote JSON log → {log_path}[/green]")
    print(f"[green]Wrote CSV → {csv_path}[/green]")


if __name__ == "__main__":
    main()
