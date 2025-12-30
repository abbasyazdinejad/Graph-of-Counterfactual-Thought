from __future__ import annotations

import os
import csv
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich import print
from rich.table import Table

from cycb.io import load_jsonl
from cycb.perturbations import apply_perturbation
from cycb.prompts import DEFAULT_PROMPTS
from cycb.metrics import css_from_cva, dsi_for_instance, aggregate_metrics
from cycb.audit import compute_atd
from cycb.llm_openai import OpenAIChatLLM



MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0

# CVA perturbations used in your existing working gpt41 script pattern
PERTURBATIONS = [
    "entity_swap",
    "time_shift",
    "negation_flip",
    "irrelevant_injection",
]


def main() -> None:
    load_dotenv()

    # IMPORTANT: OpenAIChatLLM overrides model from env OPENAI_MODEL
    os.environ["OPENAI_MODEL"] = MODEL_NAME

    repo_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text())

    dataset_path = repo_root / cfg["dataset_path"]
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)

    log_path = out_dir / f"openai_eval_log_{MODEL_NAME}.jsonl"
    out_csv = out_dir / f"table_metrics_openai_{MODEL_NAME}.csv"

    # fresh log
    if log_path.exists():
        log_path.unlink()

    llm = OpenAIChatLLM(temperature=TEMPERATURE)

    instances = load_jsonl(str(dataset_path))
    methods = ["Direct", "CoT", "ToT", "CVA"]

    # This is the structure expected by cycb.metrics.aggregate_metrics()
    # Dict[str, List[dict]] where each dict includes keys like:
    # sample_id, decision, dsi, atd, orig_acc, css, cf_acc
    results: dict[str, list[dict]] = {m: [] for m in methods}

    for inst in instances:
        print(f"[bold cyan]== {inst.sample_id} ({inst.category}) ==[/bold cyan]")

        evidence = inst.evidence_text()

  
        sys_direct = DEFAULT_PROMPTS["direct"]["system"]
        usr_direct = DEFAULT_PROMPTS["direct"]["user"].format(evidence=evidence)
        r_direct = llm.predict(sys_direct, usr_direct, method="Direct")
        acc_o_direct = 1.0 if (r_direct.decision == inst.label) else 0.0
        dsi_direct = dsi_for_instance(inst, r_direct.decision, r_direct.reasoning, css=None)
        atd_direct = compute_atd(r_direct.reasoning or "", method="Direct")

        sys_cot = DEFAULT_PROMPTS["cot"]["system"]
        usr_cot = DEFAULT_PROMPTS["cot"]["user"].format(evidence=evidence)
        r_cot = llm.predict(sys_cot, usr_cot, method="CoT")
        acc_o_cot = 1.0 if (r_cot.decision == inst.label) else 0.0
        dsi_cot = dsi_for_instance(inst, r_cot.decision, r_cot.reasoning, css=None)
        atd_cot = compute_atd(r_cot.reasoning or "", method="CoT")

        sys_tot = DEFAULT_PROMPTS["tot"]["system"]
        usr_tot = DEFAULT_PROMPTS["tot"]["user"].format(evidence=evidence)
        r_tot = llm.predict(sys_tot, usr_tot, method="ToT")
        acc_o_tot = 1.0 if (r_tot.decision == inst.label) else 0.0
        dsi_tot = dsi_for_instance(inst, r_tot.decision, r_tot.reasoning, css=None)
        atd_tot = compute_atd(r_tot.reasoning or "", method="ToT")

     
        # -----------------------
        sys_cva = DEFAULT_PROMPTS["cva"]["system"]
        usr_cva = DEFAULT_PROMPTS["cva"]["user"].format(evidence=evidence)
        root = llm.predict(sys_cva, usr_cva, method="CVA_root")
        acc_o_cva = 1.0 if (root.decision == inst.label) else 0.0

        cfs = []
        for pname in PERTURBATIONS:
            cf_evidence = apply_perturbation(evidence, pname)
            usr_cf = DEFAULT_PROMPTS["cva"]["user"].format(evidence=cf_evidence)
            rr = llm.predict(sys_cva, usr_cf, method=f"CVA_{pname}")
            cfs.append((pname, rr))

        # CSS + CF Acc for CVA
        cva_css = css_from_cva(root.decision, [rr.decision for _, rr in cfs])
        acc_cf_cva = sum(1.0 for _, rr in cfs if rr.decision == inst.label) / max(len(cfs), 1)

        # DSI + ATD for CVA
        dsi_cva = dsi_for_instance(inst, root.decision, root.reasoning, css=cva_css)
        atd_cva = compute_atd(root.reasoning or "", method="CVA")

        # build a compact g_cot artifact (optional, but keeps logs interpretable)
        g_cot = {
            "root": {"decision": root.decision, "reasoning": root.reasoning},
            "counterfactuals": [
                {"perturbation": pname, "decision": rr.decision, "reasoning": rr.reasoning}
                for pname, rr in cfs
            ],
        }

     
        # -----------------------
        results["Direct"].append(
            {"sample_id": inst.sample_id, "decision": r_direct.decision, "dsi": dsi_direct,
             "atd": atd_direct, "orig_acc": acc_o_direct, "css": None, "cf_acc": None}
        )
        results["CoT"].append(
            {"sample_id": inst.sample_id, "decision": r_cot.decision, "dsi": dsi_cot,
             "atd": atd_cot, "orig_acc": acc_o_cot, "css": None, "cf_acc": None}
        )
        results["ToT"].append(
            {"sample_id": inst.sample_id, "decision": r_tot.decision, "dsi": dsi_tot,
             "atd": atd_tot, "orig_acc": acc_o_tot, "css": None, "cf_acc": None}
        )
        results["CVA"].append(
            {"sample_id": inst.sample_id, "decision": root.decision, "dsi": dsi_cva,
             "atd": atd_cva, "orig_acc": acc_o_cva, "css": cva_css, "cf_acc": acc_cf_cva}
        )

   
        record = {
            "sample_id": inst.sample_id,
            "category": inst.category,
            "label": inst.label,
            "model": MODEL_NAME,
            "predictions": {
                "Direct": {"decision": r_direct.decision, "reasoning": r_direct.reasoning},
                "CoT": {"decision": r_cot.decision, "reasoning": r_cot.reasoning},
                "ToT": {"decision": r_tot.decision, "reasoning": r_tot.reasoning},
                "CVA": {
                    "root": {"decision": root.decision, "reasoning": root.reasoning},
                    "counterfactuals": [
                        {"perturbation": pname, "decision": rr.decision, "reasoning": rr.reasoning}
                        for pname, rr in cfs
                    ],
                    "g_cot": g_cot,
                },
            },
            "metrics": {
                "Direct": {"orig_acc": acc_o_direct, "dsi": dsi_direct, "atd": atd_direct},
                "CoT": {"orig_acc": acc_o_cot, "dsi": dsi_cot, "atd": atd_cot},
                "ToT": {"orig_acc": acc_o_tot, "dsi": dsi_tot, "atd": atd_tot},
                "CVA": {"orig_acc": acc_o_cva, "cf_acc": acc_cf_cva, "dsi": dsi_cva, "atd": atd_cva, "css": cva_css},
            },
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


    avgs = aggregate_metrics(instances, results)

    t = Table(title=f"Step 4b (OpenAI) — {MODEL_NAME} — Summary + JSON Log")
    t.add_column("Method")
    t.add_column("Avg CSS", justify="right")
    t.add_column("Avg DSI", justify="right")
    t.add_column("Avg ATD", justify="right")
    t.add_column("Orig Acc", justify="right")
    t.add_column("CF Acc", justify="right")

    for m in methods:
        t.add_row(
            m,
            f"{avgs[m].avg_css:.2f}" if avgs[m].avg_css is not None else "—",
            f"{avgs[m].avg_dsi:.2f}",
            f"{avgs[m].avg_atd:.2f}",
            f"{avgs[m].orig_acc:.2f}",
            f"{avgs[m].cf_acc:.2f}" if avgs[m].cf_acc is not None else "—",
        )
    print(t)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Avg_CSS", "Avg_DSI", "Avg_ATD", "Orig_Acc", "CF_Acc"])
        for m in methods:
            w.writerow([
                m,
                f"{avgs[m].avg_css:.4f}" if avgs[m].avg_css is not None else "",
                f"{avgs[m].avg_dsi:.4f}",
                f"{avgs[m].avg_atd:.4f}",
                f"{avgs[m].orig_acc:.4f}",
                f"{avgs[m].cf_acc:.4f}" if avgs[m].cf_acc is not None else "",
            ])

    print(f"[bold green]Wrote JSONL log -> {log_path}[/bold green]")
    print(f"[bold green]Wrote CSV -> {out_csv}[/bold green]")


if __name__ == "__main__":
    main()
