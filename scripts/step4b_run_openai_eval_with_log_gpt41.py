from __future__ import annotations

import json
import csv
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich import print
from rich.table import Table

from cycb.io import load_jsonl
from cycb.perturbations import apply_perturbation
from cycb.prompts import DEFAULT_PROMPTS
from cycb.metrics import css_from_cva, alignment_accuracy, dsi_for_instance, aggregate_metrics
from cycb.audit import compute_atd
from cycb.llm_openai import OpenAIChatLLM


def run_llm(inst, llm, mode: str):
    p = DEFAULT_PROMPTS
    if mode == "Direct":
        user = p.direct.format(evidence_text=inst.evidence_text())
        return llm.predict(p.system, user, method="Direct")
    if mode == "CoT":
        user = p.cot.format(evidence_text=inst.evidence_text())
        return llm.predict(p.system, user, method="CoT")
    if mode == "ToT":
        user = p.tot.format(evidence_text=inst.evidence_text())
        return llm.predict(p.system, user, method="ToT")
    raise ValueError(mode)


def run_cva(inst, llm):
    # Root with CoT prompt (more stable + auditable)
    root = run_llm(inst, llm, "CoT")

    cfs = []
    for pname in inst.counterfactual_labels.keys():
        inst_cf = apply_perturbation(inst, pname)
        r_cf = run_llm(inst_cf, llm, "CoT")
        cfs.append((pname, r_cf))

    g_cot = {
        "root": {"sample_id": inst.sample_id, "decision": root.decision},
        "counterfactuals": [{"perturbation": p, "decision": r.decision} for p, r in cfs],
    }
    return root, cfs, g_cot


def main() -> None:
    # --- Resolve repo root no matter where you run this from ---
    repo_root = Path(__file__).resolve().parents[1]

    # --- Load .env from repo root (fixes your OPENAI_API_KEY issue) ---
    load_dotenv(repo_root / ".env", override=True)

    # --- Load config and dataset ---
    cfg = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text(encoding="utf-8"))
    dataset_path = repo_root / cfg["dataset_path"]
    instances = load_jsonl(str(dataset_path))

    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON log you want ---
    log_path = out_dir / "openai_eval_log.jsonl"

    # --- OpenAI LLM ---
    llm = OpenAIChatLLM(temperature=0.0)

    methods = ["Direct", "CoT", "ToT", "CVA"]
    results = {m: [] for m in methods}

    # Clear old log if exists
    if log_path.exists():
        log_path.unlink()

    for inst in instances:
        print(f"\n[bold cyan]=== {inst.sample_id} ({inst.category}) ===[/bold cyan]")

        r_direct = run_llm(inst, llm, "Direct")
        r_cot = run_llm(inst, llm, "CoT")
        r_tot = run_llm(inst, llm, "ToT")

        root, cfs, g_cot = run_cva(inst, llm)
        cva_css = css_from_cva(type("Tmp", (), {"root": root, "counterfactuals": cfs})())
        cva_cf_decisions = [(p, rr.decision) for p, rr in cfs]

        # alignment
        acc_o_direct, _ = alignment_accuracy(inst, r_direct.decision, [])
        acc_o_cot, _ = alignment_accuracy(inst, r_cot.decision, [])
        acc_o_tot, _ = alignment_accuracy(inst, r_tot.decision, [])
        acc_o_cva, acc_cf_cva = alignment_accuracy(inst, root.decision, cva_cf_decisions)

        # ATD
        atd_direct = compute_atd(r_direct.reasoning, "Direct")
        atd_cot = compute_atd(r_cot.reasoning, "CoT")
        atd_tot = compute_atd(r_tot.reasoning, "ToT")
        atd_cva = compute_atd("", "CVA", num_nodes=1 + len(cfs))

        # DSI
        dsi_direct = dsi_for_instance(inst, r_direct.decision, r_direct.reasoning, css=None)
        dsi_cot = dsi_for_instance(inst, r_cot.decision, r_cot.reasoning, css=None)
        dsi_tot = dsi_for_instance(inst, r_tot.decision, r_tot.reasoning, css=None)
        dsi_cva = dsi_for_instance(inst, root.decision, root.reasoning, css=cva_css)

        results["Direct"].append({"sample_id": inst.sample_id, "decision": r_direct.decision, "dsi": dsi_direct,
                                 "atd": atd_direct, "orig_acc": acc_o_direct, "css": None, "cf_acc": None})
        results["CoT"].append({"sample_id": inst.sample_id, "decision": r_cot.decision, "dsi": dsi_cot,
                               "atd": atd_cot, "orig_acc": acc_o_cot, "css": None, "cf_acc": None})
        results["ToT"].append({"sample_id": inst.sample_id, "decision": r_tot.decision, "dsi": dsi_tot,
                               "atd": atd_tot, "orig_acc": acc_o_tot, "css": None, "cf_acc": None})
        results["CVA"].append({"sample_id": inst.sample_id, "decision": root.decision, "dsi": dsi_cva,
                               "atd": atd_cva, "orig_acc": acc_o_cva, "css": cva_css, "cf_acc": acc_cf_cva})

        # --- Write one JSONL record per instance ---
        record = {
            "sample_id": inst.sample_id,
            "category": inst.category,
            "label": inst.label,
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

    # --- Summary (same style as step4) ---
    avgs = aggregate_metrics(instances, results)
    t = Table(title="Step 4b (OpenAI) — Summary + JSON Log")
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

    # Optionally also save a CSV like step4 (keeps things consistent)
    out_csv = out_dir / "table_metrics_openai_step4b.csv"
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
