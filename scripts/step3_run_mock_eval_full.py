from __future__ import annotations
import csv
import yaml
from pathlib import Path
from rich import print
from rich.table import Table

from cycb.io import load_jsonl
from cycb.agents import MockLLM, run_direct, run_cot, run_tot, run_multi_agent, run_cva
from cycb.metrics import css_from_cva, alignment_accuracy, dsi_for_instance, aggregate_metrics
from cycb.audit import compute_atd

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

def main() -> None:
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    instances = load_jsonl(cfg["dataset_path"])
    llm = MockLLM()

    methods = ["Direct", "CoT", "ToT", "MultiAgent", "CVA"]
    results = {m: [] for m in methods}

    for inst in instances:
        # Run baselines
        r_direct = run_direct(inst, llm)
        r_cot = run_cot(inst, llm)
        r_tot = run_tot(inst, llm)
        r_ma = run_multi_agent(inst, llm, n_agents=3)

        # Run CVA (root + counterfactuals)
        r_cva = run_cva(inst, llm)
        cva_css = css_from_cva(r_cva)
        cva_cf_decisions = [(p, rr.decision) for p, rr in r_cva.counterfactuals]

        # alignment
        acc_o_direct, _ = alignment_accuracy(inst, r_direct.decision, [])
        acc_o_cot, _ = alignment_accuracy(inst, r_cot.decision, [])
        acc_o_tot, _ = alignment_accuracy(inst, r_tot.decision, [])
        acc_o_ma, _ = alignment_accuracy(inst, r_ma.decision, [])
        acc_o_cva, acc_cf_cva = alignment_accuracy(inst, r_cva.root.decision, cva_cf_decisions)

        # ATD
        atd_direct = compute_atd(r_direct.reasoning, "Direct")
        atd_cot = compute_atd(r_cot.reasoning, "CoT")
        atd_tot = compute_atd(r_tot.reasoning, "ToT")
        atd_ma = compute_atd(r_ma.reasoning, "MultiAgent")
        atd_cva = compute_atd("", "CVA", num_nodes=1 + len(r_cva.counterfactuals))

        # DSI
        dsi_direct = dsi_for_instance(inst, r_direct.decision, r_direct.reasoning, css=None)
        dsi_cot = dsi_for_instance(inst, r_cot.decision, r_cot.reasoning, css=None)
        dsi_tot = dsi_for_instance(inst, r_tot.decision, r_tot.reasoning, css=None)
        dsi_ma = dsi_for_instance(inst, r_ma.decision, r_ma.reasoning, css=None)
        dsi_cva = dsi_for_instance(inst, r_cva.root.decision, r_cva.root.reasoning, css=cva_css)

        # store rows
        results["Direct"].append({"sample_id": inst.sample_id, "decision": r_direct.decision, "dsi": dsi_direct,
                                 "atd": atd_direct, "orig_acc": acc_o_direct, "css": None, "cf_acc": None})
        results["CoT"].append({"sample_id": inst.sample_id, "decision": r_cot.decision, "dsi": dsi_cot,
                               "atd": atd_cot, "orig_acc": acc_o_cot, "css": None, "cf_acc": None})
        results["ToT"].append({"sample_id": inst.sample_id, "decision": r_tot.decision, "dsi": dsi_tot,
                               "atd": atd_tot, "orig_acc": acc_o_tot, "css": None, "cf_acc": None})
        results["MultiAgent"].append({"sample_id": inst.sample_id, "decision": r_ma.decision, "dsi": dsi_ma,
                                      "atd": atd_ma, "orig_acc": acc_o_ma, "css": None, "cf_acc": None})
        results["CVA"].append({"sample_id": inst.sample_id, "decision": r_cva.root.decision, "dsi": dsi_cva,
                               "atd": atd_cva, "orig_acc": acc_o_cva, "css": cva_css, "cf_acc": acc_cf_cva})

    # Aggregate
    avgs = aggregate_metrics(instances, results)

    # Print Tables 5–8 scaffolds
    t5 = Table(title="Table 5 scaffold — CSS (higher is better)")
    t5.add_column("Method")
    t5.add_column("Avg CSS", justify="right")
    for m in methods:
        t5.add_row(m, f"{avgs[m].avg_css:.2f}" if avgs[m].avg_css is not None else "—")
    print(t5)

    t6 = Table(title="Table 6 scaffold — DSI (higher is better)")
    t6.add_column("Method")
    t6.add_column("Avg DSI", justify="right")
    for m in methods:
        t6.add_row(m, f"{avgs[m].avg_dsi:.2f}")
    print(t6)

    t7 = Table(title="Table 7 scaffold — ATD (higher is richer auditability)")
    t7.add_column("Method")
    t7.add_column("Avg ATD", justify="right")
    for m in methods:
        t7.add_row(m, f"{avgs[m].avg_atd:.2f}")
    print(t7)

    t8 = Table(title="Table 8 scaffold — Alignment accuracy")
    t8.add_column("Method")
    t8.add_column("Orig Acc", justify="right")
    t8.add_column("CF Acc", justify="right")
    for m in methods:
        t8.add_row(
            m,
            f"{avgs[m].orig_acc:.2f}",
            f"{avgs[m].cf_acc:.2f}" if avgs[m].cf_acc is not None else "—",
        )
    print(t8)

    # Export CSV for paper tables
    out_path = OUT_DIR / "table_metrics_mock.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
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

    print(f"\n[bold green]Saved CSV:[/bold green] {out_path}")

if __name__ == "__main__":
    main()
