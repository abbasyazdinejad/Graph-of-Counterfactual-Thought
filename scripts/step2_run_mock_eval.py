from __future__ import annotations
import yaml
from pathlib import Path
from rich import print
from rich.table import Table

from cycb.io import load_jsonl
from cycb.agents import MockLLM, run_direct, run_cot, run_tot, run_multi_agent, run_cva
from cycb.metrics import css_from_cva, alignment_accuracy

def main() -> None:
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    instances = load_jsonl(cfg["dataset_path"])

    llm = MockLLM()

    # Track aggregates
    methods = ["Direct", "CoT", "ToT", "MultiAgent", "CVA"]
    css_sum = {m: 0.0 for m in methods}
    acc_orig_sum = {m: 0.0 for m in methods}
    acc_cf_sum = {m: 0.0 for m in methods}
    n = len(instances)

    for inst in instances:
        print(f"\n[bold cyan]=== {inst.sample_id} ({inst.category}) ===[/bold cyan]")
        print(f"GT label: {inst.label}")
        print(f"CF GT labels: {inst.counterfactual_labels}")

        # Baselines (for now we treat them as "no counterfactual run", so css=NA later)
        r_direct = run_direct(inst, llm)
        r_cot = run_cot(inst, llm)
        r_tot = run_tot(inst, llm)
        r_ma = run_multi_agent(inst, llm, n_agents=3)
        r_cva = run_cva(inst, llm)

        # compute metrics
        cva_css = css_from_cva(r_cva)
        cva_cf_decisions = [(p, rr.decision) for p, rr in r_cva.counterfactuals]

        for m, root_dec in [
            ("Direct", r_direct.decision),
            ("CoT", r_cot.decision),
            ("ToT", r_tot.decision),
            ("MultiAgent", r_ma.decision),
            ("CVA", r_cva.root.decision),
        ]:
            acc_o, acc_c = alignment_accuracy(inst, root_dec, cva_cf_decisions if m == "CVA" else [])
            acc_orig_sum[m] += acc_o
            acc_cf_sum[m] += acc_c

        css_sum["CVA"] += cva_css

        print(f"Direct: {r_direct.decision}")
        print(f"CoT: {r_cot.decision}")
        print(f"ToT: {r_tot.decision}")
        print(f"MultiAgent: {r_ma.decision}")
        print(f"CVA root: {r_cva.root.decision} | CSS={cva_css:.2f}")
        for pname, rr in r_cva.counterfactuals:
            print(f"  - CF({pname}): {rr.decision}")

    # Summary table (scaffold for Table 5 + Table 8)
    t = Table(title="Step 2 (Mock) — Summary (CSS + Alignment)")
    t.add_column("Method")
    t.add_column("Avg CSS", justify="right")
    t.add_column("Orig Acc", justify="right")
    t.add_column("CF Acc", justify="right")

    for m in methods:
        avg_css = (css_sum[m] / n) if m == "CVA" else 0.0
        t.add_row(
            m,
            f"{avg_css:.2f}" if m == "CVA" else "—",
            f"{(acc_orig_sum[m]/n):.2f}",
            f"{(acc_cf_sum[m]/n):.2f}" if m == "CVA" else "—",
        )

    print("\n")
    print(t)

if __name__ == "__main__":
    main()
