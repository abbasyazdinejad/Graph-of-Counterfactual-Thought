from __future__ import annotations
import csv
from pathlib import Path

CSV_PATH = Path("results/table_metrics_openai.csv")
OUT_TEX = Path("results/tables_openai.tex")

def f(x: str) -> str:
    if x is None:
        return ""
    x = x.strip()
    if x == "" or x.lower() == "nan":
        return ""
    return x

def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH}. Run step4_run_openai_eval.py first.")

    rows = []
    with CSV_PATH.open("r", encoding="utf-8") as fcsv:
        r = csv.DictReader(fcsv)
        for row in r:
            rows.append(row)

    # Expected columns:
    # Method, Avg_CSS, Avg_DSI, Avg_ATD, Orig_Acc, CF_Acc
    # (your code wrote: "Method","Avg_CSS","Avg_DSI","Avg_ATD","Orig_Acc","CF_Acc")
    by_method = {row["Method"]: row for row in rows}

    methods = ["Direct", "CoT", "ToT", "MultiAgent", "CVA"]
    # Some runs may not include MultiAgent — handle gracefully
    methods_present = [m for m in methods if m in by_method]

    # Build LaTeX tables
    def table_metric(caption: str, label: str, col_key: str, header: str) -> str:
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{l r}")
        lines.append(r"\toprule")
        lines.append(rf"Method & {header} \\")
        lines.append(r"\midrule")
        for m in methods_present:
            val = f(by_method[m].get(col_key, ""))
            if val == "":
                val = r"\textemdash"
            lines.append(rf"{m} & {val} \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(rf"\caption{{{caption}}}")
        lines.append(rf"\label{{{label}}}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def table_alignment() -> str:
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{l r r}")
        lines.append(r"\toprule")
        lines.append(r"Method & Orig Acc & CF Acc \\")
        lines.append(r"\midrule")
        for m in methods_present:
            o = f(by_method[m].get("Orig_Acc", ""))
            c = f(by_method[m].get("CF_Acc", ""))
            if o == "": o = r"\textemdash"
            if c == "": c = r"\textemdash"
            lines.append(rf"{m} & {o} & {c} \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Alignment accuracy on original instances and counterfactual instances.}")
        lines.append(r"\label{tab:alignment-acc}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    tex_blocks = []
    tex_blocks.append("% Auto-generated from results/table_metrics_openai.csv\n")
    tex_blocks.append(table_metric(
        caption="Counterfactual Stability Score (CSS) for methods that generate counterfactual reasoning.",
        label="tab:css",
        col_key="Avg_CSS",
        header="Avg CSS"
    ))
    tex_blocks.append("")
    tex_blocks.append(table_metric(
        caption="Decision Shift Index (DSI) across methods (lower is more stable).",
        label="tab:dsi",
        col_key="Avg_DSI",
        header="Avg DSI"
    ))
    tex_blocks.append("")
    tex_blocks.append(table_metric(
        caption="Average Token Distance (ATD) as a proxy for reasoning divergence cost.",
        label="tab:atd",
        col_key="Avg_ATD",
        header="Avg ATD"
    ))
    tex_blocks.append("")
    tex_blocks.append(table_alignment())

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n\n".join(tex_blocks) + "\n", encoding="utf-8")
    print(f" Wrote LaTeX tables -> {OUT_TEX}")

if __name__ == "__main__":
    main()
