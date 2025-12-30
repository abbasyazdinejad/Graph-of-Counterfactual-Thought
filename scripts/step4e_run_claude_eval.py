# scripts/step4e_run_claude_eval.py
from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich import print
from rich.table import Table

import anthropic

from cycb.labels import normalize_label
from cycb.io import load_jsonl
from cycb.perturbations import apply_perturbation
from cycb.prompts import DEFAULT_PROMPTS
from cycb.metrics import alignment_accuracy, dsi_for_instance, aggregate_metrics
from cycb.audit import compute_atd

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)




@dataclass
class AgentResult:
    decision: str
    reasoning: str


def _extract_label(output: str) -> tuple[str, str]:
    """
    Supports both:
      - "FINAL LABEL: <label>"
      - "LABEL: <label>"
    Returns (decision, reasoning).
    """
    out = (output or "").strip()
    for tag in ("FINAL LABEL:", "LABEL:", "Final Label:", "Final label:", "label:"):
        if tag in out:
            before, after = out.split(tag, 1)
            reasoning = before.strip()
            decision = after.strip().splitlines()[0].strip()
            return decision.rstrip(".").strip(), reasoning

    # fallback: first line
    first = out.splitlines()[0].strip() if out else ""
    return first.rstrip(".").strip(), ""


class ClaudeChatLLM:
    """
    Drop-in for OpenAIChatLLM used by your step4 pipeline:
      predict(system_prompt=..., user_prompt=..., method=...)
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0, max_tokens: int = 1024):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
        self.temperature = float(os.getenv("CLAUDE_TEMPERATURE", str(temperature)))
        self.max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", str(max_tokens)))

    def predict(self, system_prompt: str, user_prompt: str, method: str = "") -> AgentResult:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # msg.content is a list of blocks; concatenate all .text safely
        chunks = []
        for b in (msg.content or []):
            t = getattr(b, "text", None)
            if t:
                chunks.append(t)
        output = "\n".join(chunks).strip()

        decision, reasoning = _extract_label(output)
        return AgentResult(decision=decision, reasoning=reasoning)




def safe_format(template: str, **kwargs) -> str:
    """Fail fast on prompt brace issues."""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise KeyError(
            f"Prompt formatting KeyError: {e}. "
            f"Likely you have unescaped braces '{{...}}' in prompts.py. "
            f"Fix by replacing {{...}} with [...], or escape as '{{{{...}}}}'."
        ) from e


def run_llm(inst, llm, mode: str):
    p = DEFAULT_PROMPTS

    if mode == "Direct":
        user_prompt = safe_format(p.direct, evidence_text=inst.evidence_text())
    elif mode == "CoT":
        user_prompt = safe_format(p.cot, evidence_text=inst.evidence_text())
    elif mode == "ToT":
        user_prompt = safe_format(p.tot, evidence_text=inst.evidence_text())
    else:
        raise ValueError(mode)

    return llm.predict(
        system_prompt=p.system,
        user_prompt=user_prompt,
        method=mode,
    )



def _etype(e) -> str:
    return str(getattr(e, "etype", getattr(e, "type", "")) or "").strip().lower()


def _etext(e) -> str:
    return str(getattr(e, "text", "") or "").strip().lower()


def force_apply_perturbation(inst, pname: str):
    """
    Safety net: enforce the intended semantic perturbation if apply_perturbation
    is brittle or misses a case.
    """
    try:
        inst2 = inst.model_copy(deep=True)
    except Exception:
        inst2 = inst.copy(deep=True)

    if not hasattr(inst2, "evidence") or inst2.evidence is None:
        return inst2

    ev = list(inst2.evidence)
    p = pname.strip().lower()

    if p == "remove_c2":
        inst2.evidence = [
            x
            for x in ev
            if _etype(x) not in {"c2", "command_and_control", "command-control", "command and control"}
            and "beacon" not in _etext(x)
            and "command" not in _etext(x)
            and "c2" not in _etext(x)
        ]

    elif p == "remove_credential_access":
        inst2.evidence = [
            x
            for x in ev
            if _etype(x) not in {"credential_access", "credential", "creds", "credentials"}
            and "credential" not in _etext(x)
            and "lsass" not in _etext(x)
            and "password" not in _etext(x)
            and "cookie" not in _etext(x)
        ]

    elif p == "remove_encryption_and_coercion":
        inst2.evidence = [
            x
            for x in ev
            if _etype(x) not in {"encryption", "ransom", "coercion"}
            and "encrypt" not in _etext(x)
            and "encryption" not in _etext(x)
            and "ransom" not in _etext(x)
            and "bitcoin" not in _etext(x)
            and "payment" not in _etext(x)
            and "decrypt" not in _etext(x)
        ]

    elif p == "remove_persistence":
        inst2.evidence = [x for x in ev if _etype(x) != "persistence" and "persistence" not in _etext(x)]

    # modify_c2 / mask_encryption / inject_c2_backdoor left to apply_perturbation
    return inst2



def css_from_cf(root_decision: str, cf_decisions: list[str]) -> float:
    """CSS = fraction of CF decisions equal to root decision (after normalization)."""
    if not cf_decisions:
        return 1.0
    r = normalize_label(root_decision)
    if not r:
        return 0.0
    same = sum(1 for d in cf_decisions if normalize_label(d) == r)
    return same / len(cf_decisions)


def flip_rate_from_cf(root_decision: str, cf_decisions: list[str]) -> float:
    if not cf_decisions:
        return 0.0
    r = normalize_label(root_decision)
    if not r:
        return 1.0
    diff = sum(1 for d in cf_decisions if normalize_label(d) != r)
    return diff / len(cf_decisions)


def run_baseline_with_counterfactuals(inst, llm, mode: str):
    """Run Direct/CoT/ToT on original + each CF; compute CSS/Flip + CF alignment."""
    root = run_llm(inst, llm, mode)

    cfs = []
    cf_decisions = []
    detail_rows = []

    for pname in inst.counterfactual_labels.keys():
        inst_cf = apply_perturbation(inst, pname)
        inst_cf = force_apply_perturbation(inst_cf, pname)
        r_cf = run_llm(inst_cf, llm, mode)

        exp_norm = normalize_label(inst.counterfactual_labels[pname])
        pred_root = normalize_label(root.decision)
        pred_cf = normalize_label(r_cf.decision)

        flipped = 1 if (pred_cf != pred_root) else 0
        correct = 1 if (pred_cf == exp_norm) else 0

        cfs.append((pname, r_cf))
        cf_decisions.append(r_cf.decision)

        detail_rows.append(
            {
                "sample_id": inst.sample_id,
                "category": inst.category,
                "method": mode,
                "perturbation": pname,
                "expected_cf": exp_norm,
                "pred_root": pred_root,
                "pred_cf": pred_cf,
                "flipped": flipped,
                "correct": correct,
            }
        )

    css = css_from_cf(root.decision, cf_decisions)
    flip_rate = flip_rate_from_cf(root.decision, cf_decisions)

    cf_pairs = [(p, normalize_label(r.decision)) for p, r in cfs]
    acc_orig, acc_cf = alignment_accuracy(inst, normalize_label(root.decision), cf_pairs)

    return root, cfs, css, flip_rate, acc_orig, acc_cf, detail_rows


def run_cva(inst, llm):
    """
    CVA: root (CoT) + CF runs (CoT).
    """
    root = run_llm(inst, llm, "CoT")

    cfs = []
    detail_rows = []

    for pname in inst.counterfactual_labels.keys():
        inst_cf = apply_perturbation(inst, pname)
        inst_cf = force_apply_perturbation(inst_cf, pname)
        r_cf = run_llm(inst_cf, llm, "CoT")
        cfs.append((pname, r_cf))

        exp = normalize_label(inst.counterfactual_labels[pname])
        pred_root = normalize_label(root.decision)
        pred_cf = normalize_label(r_cf.decision)

        detail_rows.append(
            {
                "sample_id": inst.sample_id,
                "category": inst.category,
                "method": "CVA",
                "perturbation": pname,
                "expected_cf": exp,
                "pred_root": pred_root,
                "pred_cf": pred_cf,
                "flipped": 1 if (pred_cf != pred_root) else 0,
                "correct": 1 if (pred_cf == exp) else 0,
            }
        )

    g_cot = {
        "root": {"sample_id": inst.sample_id, "decision": root.decision},
        "counterfactuals": [{"perturbation": p, "decision": r.decision} for p, r in cfs],
    }
    return root, cfs, g_cot, detail_rows


#

def main():
    load_dotenv()

    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    instances = load_jsonl(cfg["dataset_path"])

    model_name = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
    model_tag = model_name.replace("/", "_").replace(":", "_")
    llm = ClaudeChatLLM(model=model_name, temperature=float(os.getenv("CLAUDE_TEMPERATURE", "0.0")))

    methods = ["Direct", "CoT", "ToT", "CVA"]
    results = {m: [] for m in methods}

    per_pert_rows: list[dict] = []

    for inst in instances:
        print(f"\n[bold cyan]=== {inst.sample_id} ({inst.category}) ===[/bold cyan]")

        # Baselines WITH CF-eval
        r_direct, _, css_direct, flip_direct, acc_o_direct, acc_cf_direct, rows_direct = run_baseline_with_counterfactuals(
            inst, llm, "Direct"
        )
        r_cot, _, css_cot, flip_cot, acc_o_cot, acc_cf_cot, rows_cot = run_baseline_with_counterfactuals(
            inst, llm, "CoT"
        )
        r_tot, _, css_tot, flip_tot, acc_o_tot, acc_cf_tot, rows_tot = run_baseline_with_counterfactuals(
            inst, llm, "ToT"
        )

        per_pert_rows.extend(rows_direct)
        per_pert_rows.extend(rows_cot)
        per_pert_rows.extend(rows_tot)

        # CVA
        root, cfs, _, rows_cva = run_cva(inst, llm)
        per_pert_rows.extend(rows_cva)

        cva_css = css_from_cf(root.decision, [r.decision for _, r in cfs])
        cva_flip = 1.0 - cva_css

        cva_cf_decisions = [(p, normalize_label(rr.decision)) for p, rr in cfs]
        acc_o_cva, acc_cf_cva = alignment_accuracy(inst, normalize_label(root.decision), cva_cf_decisions)

        # ATD (root only)
        atd_direct = compute_atd(r_direct.reasoning, "Direct")
        atd_cot = compute_atd(r_cot.reasoning, "CoT")
        atd_tot = compute_atd(r_tot.reasoning, "ToT")
        atd_cva = compute_atd("", "CVA", num_nodes=1 + len(cfs))

        # DSI
        dsi_direct = dsi_for_instance(inst, normalize_label(r_direct.decision), r_direct.reasoning, css=css_direct)
        dsi_cot = dsi_for_instance(inst, normalize_label(r_cot.decision), r_cot.reasoning, css=css_cot)
        dsi_tot = dsi_for_instance(inst, normalize_label(r_tot.decision), r_tot.reasoning, css=css_tot)
        dsi_cva = dsi_for_instance(inst, normalize_label(root.decision), root.reasoning, css=cva_css)

        results["Direct"].append(
            {
                "sample_id": inst.sample_id,
                "decision": normalize_label(r_direct.decision),
                "dsi": dsi_direct,
                "atd": atd_direct,
                "orig_acc": acc_o_direct,
                "css": css_direct,
                "cf_acc": acc_cf_direct,
                "flip": flip_direct,
            }
        )
        results["CoT"].append(
            {
                "sample_id": inst.sample_id,
                "decision": normalize_label(r_cot.decision),
                "dsi": dsi_cot,
                "atd": atd_cot,
                "orig_acc": acc_o_cot,
                "css": css_cot,
                "cf_acc": acc_cf_cot,
                "flip": flip_cot,
            }
        )
        results["ToT"].append(
            {
                "sample_id": inst.sample_id,
                "decision": normalize_label(r_tot.decision),
                "dsi": dsi_tot,
                "atd": atd_tot,
                "orig_acc": acc_o_tot,
                "css": css_tot,
                "cf_acc": acc_cf_tot,
                "flip": flip_tot,
            }
        )
        results["CVA"].append(
            {
                "sample_id": inst.sample_id,
                "decision": normalize_label(root.decision),
                "dsi": dsi_cva,
                "atd": atd_cva,
                "orig_acc": acc_o_cva,
                "css": cva_css,
                "cf_acc": acc_cf_cva,
                "flip": cva_flip,
            }
        )

        print(f"Direct: {normalize_label(r_direct.decision)} | CSS={css_direct:.2f} | Flip={flip_direct:.2f} | CF_Acc={acc_cf_direct:.2f}")
        print(f"CoT:    {normalize_label(r_cot.decision)}    | CSS={css_cot:.2f}    | Flip={flip_cot:.2f}    | CF_Acc={acc_cf_cot:.2f}")
        print(f"ToT:    {normalize_label(r_tot.decision)}    | CSS={css_tot:.2f}    | Flip={flip_tot:.2f}    | CF_Acc={acc_cf_tot:.2f}")
        print(f"CVA:    {normalize_label(root.decision)}     | CSS={cva_css:.2f}    | Flip={cva_flip:.2f}    | CF_Acc={acc_cf_cva:.2f}")

    # Aggregate metrics
    avgs = aggregate_metrics(instances, results)

    t = Table(title=f"Step 4 (Claude) — Summary (Baselines + CVA with Counterfactual Eval) [{model_name}]")
    t.add_column("Method")
    t.add_column("Avg CSS", justify="right")
    t.add_column("Avg Flip", justify="right")
    t.add_column("Avg DSI", justify="right")
    t.add_column("Avg ATD", justify="right")
    t.add_column("Orig Acc", justify="right")
    t.add_column("CF Acc", justify="right")

    for m in methods:
        avg_flip = sum(x["flip"] for x in results[m]) / max(1, len(results[m]))
        t.add_row(
            m,
            f"{avgs[m].avg_css:.2f}" if avgs[m].avg_css is not None else "—",
            f"{avg_flip:.2f}",
            f"{avgs[m].avg_dsi:.2f}",
            f"{avgs[m].avg_atd:.2f}",
            f"{avgs[m].orig_acc:.2f}",
            f"{avgs[m].cf_acc:.2f}" if avgs[m].cf_acc is not None else "—",
        )
    print(t)

    # Save main CSV (model-specific)
    out_path = OUT_DIR / f"table_metrics_claude_{model_tag}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Avg_CSS", "Avg_Flip", "Avg_DSI", "Avg_ATD", "Orig_Acc", "CF_Acc"])
        for m in methods:
            avg_flip = sum(x["flip"] for x in results[m]) / max(1, len(results[m]))
            w.writerow(
                [
                    m,
                    f"{avgs[m].avg_css:.4f}" if avgs[m].avg_css is not None else "",
                    f"{avg_flip:.4f}",
                    f"{avgs[m].avg_dsi:.4f}",
                    f"{avgs[m].avg_atd:.4f}",
                    f"{avgs[m].orig_acc:.4f}",
                    f"{avgs[m].cf_acc:.4f}" if avgs[m].cf_acc is not None else "",
                ]
            )
    print(f"[bold green]Saved CSV:[/bold green] {out_path}")

    # Save per-perturbation audit CSV (model-specific)
    per_path = OUT_DIR / f"per_perturbation_claude_{model_tag}.csv"
    with per_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "category",
                "method",
                "perturbation",
                "expected_cf",
                "pred_root",
                "pred_cf",
                "flipped",
                "correct",
            ],
        )
        w.writeheader()
        w.writerows(per_pert_rows)
    print(f"[bold green]Saved per-perturbation audit:[/bold green] {per_path}")

    # Print per-perturbation summary
    agg = defaultdict(lambda: {"n": 0, "flip": 0, "correct": 0})
    for r in per_pert_rows:
        k = (r["method"], r["perturbation"])
        agg[k]["n"] += 1
        agg[k]["flip"] += int(r["flipped"])
        agg[k]["correct"] += int(r["correct"])

    pt = Table(title="Per-Perturbation Summary (Flip & CF Correctness)")
    pt.add_column("Method")
    pt.add_column("Perturbation")
    pt.add_column("N", justify="right")
    pt.add_column("FlipRate", justify="right")
    pt.add_column("CF_Acc", justify="right")

    for (m, p), v in sorted(agg.items()):
        n = v["n"]
        pt.add_row(
            m,
            p,
            str(n),
            f"{(v['flip'] / n):.2f}" if n else "—",
            f"{(v['correct'] / n):.2f}" if n else "—",
        )
    print(pt)


if __name__ == "__main__":
    main()
