from __future__ import annotations
import yaml
from rich import print
from rich.table import Table
from pathlib import Path


from cycb.io import load_jsonl
from cycb.stats import dataset_stats
from cycb.perturbations import PERTURBATIONS, apply_perturbation


def main() -> None:
    cfg_path = Path("configs/default.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    dataset_path = cfg["dataset_path"]

    instances = load_jsonl(dataset_path)

    print(f"[bold green]Loaded[/bold green] {len(instances)} instances from {dataset_path}")
    print(f"Available perturbations: {list(PERTURBATIONS.keys())}")

    # Validate perturbations apply cleanly
    for inst in instances:
        for pname in inst.counterfactual_labels.keys():
            _ = apply_perturbation(inst, pname)

    print("[bold green]Perturbation application check passed[/bold green] ")

    # Build Table 1-like stats
    stats = dataset_stats(instances)

    t = Table(title="CyCB (Toy) — Summary Stats (Table 1 scaffold)")
    t.add_column("Category")
    t.add_column("Instances", justify="right")
    t.add_column("Avg. Evidence Items", justify="right")
    t.add_column("Avg. Counterfactuals", justify="right")

    total_instances = 0
    total_e = 0.0
    total_cf = 0.0

    for cat, s in stats.items():
        t.add_row(cat, str(s.instances), f"{s.avg_evidence_items:.2f}", f"{s.avg_counterfactuals:.2f}")
        total_instances += s.instances
        total_e += s.avg_evidence_items * s.instances
        total_cf += s.avg_counterfactuals * s.instances

    # total row
    t.add_row(
        "Total",
        str(total_instances),
        f"{(total_e/total_instances):.2f}",
        f"{(total_cf/total_instances):.2f}",
    )

    print(t)

if __name__ == "__main__":
    main()
