# Graph-of-Counterfactual-Thought (G-CoT)

This repository contains the reference implementation for **Graph-of-Counterfactual-Thought (G-CoT)**,
a structured reasoning framework for **counterfactually-verifiable LLM agents** in cybersecurity decision-making.

The code accompanies the paper:

> **Counterfactually-Verifiable LLM Agents for Safe and Auditable Cybersecurity Decision-Making**

## Overview

Large Language Model (LLM) agents are increasingly deployed in high-stakes domains such as malware analysis.
However, existing reasoning approaches (e.g., Chain-of-Thought, Tree-of-Thought) fail to expose **decision fragility**
under plausible evidence perturbations.

**G-CoT** addresses this by:
- Structuring reasoning as a **graph**
- Explicitly modeling **counterfactual evidence removals**
- Enabling **verifiable stability, safety, and auditability**

<p align="center">
  <img src="T1.png" width="900">
</p>


## Core Concepts

- **G-CoT (Graph-of-Counterfactual-Thought)**  
  Nodes represent reasoning states; edges represent counterfactual perturbations.

- **CVA (Counterfactually-Verifiable Agents)**  
  Agents that must justify decisions under structured counterfactual stress tests.

- **CyCB Benchmark**  
  A cybersecurity benchmark designed to evaluate reasoning stability under evidence perturbation.


## Code Structure

- `src/cycb/agents.py`  
  Implements Direct, CoT, and CVA agents.

- `src/cycb/perturbations.py`  
  Counterfactual evidence removal and perturbation logic.

- `src/cycb/metrics.py`  
  Stability, sensitivity, and auditability metrics (CSS, DSI, ATD).

- `scripts/`  
  End-to-end evaluation pipelines for different LLM backends.


## Dataset Used in Experiments

The main experiments are conducted on:

`cycb_synth.jsonl`

This dataset contains:
- 100 samples
- Structured evidence for malware analysis
- Ground-truth labels and counterfactual labels

It is designed as a controlled benchmark for evaluating counterfactual reasoning and auditability.

Larger dataset variants (e.g., `cycb_v1`, `cycb_v50`) are provided for scalability experiments and future work.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (.venv) (base) abbasyazdinejad@Abbass-MacBook-Pro cva-cycb % python ./scripts/step4_run_openai_eval.py

# python -m scripts.step4_run_openai_eval

# python -m scripts.step0_generate_synth_cycb --n-per-class 25 --seed 42

export OPENAI_MODEL="gpt-4o"
python -m scripts.step4_run_openai_eval
# results/table_metrics_openai_gpt-4o.csv
# results/per_perturbation_openai_gpt-4o.csv


export OPENAI_MODEL="gpt-4o-mini"
python -m scripts.step4_run_openai_eval
# results/table_metrics_openai_gpt-4o-mini.csv
# results/per_perturbation_openai_gpt-4o-mini.csv


export OPENAI_MODEL="gpt-4.1"
python -m scripts.step4_run_openai_eval
# results/table_metrics_openai_gpt-4.1.csv
# results/per_perturbation_openai_gpt-4.1.csv
