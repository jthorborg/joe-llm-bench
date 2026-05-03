#!/usr/bin/env python
"""
Compare benchmark results across models and runs.

Usage:
  python compare_runs.py                        # all files in results/
  python compare_runs.py results/a.json b.json  # specific files
  python compare_runs.py --dir other_dir
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_results(results_dir: str = "results", files: list = None) -> list:
    runs = []
    if files:
        paths = [Path(f) for f in files]
    else:
        paths = sorted(Path(results_dir).glob("benchmarks_*.json"))

    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
            data["_file"] = path.name
            runs.append(data)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")

    return runs


def model_label(model_field) -> str:
    if isinstance(model_field, dict):
        return model_field.get("model_alias", "unknown")
    return str(model_field) if model_field else "unknown"


def run_key(run: dict) -> str:
    """Column grouping key: model alias + optional run label."""
    model_field = run.get("prefill", {}).get("model") or run.get("output", {}).get("model")
    alias = model_label(model_field)
    label = run.get("prefill", {}).get("run_label") or run.get("output", {}).get("run_label")
    return f"{alias} [{label}]" if label else alias


def extract_prefill(run: dict) -> list[dict]:
    """Return list of {label, prompt_tokens, tok_s} for each prefill test."""
    results = run.get("prefill", {}).get("results", [])
    rows = []
    for result in results:
        content_type = result.get("content_type", "text")
        target = result.get("target_tokens", result.get("prompt_tokens", "?"))
        label = f"{content_type}~{target}"
        tok_s = result.get("statistics", {}).get("mean_tokens_per_second", 0)
        prompt_tokens = result.get("prompt_tokens", 0)
        rows.append({"label": label, "prompt_tokens": prompt_tokens, "tok_s": tok_s})
    return rows


def extract_output(run: dict) -> list[dict]:
    """Return list of {content_type, throughput, ttft, generated_tokens}."""
    results = run.get("output", {}).get("results", [])
    rows = []
    for result in results:
        stats = result.get("statistics", {})
        rows.append({
            "content_type": result.get("content_type", "text"),
            "throughput": stats.get("mean_throughput", 0),
            "ttft": stats.get("mean_ttft_seconds", 0),
            "generated_tokens": stats.get("mean_generated_tokens", 0),
        })
    return rows


def mean(values):
    return sum(values) / len(values) if values else 0


def std(values):
    if len(values) < 2:
        return 0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def fmt(value, decimals=1):
    return f"{value:.{decimals}f}"


def fmt_stat(values, decimals=1):
    if not values:
        return "  -  "
    if len(values) == 1:
        return fmt(values[0], decimals)
    return f"{fmt(mean(values), decimals)} +-{fmt(std(values), decimals)}"


def compare(runs: list):
    # Group runs by model + run_label
    by_key: dict[str, list] = defaultdict(list)
    for run in runs:
        by_key[run_key(run)].append(run)
    run_keys = sorted(by_key.keys())

    # Collect prefill labels in order, deduped
    seen_prefill: dict[str, int] = {}
    for run in runs:
        for r in extract_prefill(run):
            if r["label"] not in seen_prefill:
                seen_prefill[r["label"]] = r["prompt_tokens"]
    prefill_labels = list(seen_prefill.keys())

    # Collect output content types in order, deduped
    output_types: list[str] = []
    seen_out: set[str] = set()
    for run in runs:
        for o in extract_output(run):
            if o["content_type"] not in seen_out:
                seen_out.add(o["content_type"])
                output_types.append(o["content_type"])

    label_w = max(40, max((len(k) for k in run_keys), default=0) + 2)
    col_w   = 12

    ct_abbr = {"text": "txt", "code": "cod"}

    def abbr(ct):
        return ct_abbr.get(ct, ct[:3])

    def hline(n_cols):
        print("+" + "-" * label_w + ("+" + "-" * col_w) * n_cols + "+")

    def trow(first, cells, n_cols):
        truncated = (first[:label_w - 3] + ">") if len(first) > label_w - 2 else first
        line = f"| {truncated:<{label_w - 2}} "
        for cell in cells:
            line += f"| {cell:^{col_w - 2}} "
        # pad missing cells
        for _ in range(n_cols - len(cells)):
            line += f"| {'':^{col_w - 2}} "
        print(line + "|")

    width = label_w + 2 + (col_w + 1) * max(len(prefill_labels), len(output_types) * 2, 1)
    print()
    print("=" * width)
    print("  BENCHMARK COMPARISON")
    print(f"  {len(runs)} run(s)  |  {len(run_keys)} group(s)")
    print("=" * width)

    # ── Prefill table ──────────────────────────────────────────────────────────
    n_pc = len(prefill_labels)
    print("\n  PREFILL  (tok/s)\n")
    hline(n_pc)
    trow("Run", [lbl.replace("text~", "t~").replace("code~", "c~") for lbl in prefill_labels], n_pc)
    trow("",    [f"[{seen_prefill[lbl]}]" for lbl in prefill_labels], n_pc)
    hline(n_pc)
    for key in run_keys:
        cells = []
        for lbl in prefill_labels:
            vals = [r["tok_s"] for run in by_key[key] for r in extract_prefill(run)
                    if r["label"] == lbl and r["tok_s"] > 0]
            cells.append(fmt_stat(vals))
        trow(key, cells, n_pc)
    hline(n_pc)

    # ── Output table ───────────────────────────────────────────────────────────
    if output_types:
        n_oc = len(output_types) * 2
        print("\n  OUTPUT GENERATION  (tok/s ^ | TTFT s v)\n")
        hline(n_oc)
        trow("Run",
             [h for ct in output_types for h in (f"[{abbr(ct)}] thr", f"[{abbr(ct)}] TTFT")],
             n_oc)
        hline(n_oc)
        for key in run_keys:
            cells = []
            for ct in output_types:
                thr  = [o["throughput"] for run in by_key[key] for o in extract_output(run)
                        if o["content_type"] == ct and o.get("throughput", 0) > 0]
                ttft = [o["ttft"]       for run in by_key[key] for o in extract_output(run)
                        if o["content_type"] == ct and o.get("ttft", 0) > 0]
                cells.append(fmt_stat(thr))
                cells.append(fmt_stat(ttft, decimals=3))
            trow(key, cells, n_oc)
        hline(n_oc)

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare llama.cpp benchmark runs")
    parser.add_argument("files", nargs="*", help="Specific JSON files to compare")
    parser.add_argument("--dir", default="results", help="Results directory (default: results)")
    args = parser.parse_args()

    runs = load_results(results_dir=args.dir, files=args.files if args.files else None)
    if not runs:
        print("No benchmark results found.")
        sys.exit(1)

    compare(runs)


if __name__ == "__main__":
    main()
