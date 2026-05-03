#!/usr/bin/env python
"""
Analyze and compare benchmark results from multiple test runs.

This script:
1. Load multiple benchmark result JSON files
2. Compare metrics across runs
3. Generate summary statistics
4. Optionally visualize differences
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_results(file_path: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_all_results(results_dir: str) -> list:
    """Load all benchmark results from a directory."""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' not found")
        return results
    
    for json_file in sorted(results_path.glob("benchmarks_*.json")):
        try:
            data = load_results(str(json_file))
            data["_filename"] = json_file.name
            data["_timestamp"] = json_file.stem.split("_")[1:]  # Extract timestamp
            results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results


def calculate_stats(values: list) -> dict:
    """Calculate statistics for a list of values."""
    if not values:
        return {
            "mean": 0,
            "min": 0,
            "max": 0,
            "std": 0
        }
    
    n = len(values)
    mean = sum(values) / n
    min_val = min(values)
    max_val = max(values)
    
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / n
        std = variance ** 0.5
    else:
        std = 0
    
    return {
        "mean": mean,
        "min": min_val,
        "max": max_val,
        "std": std,
        "count": n
    }


def extract_prefill_metrics(results: list) -> dict:
    """Extract prefill metrics from all runs."""
    metrics = {
        "prompt_tokens": [],
        "prefill_time": [],
        "tokens_per_second": []
    }
    
    for result in results:
        for run in result.get("prefill", {}).get("results", []):
            stats = run.get("statistics", {})
            metrics["prompt_tokens"].append(run.get("prompt_tokens", 0))
            metrics["prefill_time"].append(stats.get("mean_time_seconds", 0))
            metrics["tokens_per_second"].append(stats.get("mean_tokens_per_second", 0))
    
    return metrics


def extract_output_metrics(results: list) -> dict:
    """Extract output generation metrics from all runs."""
    metrics = {
        "prompt_tokens": [],
        "ttft": [],
        "throughput": [],
        "total_time": []
    }
    
    for result in results:
        for run in result.get("output", {}).get("results", []):
            stats = run.get("statistics", {})
            metrics["prompt_tokens"].append(run.get("prompt_tokens", 0))
            metrics["ttft"].append(stats.get("mean_ttft_seconds", 0))
            metrics["throughput"].append(stats.get("mean_throughput", 0))
            metrics["total_time"].append(stats.get("mean_total_time_seconds", 0))
    
    return metrics


def print_comparison_table(name: str, data: dict):
    """Print a comparison table for metrics."""
    print(f"\n{name}")
    print("-" * 70)
    
    for metric, values in data.items():
        stats = calculate_stats(values)
        print(f"  {metric:20} mean: {stats['mean']:>10.3f}  min: {stats['min']:>10.3f}  max: {stats['max']:>10.3f}  (n={stats['count']})")


def print_detailed_comparison(results: list):
    """Print detailed comparison across all runs."""
    if not results:
        print("No results to compare")
        return
    
    print("\n" + "=" * 70)
    print("  DETAILED COMPARISON")
    print("=" * 70)
    
    for i, result in enumerate(results):
        filename = result.get("_filename", f"Run {i+1}")
        model = result.get("model", {})
        if isinstance(model, dict):
            model_label = model.get("model_alias", "unknown")
        else:
            model_label = str(model)

        print(f"\n--- {filename} (Model: {model_label}) ---")
        
        # Prefill metrics
        prefill_stats = result.get("prefill", {}).get("results", [])
        if prefill_stats:
            print("\n  Prefill Performance:")
            for run in prefill_stats:
                tokens = run.get("prompt_tokens", 0)
                speed = run.get("statistics", {}).get("mean_tokens_per_second", 0)
                print(f"    {tokens:>6} tokens: {speed:>8.2f} tok/s")
        
        # Output metrics
        output_stats = result.get("output", {}).get("results", [])
        if output_stats:
            print("\n  Output Generation:")
            for run in output_stats:
                ttft = run.get("statistics", {}).get("mean_ttft_seconds", 0)
                throughput = run.get("statistics", {}).get("mean_throughput", 0)
                generated = run.get("statistics", {}).get("mean_generated_tokens", 0)
                print(f"    TTFT: {ttft:.3f}s | Throughput: {throughput:.2f} tok/s | Generated: {generated:.0f} tokens")


def generate_summary(results: list):
    """Generate and print summary statistics."""
    if not results:
        print("No results to summarize")
        return
    
    print("\n" + "=" * 70)
    print("  SUMMARY STATISTICS")
    print("=" * 70)
    
    # Extract metrics
    prefill = extract_prefill_metrics(results)
    output = extract_output_metrics(results)
    
    print("\n--- Input (Prefill) Performance ---")
    print_comparison_table("Token Count", {"prompt_tokens": prefill["prompt_tokens"]})
    print_comparison_table("Prefill Time", {"time_seconds": prefill["prefill_time"]})
    print_comparison_table("Tokens/Second", {"speed": prefill["tokens_per_second"]})
    
    print("\n--- Output Generation Performance ---")
    print_comparison_table("TTFT (s)", {"ttft": output["ttft"]})
    print_comparison_table("Throughput", {"tokens_per_second": output["throughput"]})
    print_comparison_table("Total Time", {"time_seconds": output["total_time"]})
    
    # Calculate aggregated stats
    mean_speed = calculate_stats(prefill["tokens_per_second"])
    mean_ttft = calculate_stats(output["ttft"])
    
    print("\n--- Overall Summary ---")
    print(f"  Mean prefill speed: {mean_speed['mean']:.2f} tok/s (±{mean_speed['std']:.2f})")
    print(f"  Mean TTFT: {mean_ttft['mean']:.3f}s (±{mean_ttft['std']:.3f})")
    print(f"  Total runs analyzed: {len(results)}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze and compare llama.cpp benchmark results"
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results",
        help="Directory containing benchmark result JSON files (default: results)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed comparison for each run"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare specific files instead of directory"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        results = []
        for file_path in args.compare:
            if os.path.exists(file_path):
                data = load_results(file_path)
                data["_filename"] = os.path.basename(file_path)
                results.append(data)
            else:
                print(f"Warning: File not found: {file_path}")
    else:
        results = load_all_results(args.results_dir)
    
    if not results:
        print("No benchmark results found")
        print(f"Looked in: {args.results_dir}")
        sys.exit(1)
    
    print(f"\nLoaded {len(results)} benchmark result(s)")
    
    if args.detailed:
        print_detailed_comparison(results)
    
    generate_summary(results)
    
    print("\n")


if __name__ == "__main__":
    main()
