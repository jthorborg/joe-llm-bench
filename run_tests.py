#!/usr/bin/env python
"""
Run all benchmarks for llama.cpp server performance evaluation.

This script:
1. Loads configuration from config.yaml
2. Runs input (pre-fill) benchmarks for 5k and 50k token inputs
3. Runs output generation benchmarks
4. Prints results to terminal and saves to JSON file
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from benchmarker import (
    LlamaBenchmarker,
    load_file_content,
    load_conversation_context,
    save_results
)


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_header(text):
    """Print a formatted header."""
    print_separator()
    print(f"  {text}")
    print_separator()


def print_result(name, value, unit=""):
    """Print a single result metric."""
    if isinstance(value, float):
        print(f"  {name:40} {value:>10.3f} {unit}")
    else:
        print(f"  {name:40} {value}")


def run_prefill_benchmarks(benchmarker, config):
    """Run all prefill benchmarks."""
    print_header("INPUT (PREFILL) BENCHMARKS")
    
    results = {
        "test_type": "prefill",
        "timestamp": datetime.now().isoformat(),
        "model": config["server"]["model"],
        "config": config["test_params"],
        "results": []
    }
    
    text_dir = Path(config["test_data"]["text_dir"])
    code_dir = Path(config["test_data"]["code_dir"])
    temperature = config["test_params"]["temperature"]
    iterations = config["test_params"]["iterations"]
    
    test_files = []
    
    # Add text files
    for target in config["test_params"]["input_targets"]:
        text_path = text_dir / f"text_{target}.txt"
        conv_path = text_dir / f"conversation_{target}.txt"
        
        if conv_path.exists():
            test_files.append({
                "name": f"Text {target}k tokens (with context)",
                "path": str(conv_path),
                "target_tokens": target
            })
        elif text_path.exists():
            test_files.append({
                "name": f"Text {target}k tokens (raw)",
                "path": str(text_path),
                "target_tokens": target
            })
    
    # Add code files
    for target in config["test_params"]["input_targets"]:
        code_path = code_dir / f"code_{target}.py"
        if code_path.exists():
            test_files.append({
                "name": f"Code {target}k tokens",
                "path": str(code_path),
                "target_tokens": target
            })
    
    print(f"\nRunning {len(test_files)} input benchmarks ({iterations} iterations each)...\n")
    
    for test_file in test_files:
        print(f"Testing: {test_file['name']}")
        
        # Load content
        try:
            if "conversation" in test_file["name"]:
                prompt = load_conversation_context(test_file["path"])
            else:
                prompt = load_file_content(test_file["path"])
        except Exception as e:
            print(f"  Error loading file: {e}")
            continue
        
        # Run benchmark
        test_result = benchmarker.run_prefill_benchmark(
            prompt,
            iterations=iterations,
            temperature=temperature
        )
        
        results["results"].append(test_result)
        
        # Print results
        stats = test_result["statistics"]
        print(f"  Prompt tokens: {test_result['prompt_tokens']}")
        print(f"  Iterations: {stats['successful_iterations']}/{stats['iterations']}")
        print_result("Mean prefill time", stats["mean_time_seconds"], "s")
        print_result("Mean tokens/sec", stats["mean_tokens_per_second"], "tokens/s")
        print_result("Min tokens/sec", stats["min_tokens_per_second"], "tokens/s")
        print_result("Max tokens/sec", stats["max_tokens_per_second"], "tokens/s")
        print()
    
    return results


def run_output_benchmarks(benchmarker, config):
    """Run output generation benchmarks."""
    print_header("OUTPUT GENERATION BENCHMARKS")
    
    results = {
        "test_type": "output",
        "timestamp": datetime.now().isoformat(),
        "model": config["server"]["model"],
        "config": config["test_params"],
        "results": []
    }
    
    text_dir = Path(config["test_data"]["text_dir"])
    temperature = config["test_params"]["temperature"]
    stop = config["test_params"]["output_stop"]
    iterations = config["test_params"]["iterations"]
    
    # Use medium-sized input for output tests (~1k tokens)
    medium_input = text_dir / "text_5000.txt"
    if not medium_input.exists():
        print("Warning: 5k text file not found, using sample prompt")
        prompt = "The quick brown fox jumps over the lazy dog. "
        prompt += "This is a sample prompt for testing output generation. " * 20
    else:
        prompt = load_file_content(str(medium_input))
    
    print(f"\nRunning output benchmarks with ~1k token input...")
    print(f"Using stop sequence: '{stop}'")
    print(f"Iterations: {iterations}\n")
    
    test_result = benchmarker.run_output_benchmark(
        prompt,
        iterations=iterations,
        temperature=temperature,
        stop=stop
    )
    
    results["results"].append(test_result)
    
    # Print results
    stats = test_result["statistics"]
    print(f"  Prompt tokens: {test_result['prompt_tokens']}")
    print(f"  Iterations: {stats['successful_iterations']}/{stats['iterations']}")
    print_result("Mean TTFT", stats["mean_ttft_seconds"], "s")
    print_result("Mean throughput", stats["mean_throughput"], "tokens/s")
    print_result("Min throughput", stats["min_throughput"], "tokens/s")
    print_result("Max throughput", stats["max_throughput"], "tokens/s")
    print_result("Mean total time", stats["mean_total_time_seconds"], "s")
    print_result("Mean generated tokens", stats["mean_generated_tokens"], "tokens")
    print()
    
    return results


def print_summary(all_results):
    """Print summary of all benchmarks."""
    print_header("SUMMARY")
    
    print("\n--- Input (Prefill) Performance ---")
    for result in all_results.get("prefill", {}).get("results", []):
        stats = result["statistics"]
        print(f"  {result['prompt_tokens']} tokens: {stats['mean_tokens_per_second']:.2f} tok/s")
    
    print("\n--- Output Generation Performance ---")
    for result in all_results.get("output", {}).get("results", []):
        stats = result["statistics"]
        print(f"  TTFT: {stats['mean_ttft_seconds']:.3f}s | ", end="")
        print(f"Throughput: {stats['mean_throughput']:.2f} tok/s")
    
    print()


def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("  LLAMA.CPP PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)
    
    # Load configuration
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nConfiguration loaded from: {config_path}")
    print(f"Server URL: {config['server']['url']}")
    print(f"Model: {config['server']['model']}")
    print(f"Test iterations: {config['test_params']['iterations']}")
    
    # Initialize benchmarker
    benchmarker = LlamaBenchmarker(
        server_url=config["server"]["url"],
        model_name=config["server"]["model"]
    )
    
    # Test server connectivity
    print("\nTesting server connectivity...")
    try:
        response = benchmarker.send_completion_request(
            "test",
            n_predict=1,
            stop=["."]
        )
        if "error" in response:
            print(f"Warning: Server returned error: {response['error']}")
        else:
            print("Server connection: OK")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure llama.cpp server is running on", config["server"]["url"])
        sys.exit(1)
    
    # Run benchmarks
    print()
    prefill_results = run_prefill_benchmarks(benchmarker, config)
    output_results = run_output_benchmarks(benchmarker, config)
    
    # Combine results
    all_results = {
        "prefill": prefill_results,
        "output": output_results
    }
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    if config["output"]["save_json"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(config["output"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = results_dir / f"benchmarks_{timestamp}.json"
        save_results(all_results, str(output_path))
    
    print("\nBenchmark suite complete!\n")


if __name__ == "__main__":
    main()
