"""
Core benchmarker for testing llama.cpp inference performance.

Tests:
1. Input (pre-fill) processing speed - measures time to process prompts
2. Output generation speed - measures TTFT (time to first token) and throughput
"""

import json
import time
import requests
from typing import Dict, List, Any, Tuple
from pathlib import Path


class LlamaBenchmarker:
    """Benchmark llama.cpp server performance."""
    
    def __init__(self, server_url: str, model_name: str = "default"):
        """
        Initialize the benchmarker.
        
        Args:
            server_url: URL of the llama.cpp server (e.g., http://localhost:8080)
            model_name: Name of the model to test
        """
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.completion_endpoint = f"{self.server_url}/completion"
    
    def fetch_model_info(self) -> Dict[str, str]:
        """Fetch model alias and path from the server's /props endpoint."""
        try:
            response = requests.get(f"{self.server_url}/props", timeout=10)
            response.raise_for_status()
            props = response.json()
            return {
                "model_alias": props.get("model_alias", "unknown"),
                "model_path": props.get("model_path", "unknown"),
            }
        except Exception as e:
            return {"model_alias": self.model_name, "model_path": "unknown", "error": str(e)}

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken (cl100k_base encoding)."""
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: estimate ~4 chars per token
            return len(text) // 4
    
    def send_completion_request(
        self,
        prompt: str,
        n_predict: int = None,
        temperature: float = 0.7,
        stop: str = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a completion request to the llama.cpp server.
        
        Args:
            prompt: The input prompt
            n_predict: Maximum number of tokens to generate (None for dynamic)
            temperature: Sampling temperature
            stop: Stop sequence(s) - can be string or list of strings
            stream: Whether to use streaming
        
        Returns:
            Response JSON as dictionary
        """
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "stream": stream
        }
        
        if n_predict is not None:
            payload["n_predict"] = n_predict
        
        if stop is not None:
            if isinstance(stop, list):
                payload["stop"] = stop
            else:
                payload["stop"] = [stop]
        
        try:
            response = requests.post(
                self.completion_endpoint,
                json=payload,
                timeout=300  # 5 minute timeout for large inputs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error sending request: {e}")
            return {"error": str(e)}
    
    def measure_prefill(self, prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Measure input (pre-fill) processing speed.
        
        Args:
            prompt: The input prompt to process
        
        Returns:
            Dictionary with timing metrics
        """
        # Count tokens
        prompt_tokens = self.count_tokens(prompt)
        
        # Measure time
        start_time = time.time()
        response = self.send_completion_request(
            prompt,
            n_predict=1,  # Just need to trigger prefill
            temperature=temperature,
            stop=["."]
        )
        elapsed = time.time() - start_time
        
        # Calculate metrics
        if "error" not in response:
            prefill_time   = response.get("timings", {}).get("prompt_ms", elapsed * 1000) / 1000
            prefill_tokens = response.get("timings", {}).get("prompt_n", prompt_tokens)

            # Detect KV cache hit: server only processed a tiny fraction of the prompt
            # (typically reports prompt_n=1 when the full prompt was already cached)
            if prefill_tokens < max(2, prompt_tokens * 0.1):
                return {
                    "status": "cache_hit",
                    "prompt_tokens": prefill_tokens,
                    "prefill_time_seconds": prefill_time,
                    "tokens_per_second": 0,
                    "error": f"KV cache hit — server reused prior result (reported {prefill_tokens} tokens processed)"
                }

            tokens_per_second = prefill_tokens / prefill_time if prefill_time > 0 else 0

            return {
                "status": "success",
                "prompt_tokens": prefill_tokens,
                "prefill_time_seconds": prefill_time,
                "tokens_per_second": tokens_per_second,
                "error": None
            }
        else:
            return {
                "status": "error",
                "prompt_tokens": prompt_tokens,
                "prefill_time_seconds": elapsed,
                "tokens_per_second": 0,
                "error": response["error"]
            }
    
    def measure_output_generation(
        self,
        prompt: str,
        temperature: float = 0.7,
        stop: str = "### END",
        n_predict: int = 200
    ) -> Dict[str, Any]:
        """
        Measure output generation speed (TTFT and throughput).
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            stop: Stop sequence to terminate generation
        
        Returns:
            Dictionary with timing metrics
        """
        # Count prompt tokens
        prompt_tokens = self.count_tokens(prompt)
        
        # Measure time
        start_time = time.time()
        first_token_time = None
        
        response = self.send_completion_request(
            prompt,
            n_predict=n_predict,
            temperature=temperature,
            stop=stop
        )
        elapsed = time.time() - start_time
        
        if "error" in response:
            return {
                "status": "error",
                "prompt_tokens": prompt_tokens,
                "generated_tokens": 0,
                "time_to_first_token_seconds": None,
                "total_time_seconds": elapsed,
                "tokens_per_second": 0,
                "error": response["error"]
            }
        
        # Extract timing info
        timings = response.get("timings", {})
        
        # Get generated tokens
        generated_text = response.get("content", response.get("text", ""))
        generated_tokens = self.count_tokens(generated_text)
        
        # Get timing info if available
        if "prompt_ms" in timings:
            prefill_time = timings["prompt_ms"] / 1000
        else:
            prefill_time = None
        
        if "predicted_ms" in timings:
            decode_time = timings["predicted_ms"] / 1000
        else:
            decode_time = None
        
        # Calculate metrics
        if prefill_time is not None and decode_time is not None:
            time_to_first_token = prefill_time  # TTFT is essentially prefill time
            total_generation_time = prefill_time + decode_time
            tokens_per_second = generated_tokens / decode_time if decode_time > 0 else 0
        else:
            # Fallback: estimate from total time
            # This is approximate since we don't have TTFT specifically
            time_to_first_token = elapsed * 0.3  # Rough estimate (30% for prefill)
            decode_time = elapsed - time_to_first_token
            tokens_per_second = generated_tokens / decode_time if decode_time > 0 else 0
        
        return {
            "status": "success",
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "time_to_first_token_seconds": time_to_first_token,
            "total_time_seconds": elapsed,
            "tokens_per_second": tokens_per_second,
            "error": None
        }
    
    def run_prefill_benchmark(
        self,
        prompt: str,
        iterations: int = 3,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        results = []

        for i in range(iterations):
            print(f"    iteration {i+1}/{iterations}...", end=" ", flush=True)
            t0 = time.time()
            result = self.measure_prefill(prompt, temperature)
            elapsed = time.time() - t0
            result["iteration"] = i + 1
            results.append(result)
            if result["status"] == "success":
                status = f"{result['tokens_per_second']:.1f} tok/s"
            elif result["status"] == "cache_hit":
                status = "cache hit (skipped)"
            else:
                status = result["error"]
            print(f"done ({elapsed:.1f}s) — {status}", flush=True)

        # Statistics exclude cache hits
        valid = [r for r in results if r["status"] == "success"]
        times = [r["prefill_time_seconds"] for r in valid]
        tokens_per_sec = [r["tokens_per_second"] for r in valid]
        cache_hits = sum(1 for r in results if r["status"] == "cache_hit")

        if times:
            stats = {
                "mean_time_seconds": sum(times) / len(times),
                "min_time_seconds": min(times),
                "max_time_seconds": max(times),
                "mean_tokens_per_second": sum(tokens_per_sec) / len(tokens_per_sec),
                "min_tokens_per_second": min(tokens_per_sec),
                "max_tokens_per_second": max(tokens_per_sec),
                "iterations": len(results),
                "successful_iterations": len(valid),
                "cache_hits": cache_hits
            }
        else:
            stats = {
                "mean_time_seconds": 0,
                "min_time_seconds": 0,
                "max_time_seconds": 0,
                "mean_tokens_per_second": 0,
                "min_tokens_per_second": 0,
                "max_tokens_per_second": 0,
                "iterations": len(results),
                "successful_iterations": 0,
                "cache_hits": cache_hits
            }
        
        return {
            "test_type": "prefill",
            "prompt_tokens": results[0].get("prompt_tokens", 0),
            "iterations": iterations,
            "temperature": temperature,
            "results": results,
            "statistics": stats
        }
    
    def run_output_benchmark(
        self,
        prompt: str,
        iterations: int = 3,
        temperature: float = 0.7,
        stop: str = "### END",
        n_predict: int = 200
    ) -> Dict[str, Any]:
        results = []

        for i in range(iterations):
            print(f"    iteration {i+1}/{iterations}...", end=" ", flush=True)
            t0 = time.time()
            result = self.measure_output_generation(prompt, temperature, stop, n_predict)
            elapsed = time.time() - t0
            result["iteration"] = i + 1
            results.append(result)
            status = f"{result['tokens_per_second']:.1f} tok/s ({result['generated_tokens']} tokens)" if result["status"] == "success" else result["error"]
            print(f"done ({elapsed:.1f}s) — {status}", flush=True)
        
        # Calculate statistics
        ttft_times = [r["time_to_first_token_seconds"] for r in results if r["status"] == "success"]
        throughput = [r["tokens_per_second"] for r in results if r["status"] == "success"]
        total_times = [r["total_time_seconds"] for r in results if r["status"] == "success"]
        generated_tokens = [r["generated_tokens"] for r in results if r["status"] == "success"]
        
        if ttft_times:
            stats = {
                "mean_ttft_seconds": sum(ttft_times) / len(ttft_times),
                "min_ttft_seconds": min(ttft_times),
                "max_ttft_seconds": max(ttft_times),
                "mean_throughput": sum(throughput) / len(throughput),
                "min_throughput": min(throughput),
                "max_throughput": max(throughput),
                "mean_total_time_seconds": sum(total_times) / len(total_times),
                "mean_generated_tokens": sum(generated_tokens) / len(generated_tokens),
                "iterations": len(results),
                "successful_iterations": sum(1 for r in results if r["status"] == "success")
            }
        else:
            stats = {
                "mean_ttft_seconds": 0,
                "min_ttft_seconds": 0,
                "max_ttft_seconds": 0,
                "mean_throughput": 0,
                "min_throughput": 0,
                "max_throughput": 0,
                "mean_total_time_seconds": 0,
                "mean_generated_tokens": 0,
                "iterations": len(results),
                "successful_iterations": 0
            }
        
        return {
            "test_type": "output",
            "prompt_tokens": results[0].get("prompt_tokens", 0),
            "iterations": iterations,
            "temperature": temperature,
            "stop_sequence": stop,
            "results": results,
            "statistics": stats
        }


def load_file_content(file_path: str) -> str:
    """Load content from a file."""
    with open(file_path, 'r') as f:
        return f.read()


def load_conversation_context(file_path: str) -> str:
    """Load conversation context from file."""
    return load_file_content(file_path)


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save benchmark results to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")
