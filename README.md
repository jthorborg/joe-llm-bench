# llama.cpp Performance Benchmark Suite

This project provides a comprehensive benchmark suite for evaluating llama.cpp server performance.

## Quick Start

1. **Start llama.cpp server** (ensure it's running on `http://localhost:8001`):
   ```bash
   ./server -m models/your-model.gguf -c 4096 --port 8001
   ```
   > Note: On Windows, use `server.exe` instead of `./server`

2. **Generate test data** (creates files with ~5k and ~50k tokens):
   ```bash
   python generate_test_data.py
   ```

3. **Run benchmarks**:
   ```bash
   python run_tests.py
   ```

4. **Analyze results**:
   ```bash
   python analyze_results.py
   ```

## Configuration

Edit `config.yaml` to customize:

```yaml
server:
  url: http://localhost:8001
  model: default

test_params:
  input_targets: [5000, 50000]    # Token counts for input tests
  output_stop: "### END"          # Stop sequence for output
  iterations: 3                   # Runs per test
  temperature: 0.7                # Sampling temp

test_data:
  text_dir: "test_data/text"
  code_dir: "test_data/code"

output:
  results_dir: "results"
  save_json: true
```

## Test Structure

### Input (Prefill) Benchmarks
- **5k tokens**: Simulates reading a single file
- **50k tokens**: Simulates session resume with context
- Includes conversation context (system prompt + user message)
- Measures: tokens/second during prefill

### Output Benchmarks
- Fixed ~1k token input prompt
- Generates until `### END` stop sequence
- Measures: TTFT (time to first token) + throughput

## Output

Results are saved to `results/benchmarks_<timestamp>.json` and displayed in terminal:

```
=== SUMMARY ===

--- Input (Prefill) Performance ---
  5000 tokens: 125.43 tok/s
  50000 tokens: 89.21 tok/s

--- Output Generation Performance ---
  TTFT: 0.234s | Throughput: 45.67 tok/s
```

## Analyzing Multiple Runs

Run benchmarks multiple times and compare:

```bash
python run_tests.py  # Run 1
python run_tests.py  # Run 2
python analyze_results.py --detailed
```

This loads all `results/benchmarks_*.json` files and shows comparisons.

## Files

| File | Description |
|------|-------------|
| `generate_test_data.py` | Creates test files with controlled token counts |
| `benchmarker.py` | Core testing logic (pre-fill + output timing) |
| `run_tests.py` | Executes all benchmarks, prints results |
| `analyze_results.py` | Compares multiple test runs |
| `config.yaml` | Configuration file |

## Requirements

- Python 3.10+
- `requests` library
- `tiktoken` library
- `pyyaml` library

Install: `pip install requests tiktoken pyyaml`

## Customization

### Adjust Token Counts
Edit `config.yaml` → `test_params.input_targets`

### Change Stop Sequence
Edit `config.yaml` → `test_params.output_stop`

### Add More Iterations
Edit `config.yaml` → `test_params.iterations`

## Notes

- TTFT (Time To First Token) includes prompt processing time
- Results vary based on hardware, model size, and context length
- Use `--detailed` flag to see individual run comparisons
