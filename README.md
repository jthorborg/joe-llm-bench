# llama.cpp Performance Benchmark Suite

This project provides a comprehensive benchmark suite for evaluating llama.cpp server performance.
It is entirely vibecoded, so use at your own risk, but should be harmless and works well.

## Quick Start

1. **Start llama.cpp server** (ensure it's running on `http://localhost:8001`):
   ```bash
   ./server -m models/your-model.gguf -c 4096 --port 8001
   ```
   > Note: On Windows, use `server.exe` instead of `./server`

2. **Generate test data** (creates files with ~5k and ~50k tokens, depending on config):
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

5. **Compare runs**:
   ```bash
   python compare_runs.py
   ```
   Example output:

```
===================================================================================================================================================
  BENCHMARK COMPARISON
  26 run(s)  |  21 group(s)
===================================================================================================================================================

  PREFILL  (tok/s)

+-------------------------------------------------------------------+------------+------------+------------+------------+------------+------------+
| Run                                                               |   t~100    |   t~1000   |  t~15000   |   c~100    |   c~1000   |  c~15000   |
|                                                                   |   [261]    |   [1146]   |  [15188]   |    [99]    |   [998]    |  [14630]   |
+-------------------------------------------------------------------+------------+------------+------------+------------+------------+------------+
| gemma/moe-4 [no-experts-gpu-layers-12]                            |   318.3    |   597.5    |   696.4    |   167.0    |   509.8    |   611.5    |
| gemma/moe-4 [rev1]                                                |   273.6    |   503.4    |   609.2    |   146.4    |   432.6    |   544.9    |
| llama/3.2-instruct [all-gpu]                                      |   4135.9   |   7298.4   |   6344.9   |   4548.1   |   7714.4   |   6477.6   |
| qwen/coder-next-3 [no-experts-gpu-layers-12]                      |    75.2    |   149.4    |   266.5    |    51.3    |   189.8    |   221.7    |
| qwen/coder-next-3 [no-experts-gpu-layers-8]                       |    80.1    |   145.5    |   254.1    |    47.5    |   172.0    |   200.4    |
| qwen/coder-next-3 [rev1]                                          | 88.3 +-0.9 | 152.2 +-0.1 | 275.1 +-3.0 | 55.5 +-1.3 | 178.7 +-2.3 | 205.8 +-2.1 |
| qwen/coder-next-3 [rev2-16k-context]                              |    95.8    |   154.5    |   273.1    |    56.3    |   179.0    |   205.6    |
| qwen/coder-next-3 [rev2-kv-quant-disabled-no-flash-no-kvu-no-fit] | 86.8 +-0.1 | 152.3 +-1.1 | 269.5 +-1.5 | 54.2 +-0.8 | 177.1 +-0.5 | 202.5 +-1.3 |
| qwen/coder-next-3 [rev2-kv-quant-disabled-no-flash-no-kvu]        |    87.2    |   149.1    |   272.4    |    54.8    |   175.7    |   204.5    |
| qwen/coder-next-3 [rev2-kv-quant-disabled-no-flash]               |    87.2    |   151.2    |   269.8    |    54.5    |   174.3    |   199.9    |
| qwen/coder-next-3 [rev2-kv-quant-disabled]                        |    84.4    |   148.8    |   264.8    |    54.0    |   176.6    |   202.7    |
| qwen/coder-next-3 [rev2-overclocked-gpu]                          |    84.4    |   149.3    |   268.7    |    54.5    |   176.2    |   203.9    |
| qwen/coder-next-3 [threads-11]                                    |    86.0    |   155.5    |   277.2    |    56.2    |   180.8    |   209.4    |
| qwen/coder-next-3 [threads-12-affinity-1c7]                       |    88.6    |   155.2    |   278.0    |    55.5    |   180.9    |   209.5    |
| qwen/coder-next-3 [threads-12-affinity-FFF]                       |    90.3    |   156.0    |   278.5    |    56.3    |   183.0    |   209.4    |
| qwen/coder-next-3 [threads-4]                                     | 88.5 +-1.8 | 151.7 +-5.2 | 273.4 +-6.0 | 55.7 +-0.9 | 178.1 +-5.0 | 205.5 +-3.7 |
| qwen/coder-next-3 [threads-5]                                     |    87.4    |   152.3    |   269.4    |    53.2    |   174.0    |   201.7    |
| qwen/coder-next-3 [threads-6-affinity-15015]                      |    89.0    |   153.9    |   277.0    |    56.5    |   180.6    |   208.1    |
| qwen/coder-next-3 [threads-6-affinity-555]                        |    86.8    |   151.0    |   269.2    |    52.3    |   176.0    |   201.8    |
| qwen/coder-next-3 [threads-6]                                     | 88.6 +-2.2 | 154.4 +-2.2 | 274.8 +-2.4 | 54.2 +-2.1 | 179.0 +-4.2 | 204.8 +-4.4 |
| qwen/coder-next-3 [threads-8]                                     | 87.9 +-2.1 | 151.2 +-5.4 | 272.7 +-5.9 | 56.2 +-0.7 | 180.1 +-2.7 | 206.1 +-3.7 |
+-------------------------------------------------------------------+------------+------------+------------+------------+------------+------------+

  OUTPUT GENERATION  (tok/s ^ | TTFT s v)

+-------------------------------------------------------------------+------------+------------+------------+------------+
| Run                                                               | [txt] thr  | [txt] TTFT | [cod] thr  | [cod] TTFT |
+-------------------------------------------------------------------+------------+------------+------------+------------+
| gemma/moe-4 [no-experts-gpu-layers-12]                            |    14.5    |   0.494    |    15.4    |   0.461    |
| gemma/moe-4 [rev1]                                                |    26.2    |   0.541    |    23.7    |   0.462    |
| llama/3.2-instruct [all-gpu]                                      |   172.2    |   0.019    |   165.6    |   0.018    |
| qwen/coder-next-3 [no-experts-gpu-layers-12]                      |    10.8    |   1.168    |    10.2    |   1.063    |
| qwen/coder-next-3 [no-experts-gpu-layers-8]                       |    9.7     |   1.348    |    9.4     |   1.220    |
| qwen/coder-next-3 [rev1]                                          | 23.3 +-1.0 | 1.111 +-0.007 | 23.0 +-0.9 | 0.953 +-0.009 |
| qwen/coder-next-3 [rev2-16k-context]                              |    27.2    |   1.066    |    27.2    |   0.946    |
| qwen/coder-next-3 [rev2-kv-quant-disabled-no-flash-no-kvu-no-fit] | 27.1 +-1.0 | 1.094 +-0.030 | 26.4 +-1.2 | 0.967 +-0.043 |
| qwen/coder-next-3 [rev2-kv-quant-disabled-no-flash-no-kvu]        |    25.9    |   1.073    |    25.0    |   0.960    |
| qwen/coder-next-3 [rev2-kv-quant-disabled-no-flash]               |    26.6    |   1.166    |    26.3    |   0.997    |
| qwen/coder-next-3 [rev2-kv-quant-disabled]                        |    26.6    |   1.135    |    26.2    |   0.975    |
| qwen/coder-next-3 [rev2-overclocked-gpu]                          |    25.4    |   1.141    |    24.9    |   1.019    |
| qwen/coder-next-3 [threads-11]                                    |    26.7    |   1.111    |    26.8    |   0.942    |
| qwen/coder-next-3 [threads-12-affinity-1c7]                       |    15.0    |   1.115    |    21.0    |   0.970    |
| qwen/coder-next-3 [threads-12-affinity-FFF]                       |    26.1    |   1.065    |    25.8    |   0.910    |
| qwen/coder-next-3 [threads-4]                                     | 27.0 +-0.8 | 1.091 +-0.012 | 26.9 +-0.7 | 0.944 +-0.010 |
| qwen/coder-next-3 [threads-5]                                     |    26.0    |   1.156    |    25.9    |   0.977    |
| qwen/coder-next-3 [threads-6-affinity-15015]                      |    26.4    |   1.086    |    26.2    |   0.933    |
| qwen/coder-next-3 [threads-6-affinity-555]                        |    26.9    |   1.109    |    26.3    |   0.935    |
| qwen/coder-next-3 [threads-6]                                     | 25.9 +-1.4 | 1.150 +-0.071 | 25.8 +-0.9 | 0.997 +-0.068 |
| qwen/coder-next-3 [threads-8]                                     | 26.1 +-0.9 | 1.102 +-0.038 | 26.4 +-1.1 | 0.939 +-0.028 |
+-------------------------------------------------------------------+------------+------------+------------+------------+
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
