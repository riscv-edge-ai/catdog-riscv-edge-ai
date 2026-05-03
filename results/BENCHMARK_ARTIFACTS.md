# Benchmark Artifact Inventory

Canonical packaged benchmark set:

- Chunk UART logs: `results/chunked/renode_direct_uart_offset_*_count_*_mpc_4.txt`
- Chunk metadata: `results/chunked/renode_direct_uart_offset_*_count_*_mpc_4.txt.json`
- Combined CSV: `results/benchmark_500.csv`
- Source manifest: `results/benchmark_500_manifest.json`

Why chunked:

- The monolithic 500-image Renode run is not stable enough in this environment; `/usr/bin/renode` is killed by the OS before completion.
- The canonical 500-row benchmark is therefore rebuilt from three preserved real reruns:
  - offset `0`, count `200`
  - offset `200`, count `200`
  - offset `400`, count `100`

Notes:

- `scripts/build_chunked_submission_artifacts.py` packages the chunked UART logs into `results/benchmark_500.csv`.
- `scripts/verify_benchmark_consistency.py` verifies that `results/benchmark_500.csv` matches the UART logs listed in `results/benchmark_500_manifest.json`.
- `scripts/build_500_sweep_from_benchmark.py` is a derived sensitivity script. It is not canonical reproduced benchmark evidence.
