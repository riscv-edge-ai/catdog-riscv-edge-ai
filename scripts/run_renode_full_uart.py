#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "scripts" / "benchmark_renode.py"
RENODE_UART = Path("/tmp/ai_classification_renode_uart_output.txt")
RESULTS_DIR = PROJECT_ROOT / "results"
FIGS_DIR = PROJECT_ROOT / "figs"
CANONICAL_RESULT_PATTERNS = [
    "chunked/renode_direct_uart_offset_*_count_*_mpc_4.txt",
    "chunked/renode_direct_uart_offset_*_count_*_mpc_4.txt.json",
    "benchmark_500.csv",
    "benchmark_500_manifest.json",
    "table1_accuracy_500.csv",
    "float_reference_accuracy.csv",
    "benchmark_500_latency_fps.csv",
    "latency_fps_summary.csv",
    "eval_dataset_manifest.csv",
    "renode_runs_mpc_*.csv",
    "renode_summary_mpc_*.csv",
    "renode_sweep.csv",
    "speedup_sweep.csv",
    "renode_speedup_vs_mpc.png",
]
CANONICAL_FIGURE_PATTERNS = [
    "speedup_sweep.pdf",
]


def remove_existing_canonical_outputs():
    for pattern in CANONICAL_RESULT_PATTERNS:
        for path in RESULTS_DIR.glob(pattern):
            if path.is_file():
                path.unlink()
    for pattern in CANONICAL_FIGURE_PATTERNS:
        for path in FIGS_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Run one large Renode-modeled benchmark point with full UART output preserved as a run-specific log."
    )
    parser.add_argument("--dataset-count", type=int, default=500)
    parser.add_argument("--dataset-offset", type=int, default=0)
    parser.add_argument("--macs-per-cycle", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--save-name", type=str, default="")
    args = parser.parse_args()

    if args.dataset_offset == 0:
        remove_existing_canonical_outputs()

    cmd = [
        sys.executable,
        str(BENCHMARK),
        "--dataset-count",
        str(args.dataset_count),
        "--dataset-offset",
        str(args.dataset_offset),
        "--macs-per-cycle",
        str(args.macs_per_cycle),
        "--timeout",
        str(args.timeout),
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    if not RENODE_UART.exists():
        raise SystemExit(f"Expected UART log not found: {RENODE_UART}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_name:
        target = RESULTS_DIR / args.save_name
    else:
        target = RESULTS_DIR / (
            f"uart_full_offset_{args.dataset_offset}_count_{args.dataset_count}_mpc_{args.macs_per_cycle}.txt"
        )
    # This script preserves the raw log from the latest run. It does not decide
    # whether that run should become the repository's authoritative packaged log.
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RENODE_UART, target)
    metadata = {
        "uart_log": str(target.relative_to(PROJECT_ROOT)),
        "dataset_count": args.dataset_count,
        "dataset_offset": args.dataset_offset,
        "macs_per_cycle": args.macs_per_cycle,
    }
    metadata_path = target.with_suffix(target.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved UART log to {target}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
