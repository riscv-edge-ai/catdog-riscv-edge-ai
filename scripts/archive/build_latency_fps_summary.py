#!/usr/bin/env python3
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_CSV = RESULTS_DIR / "benchmark_500.csv"
PER_IMAGE_CSV = RESULTS_DIR / "benchmark_500_latency_fps.csv"
SUMMARY_CSV = RESULTS_DIR / "latency_fps_summary.csv"
CPU_HZ = 25_000_000


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not BENCHMARK_CSV.exists():
        raise SystemExit(f"Missing benchmark CSV: {BENCHMARK_CSV}")

    with BENCHMARK_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"No rows found in {BENCHMARK_CSV}")

    per_image_rows = []
    accel_total = 0
    sw_total = 0

    for row in rows:
        accel_cycles = int(row["accel_cycles"])
        sw_cycles = int(row["sw_cycles"])
        accel_total += accel_cycles
        sw_total += sw_cycles
        per_image_rows.append(
            {
                "image_index": row["image_index"],
                "prediction_accel": row["prediction_accel"],
                "prediction_sw": row["prediction_sw"],
                "expected_label": row["expected_label"],
                "accel_cycles": accel_cycles,
                "sw_cycles": sw_cycles,
                "accel_latency_ms": f"{(accel_cycles / CPU_HZ) * 1000.0:.6f}",
                "sw_latency_ms": f"{(sw_cycles / CPU_HZ) * 1000.0:.6f}",
                "accel_fps": f"{(CPU_HZ / accel_cycles):.6f}",
                "sw_fps": f"{(CPU_HZ / sw_cycles):.6f}",
            }
        )

    count = len(per_image_rows)
    avg_accel_cycles = accel_total / count
    avg_sw_cycles = sw_total / count

    summary_rows = [
        {
            "path": "accelerator",
            "row_count": count,
            "avg_cycles": f"{avg_accel_cycles:.3f}",
            "avg_latency_ms": f"{(avg_accel_cycles / CPU_HZ) * 1000.0:.6f}",
            "avg_fps": f"{(CPU_HZ / avg_accel_cycles):.6f}",
        },
        {
            "path": "software",
            "row_count": count,
            "avg_cycles": f"{avg_sw_cycles:.3f}",
            "avg_latency_ms": f"{(avg_sw_cycles / CPU_HZ) * 1000.0:.6f}",
            "avg_fps": f"{(CPU_HZ / avg_sw_cycles):.6f}",
        },
    ]

    write_csv(
        PER_IMAGE_CSV,
        per_image_rows,
        [
            "image_index",
            "prediction_accel",
            "prediction_sw",
            "expected_label",
            "accel_cycles",
            "sw_cycles",
            "accel_latency_ms",
            "sw_latency_ms",
            "accel_fps",
            "sw_fps",
        ],
    )
    write_csv(
        SUMMARY_CSV,
        summary_rows,
        ["path", "row_count", "avg_cycles", "avg_latency_ms", "avg_fps"],
    )

    print(f"Wrote {PER_IMAGE_CSV}")
    print(f"Wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
