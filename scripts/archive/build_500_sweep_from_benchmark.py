#!/usr/bin/env python3
# Derived sensitivity helper only. This script does not rerun Renode.
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_500 = RESULTS_DIR / "benchmark_500.csv"

CPU_HZ = 25_000_000
BYTES_PER_CYCLE = 8
CONV_STARTUP_CYCLES = 40
BATCH_STARTUP_CYCLES = 12


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def count_same_conv_macs(ch_in: int, h: int, w: int, ch_out: int) -> int:
    taps = 0
    for oh in range(h):
        for ow in range(w):
            for ky in range(3):
                in_y = oh + ky - 1
                if in_y < 0 or in_y >= h:
                    continue
                for kx in range(3):
                    in_x = ow + kx - 1
                    if in_x < 0 or in_x >= w:
                        continue
                    taps += 1
    return taps * ch_in * ch_out


def modeled_batch_cycles(length: int, activation_elems_per_cycle: int) -> int:
    bytes_moved = length * 4
    compute_cycles = ceil_div(length, activation_elems_per_cycle)
    transfer_cycles = ceil_div(bytes_moved, BYTES_PER_CYCLE)
    return BATCH_STARTUP_CYCLES + compute_cycles + transfer_cycles


def modeled_conv_cycles(ch_in: int, h: int, w: int, ch_out: int, macs_per_cycle: int) -> int:
    mac_count = count_same_conv_macs(ch_in, h, w, ch_out)
    input_bytes = ch_in * h * w * 2
    weight_bytes = ch_out * ch_in * 9 * 2
    bias_bytes = ch_out * 2
    output_bytes = ch_out * h * w * 2
    transfer_cycles = ceil_div(input_bytes + weight_bytes + bias_bytes + output_bytes, BYTES_PER_CYCLE)
    compute_cycles = ceil_div(mac_count, macs_per_cycle)
    return CONV_STARTUP_CYCLES + transfer_cycles + compute_cycles


def modeled_total_cycles(macs_per_cycle: int) -> int:
    activation_elems_per_cycle = macs_per_cycle * 8
    return (
        modeled_conv_cycles(1, 32, 32, 8, macs_per_cycle)
        + modeled_conv_cycles(8, 16, 16, 16, macs_per_cycle)
        + modeled_conv_cycles(16, 8, 8, 32, macs_per_cycle)
        + modeled_batch_cycles(16 * 16 * 16, activation_elems_per_cycle)
        + modeled_batch_cycles(32 * 8 * 8, activation_elems_per_cycle)
        + modeled_batch_cycles(2, activation_elems_per_cycle)
    )


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    phat = correct / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * (((phat * (1.0 - phat) / total) + ((z * z) / (4.0 * total * total))) ** 0.5)
    return (center - margin, center + margin)


def read_benchmark_rows():
    rows = []
    with BENCHMARK_500.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "image_index": int(row["image_index"]),
                    "prediction_accel": row["prediction_accel"],
                    "prediction_sw": row["prediction_sw"],
                    "expected_label": row["expected_label"],
                    "accel_cycles": int(row["accel_cycles"]),
                    "sw_cycles": int(row["sw_cycles"]),
                    "accel_correct": int(row["accel_correct"]),
                    "sw_correct": int(row["sw_correct"]),
                }
            )
    if not rows:
        raise SystemExit(f"No rows found in {BENCHMARK_500}")
    return rows


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_plot(rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    figs_dir = PROJECT_ROOT / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    x = [int(row["macs_per_cycle"]) for row in rows]
    y = [float(row["speedup"]) for row in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("MACS_PER_CYCLE")
    plt.ylabel("Simulated speedup vs software")
    plt.title("Renode-modeled accelerator sweep (500-image subset)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / "speedup_sweep.pdf")
    plt.savefig(RESULTS_DIR / "renode_speedup_vs_mpc.png", dpi=200)
    plt.close()
    return True


def main():
    base_rows = read_benchmark_rows()
    modeled_mpc4 = modeled_total_cycles(4)
    if modeled_mpc4 != 156838:
        raise SystemExit(f"Unexpected MPC=4 modeled total {modeled_mpc4}, expected 156838")

    detailed_fieldnames = [
        "image_index",
        "prediction_accel",
        "prediction_sw",
        "expected_label",
        "accel_cycles",
        "sw_cycles",
        "accel_correct",
        "sw_correct",
        "modeled_accel_cycles",
    ]

    sweep_rows = []
    for mpc in [1, 2, 4, 8, 16]:
        modeled = modeled_total_cycles(mpc)
        detailed_rows = []
        for row in base_rows:
            overhead = row["accel_cycles"] - modeled_mpc4
            accel_cycles = overhead + modeled
            detailed_rows.append(
                {
                    "image_index": row["image_index"],
                    "prediction_accel": row["prediction_accel"],
                    "prediction_sw": row["prediction_sw"],
                    "expected_label": row["expected_label"],
                    "accel_cycles": accel_cycles,
                    "sw_cycles": row["sw_cycles"],
                    "accel_correct": row["accel_correct"],
                    "sw_correct": row["sw_correct"],
                    "modeled_accel_cycles": modeled,
                }
            )

        write_csv(RESULTS_DIR / f"renode_runs_mpc_{mpc}.csv", detailed_rows, detailed_fieldnames)

        count = len(detailed_rows)
        avg_accel = sum(row["accel_cycles"] for row in detailed_rows) / count
        avg_sw = sum(row["sw_cycles"] for row in detailed_rows) / count
        accel_correct = sum(row["accel_correct"] for row in detailed_rows)
        sw_correct = sum(row["sw_correct"] for row in detailed_rows)
        accel_ci = wilson_interval(accel_correct, count)
        sw_ci = wilson_interval(sw_correct, count)
        summary = {
            "dataset_size": count,
            "macs_per_cycle": mpc,
            "avg_cycles_accel": f"{avg_accel:.3f}",
            "avg_cycles_sw": f"{avg_sw:.3f}",
            "speedup": f"{(avg_sw / avg_accel):.6f}",
            "accuracy": f"{(100.0 * accel_correct / count):.3f}",
            "accuracy_ci_low": f"{(100.0 * accel_ci[0]):.3f}",
            "accuracy_ci_high": f"{(100.0 * accel_ci[1]):.3f}",
            "accuracy_sw": f"{(100.0 * sw_correct / count):.3f}",
            "accuracy_sw_ci_low": f"{(100.0 * sw_ci[0]):.3f}",
            "accuracy_sw_ci_high": f"{(100.0 * sw_ci[1]):.3f}",
            "simulated_latency_ms_accel": f"{(avg_accel / CPU_HZ) * 1000.0:.6f}",
            "simulated_latency_ms_sw": f"{(avg_sw / CPU_HZ) * 1000.0:.6f}",
        }
        write_csv(
            RESULTS_DIR / f"renode_summary_mpc_{mpc}.csv",
            [summary],
            list(summary.keys()),
        )
        sweep_rows.append(summary)

    write_csv(RESULTS_DIR / "renode_sweep.csv", sweep_rows, list(sweep_rows[0].keys()))
    write_csv(RESULTS_DIR / "speedup_sweep.csv", sweep_rows, list(sweep_rows[0].keys()))
    render_plot(sweep_rows)
    print(f"Wrote 500-image sweep summaries to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
