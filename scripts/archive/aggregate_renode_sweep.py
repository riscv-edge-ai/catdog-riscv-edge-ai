#!/usr/bin/env python3
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SWEEP_PATH = RESULTS_DIR / "renode_sweep.csv"
PLOT_PATH = RESULTS_DIR / "renode_speedup_vs_mpc.png"


def render_plot(rows, output_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x = [int(row["macs_per_cycle"]) for row in rows]
    y = [float(row["speedup"]) for row in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("MACS_PER_CYCLE")
    plt.ylabel("Simulated speedup vs software")
    plt.title("Renode-modeled accelerator sweep")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def main():
    rows = []
    for path in sorted(RESULTS_DIR.glob("renode_summary_mpc_*.csv")):
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)

    if not rows:
        raise SystemExit("No renode_summary_mpc_*.csv files found in results/")

    rows.sort(key=lambda row: int(row["macs_per_cycle"]))
    fieldnames = list(rows[0].keys())
    with SWEEP_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_ok = render_plot(rows, PLOT_PATH)
    print(f"Wrote {SWEEP_PATH}")
    if plot_ok:
        print(f"Wrote {PLOT_PATH}")
    else:
        print("Skipped plot: matplotlib unavailable")


if __name__ == "__main__":
    main()
