#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_UART = RESULTS_DIR / "renode_direct_uart_authoritative.txt"
DEFAULT_CSV = RESULTS_DIR / "benchmark_500.csv"
DEFAULT_MANIFEST = RESULTS_DIR / "benchmark_500_manifest.json"

ACCEL_RE = re.compile(r"\[ACCEL_SIM\] Cycles:\s*(\d+)")
SW_RE = re.compile(r"\[SW_ONLY\]\s*Cycles:\s*(\d+)")
IMAGE_RE = re.compile(r"=== Image\s+(\d+)")
SUMMARY_IMAGES_RE = re.compile(r"Images:\s*(\d+)")
SUMMARY_ACCEL_RE = re.compile(r"Average accelerator cycles:\s*(\d+)")
SUMMARY_SW_RE = re.compile(r"Average software cycles:\s*(\d+)")


def parse_uart(path: Path):
    rows = []
    summary = {}
    current = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip("\r")
        image_match = IMAGE_RE.match(line)
        if image_match:
            current = {
                "image_index": int(image_match.group(1)),
                "accel_cycles": None,
                "sw_cycles": None,
            }
            rows.append(current)
            continue
        if current is None:
            images_match = SUMMARY_IMAGES_RE.search(line)
            if images_match:
                summary["images"] = int(images_match.group(1))
            accel_match = SUMMARY_ACCEL_RE.search(line)
            if accel_match:
                summary["avg_accel_cycles"] = int(accel_match.group(1))
            sw_match = SUMMARY_SW_RE.search(line)
            if sw_match:
                summary["avg_sw_cycles"] = int(sw_match.group(1))
            continue

        accel_match = ACCEL_RE.search(line)
        if accel_match:
            current["accel_cycles"] = int(accel_match.group(1))
            continue
        sw_match = SW_RE.search(line)
        if sw_match:
            current["sw_cycles"] = int(sw_match.group(1))
            continue

        images_match = SUMMARY_IMAGES_RE.search(line)
        if images_match:
            summary["images"] = int(images_match.group(1))
        accel_match = SUMMARY_ACCEL_RE.search(line)
        if accel_match:
            summary["avg_accel_cycles"] = int(accel_match.group(1))
        sw_match = SUMMARY_SW_RE.search(line)
        if sw_match:
            summary["avg_sw_cycles"] = int(sw_match.group(1))

    return rows, summary


def parse_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "image_index": int(row["image_index"]),
                    "accel_cycles": int(row["accel_cycles"]),
                    "sw_cycles": int(row["sw_cycles"]),
                }
            )
    return rows


def averages(rows):
    if not rows:
        raise ValueError("No rows found")
    return {
        "row_count": len(rows),
        "avg_accel_cycles": sum(row["accel_cycles"] for row in rows) / len(rows),
        "avg_sw_cycles": sum(row["sw_cycles"] for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify that benchmark_500.csv matches its source UART log.")
    parser.add_argument("--uart-log", type=Path, default=DEFAULT_UART)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    uart_rows = []
    uart_summary = {}
    source_description = None
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if "source_uart_logs" in manifest:
            source_description = str(args.manifest)
            for chunk in manifest["source_uart_logs"]:
                chunk_path = PROJECT_ROOT / chunk["uart_log"]
                chunk_rows, _ = parse_uart(chunk_path)
                offset = int(chunk["dataset_offset"])
                for row in chunk_rows:
                    uart_rows.append(
                        {
                            "image_index": offset + row["image_index"],
                            "accel_cycles": row["accel_cycles"],
                            "sw_cycles": row["sw_cycles"],
                        }
                    )
            uart_rows.sort(key=lambda row: row["image_index"])
        elif "source_uart_log" in manifest:
            source_description = str(args.manifest)
            uart_path = PROJECT_ROOT / manifest["source_uart_log"]
            uart_rows, uart_summary = parse_uart(uart_path)
        else:
            uart_rows, uart_summary = parse_uart(args.uart_log)
            source_description = str(args.uart_log)
    else:
        uart_rows, uart_summary = parse_uart(args.uart_log)
        source_description = str(args.uart_log)

    csv_rows = parse_csv(args.csv)
    uart_avg = averages(uart_rows)
    csv_avg = averages(csv_rows)

    print(f"Source manifest/log: {source_description}")
    print(f"CSV path: {args.csv}")
    print(f"UART rows: {uart_avg['row_count']}")
    print(f"CSV rows: {csv_avg['row_count']}")
    print(f"UART avg accelerator cycles: {uart_avg['avg_accel_cycles']:.3f}")
    print(f"CSV avg accelerator cycles: {csv_avg['avg_accel_cycles']:.3f}")
    print(f"UART avg software cycles: {uart_avg['avg_sw_cycles']:.3f}")
    print(f"CSV avg software cycles: {csv_avg['avg_sw_cycles']:.3f}")
    if "avg_accel_cycles" in uart_summary:
        print(f"UART summary accelerator cycles: {uart_summary['avg_accel_cycles']}")
    if "avg_sw_cycles" in uart_summary:
        print(f"UART summary software cycles: {uart_summary['avg_sw_cycles']}")

    failures = []
    if uart_avg["row_count"] != csv_avg["row_count"]:
        failures.append(f"row_count mismatch: uart={uart_avg['row_count']} csv={csv_avg['row_count']}")
    if abs(uart_avg["avg_accel_cycles"] - csv_avg["avg_accel_cycles"]) > args.tolerance:
        failures.append(
            "accelerator average mismatch: "
            f"uart={uart_avg['avg_accel_cycles']:.12f} csv={csv_avg['avg_accel_cycles']:.12f}"
        )
    if abs(uart_avg["avg_sw_cycles"] - csv_avg["avg_sw_cycles"]) > args.tolerance:
        failures.append(
            f"software average mismatch: uart={uart_avg['avg_sw_cycles']:.12f} csv={csv_avg['avg_sw_cycles']:.12f}"
        )
    if "images" in uart_summary and uart_summary["images"] != uart_avg["row_count"]:
        failures.append(f"UART summary image count mismatch: {uart_summary['images']} vs {uart_avg['row_count']}")
    if "avg_accel_cycles" in uart_summary and uart_summary["avg_accel_cycles"] != int(uart_avg["avg_accel_cycles"]):
        failures.append(
            f"UART summary accelerator mismatch: {uart_summary['avg_accel_cycles']} vs {int(uart_avg['avg_accel_cycles'])}"
        )
    if "avg_sw_cycles" in uart_summary and uart_summary["avg_sw_cycles"] != int(uart_avg["avg_sw_cycles"]):
        failures.append(
            f"UART summary software mismatch: {uart_summary['avg_sw_cycles']} vs {int(uart_avg['avg_sw_cycles'])}"
        )

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)

    print("PASS")


if __name__ == "__main__":
    main()
