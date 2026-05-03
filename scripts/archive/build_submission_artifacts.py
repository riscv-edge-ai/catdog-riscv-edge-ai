#!/usr/bin/env python3
# Legacy-only packaging entry point. The canonical benchmark flow is chunked and
# manifest-backed via scripts/build_chunked_submission_artifacts.py.
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGS_DIR = PROJECT_ROOT / "figs"
AUTHORITATIVE_UART_PATH = RESULTS_DIR / "renode_direct_uart_authoritative.txt"
BENCHMARK_MANIFEST_PATH = RESULTS_DIR / "benchmark_500_manifest.json"
MODEL_PATH = PROJECT_ROOT / "best_catdog.pth"

ACCEL_RE = re.compile(r"\[ACCEL_SIM\] Cycles:\s*(\d+)")
SW_RE = re.compile(r"\[SW_ONLY\]\s*Cycles:\s*(\d+)")
PRED_RE = re.compile(r"Prediction:\s*(CAT|DOG)")
EXP_RE = re.compile(r"Expected:\s*(CAT|DOG)")
IMAGE_RE = re.compile(r"=== Image\s+(\d+)")
SUMMARY_IMAGES_RE = re.compile(r"Images:\s*(\d+)")
SUMMARY_ACCEL_RE = re.compile(r"Average accelerator cycles:\s*(\d+)")
SUMMARY_SW_RE = re.compile(r"Average software cycles:\s*(\d+)")


class TinyCatDogNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 4 * 4, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    phat = correct / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / total) + ((z * z) / (4.0 * total * total)))
    return (center - margin, center + margin)


def parse_uart_runs(text: str):
    rows = []
    current = None
    mode = None
    pending_prediction = None
    pending_expected = None
    summary = {}

    for raw_line in text.splitlines():
        line = raw_line.strip("\r")

        m = IMAGE_RE.match(line)
        if m:
            current = {
                "image_index": int(m.group(1)),
                "accel_cycles": None,
                "sw_cycles": None,
                "accel_prediction": None,
                "sw_prediction": None,
                "expected_label": None,
            }
            rows.append(current)
            mode = None
            pending_prediction = None
            pending_expected = None
            continue

        if current is None:
            sm = SUMMARY_IMAGES_RE.search(line)
            if sm:
                summary["images"] = int(sm.group(1))
            sm = SUMMARY_ACCEL_RE.search(line)
            if sm:
                summary["avg_accel_cycles"] = int(sm.group(1))
            sm = SUMMARY_SW_RE.search(line)
            if sm:
                summary["avg_sw_cycles"] = int(sm.group(1))
            continue

        if line.startswith("--- Renode-Modeled Accelerator"):
            mode = "accel"
            continue
        if line.startswith("--- Software Only"):
            mode = "sw"
            continue

        m = ACCEL_RE.search(line)
        if m:
            current["accel_cycles"] = int(m.group(1))
            continue
        m = SW_RE.search(line)
        if m:
            current["sw_cycles"] = int(m.group(1))
            continue
        m = PRED_RE.search(line)
        if m:
            pending_prediction = m.group(1)
            continue
        m = EXP_RE.search(line)
        if m:
            pending_expected = m.group(1)
            current["expected_label"] = pending_expected
            if mode == "accel":
                current["accel_prediction"] = pending_prediction
            elif mode == "sw":
                current["sw_prediction"] = pending_prediction
            pending_prediction = None
            pending_expected = None
            continue

        sm = SUMMARY_IMAGES_RE.search(line)
        if sm:
            summary["images"] = int(sm.group(1))
        sm = SUMMARY_ACCEL_RE.search(line)
        if sm:
            summary["avg_accel_cycles"] = int(sm.group(1))
        sm = SUMMARY_SW_RE.search(line)
        if sm:
            summary["avg_sw_cycles"] = int(sm.group(1))

    return rows, summary


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_rows(rows):
    if not rows:
        raise SystemExit("No benchmark rows available for summary")
    return {
        "row_count": len(rows),
        "avg_accel_cycles": sum(row["accel_cycles"] for row in rows) / len(rows),
        "avg_sw_cycles": sum(row["sw_cycles"] for row in rows) / len(rows),
    }


def validate_summary(rows, summary):
    derived = summarize_rows(rows)
    summary_images = summary.get("images")
    if summary_images is not None and summary_images != derived["row_count"]:
        raise SystemExit(
            f"UART summary image count mismatch: summary={summary_images} parsed_rows={derived['row_count']}"
        )
    summary_accel = summary.get("avg_accel_cycles")
    if summary_accel is not None and summary_accel != int(derived["avg_accel_cycles"]):
        raise SystemExit(
            "UART summary accelerator average mismatch: "
            f"summary={summary_accel} parsed={int(derived['avg_accel_cycles'])}"
        )
    summary_sw = summary.get("avg_sw_cycles")
    if summary_sw is not None and summary_sw != int(derived["avg_sw_cycles"]):
        raise SystemExit(
            f"UART summary software average mismatch: summary={summary_sw} parsed={int(derived['avg_sw_cycles'])}"
        )
    return derived


def write_manifest(uart_path: Path, rows, summary):
    derived = summarize_rows(rows)
    payload = {
        "benchmark_csv": "results/benchmark_500.csv",
        "source_uart_log": str(uart_path.relative_to(PROJECT_ROOT)),
        "source_uart_sha256": sha256_file(uart_path),
        "row_count": derived["row_count"],
        "avg_accel_cycles_from_rows": round(derived["avg_accel_cycles"], 3),
        "avg_sw_cycles_from_rows": round(derived["avg_sw_cycles"], 3),
        "uart_summary_avg_accel_cycles": summary.get("avg_accel_cycles"),
        "uart_summary_avg_sw_cycles": summary.get("avg_sw_cycles"),
        "notes": [
            "benchmark_500.csv is parsed from the authoritative preserved UART log above",
            "speedup_sweep.csv and renode_sweep.csv are aggregated from renode_summary_mpc_*.csv",
        ],
    }
    BENCHMARK_MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_existing_manifest():
    if not BENCHMARK_MANIFEST_PATH.exists():
        return None
    return json.loads(BENCHMARK_MANIFEST_PATH.read_text(encoding="utf-8"))


def ensure_not_overwriting_chunked_benchmark(allow_overwrite: bool):
    manifest = load_existing_manifest()
    if not manifest or "source_uart_logs" not in manifest:
        return
    if allow_overwrite:
        return
    raise SystemExit(
        "Refusing to overwrite the current chunk-backed benchmark artifacts. "
        f"{BENCHMARK_MANIFEST_PATH} records source_uart_logs, so "
        "results/benchmark_500.csv is currently canonicalized from chunked UART runs. "
        "Use scripts/build_chunked_submission_artifacts.py for the canonical path, "
        "or rerun this script with --allow-overwrite-from-authoritative if you intentionally "
        "want to replace the chunk-backed benchmark with a single authoritative UART log."
    )


def get_eval_subset(count: int):
    data_root = PROJECT_ROOT / "data"
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=False, transform=transform)

    images = []
    labels = []
    for image, label in testset:
        if label not in [3, 5]:
            continue
        images.append(image)
        labels.append(0 if label == 3 else 1)
        if len(images) == count:
            break
    return torch.stack(images, dim=0), torch.tensor(labels)


def evaluate_float_reference(count: int):
    model = TinyCatDogNet()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    inputs, labels = get_eval_subset(count)
    with torch.no_grad():
        logits = model(inputs)
        preds = logits.argmax(dim=1)
    correct = int((preds == labels).sum().item())
    low, high = wilson_interval(correct, len(labels))
    return {
        "dataset_size": len(labels),
        "float_accuracy": 100.0 * correct / len(labels),
        "float_accuracy_ci_low": 100.0 * low,
        "float_accuracy_ci_high": 100.0 * high,
    }


def render_pdf_and_png(rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x = [int(row["macs_per_cycle"]) for row in rows]
    y = [float(row["speedup"]) for row in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("MACS_PER_CYCLE")
    plt.ylabel("Simulated speedup vs software")
    plt.title("Renode-modeled accelerator sweep")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "speedup_sweep.pdf")
    plt.savefig(RESULTS_DIR / "renode_speedup_vs_mpc.png", dpi=200)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Legacy-only: build benchmark artifacts from a single preserved authoritative UART log."
    )
    parser.add_argument(
        "--allow-overwrite-from-authoritative",
        action="store_true",
        help="Permit this legacy single-UART path to overwrite an existing chunk-backed benchmark_500.csv.",
    )
    args = parser.parse_args()

    ensure_not_overwriting_chunked_benchmark(args.allow_overwrite_from_authoritative)

    uart_path = AUTHORITATIVE_UART_PATH
    if not uart_path.exists():
        raise SystemExit(
            f"Missing authoritative single-run UART file: {uart_path}. "
            "Use scripts/build_chunked_submission_artifacts.py when the benchmark is packaged from chunked UART logs."
        )

    text = uart_path.read_text(encoding="utf-8", errors="ignore")
    runs, summary = parse_uart_runs(text)
    if not runs:
        raise SystemExit("No per-image runs parsed from UART output")
    validate_summary(runs, summary)

    benchmark_rows = []
    accel_correct = 0
    sw_correct = 0
    for row in runs:
        accel_ok = row["accel_prediction"] == row["expected_label"]
        sw_ok = row["sw_prediction"] == row["expected_label"]
        if accel_ok:
            accel_correct += 1
        if sw_ok:
            sw_correct += 1
        benchmark_rows.append(
            {
                "image_index": row["image_index"],
                "prediction_accel": row["accel_prediction"],
                "prediction_sw": row["sw_prediction"],
                "expected_label": row["expected_label"],
                "accel_cycles": row["accel_cycles"],
                "sw_cycles": row["sw_cycles"],
                "accel_correct": int(accel_ok),
                "sw_correct": int(sw_ok),
            }
        )

    write_csv(
        RESULTS_DIR / "benchmark_500.csv",
        benchmark_rows,
        [
            "image_index",
            "prediction_accel",
            "prediction_sw",
            "expected_label",
            "accel_cycles",
            "sw_cycles",
            "accel_correct",
            "sw_correct",
        ],
    )

    total = len(benchmark_rows)
    accel_low, accel_high = wilson_interval(accel_correct, total)
    sw_low, sw_high = wilson_interval(sw_correct, total)
    float_ref = evaluate_float_reference(total)
    quant_acc = 100.0 * accel_correct / total
    delta = float_ref["float_accuracy"] - quant_acc

    table_rows = [
        {
            "dataset_size": total,
            "metric": "Renode-modeled Q8.8 accelerator",
            "accuracy_percent": f"{quant_acc:.3f}",
            "ci_low": f"{100.0 * accel_low:.3f}",
            "ci_high": f"{100.0 * accel_high:.3f}",
        },
        {
            "dataset_size": total,
            "metric": "Renode-modeled software-only",
            "accuracy_percent": f"{(100.0 * sw_correct / total):.3f}",
            "ci_low": f"{100.0 * sw_low:.3f}",
            "ci_high": f"{100.0 * sw_high:.3f}",
        },
        {
            "dataset_size": total,
            "metric": "Float reference",
            "accuracy_percent": f"{float_ref['float_accuracy']:.3f}",
            "ci_low": f"{float_ref['float_accuracy_ci_low']:.3f}",
            "ci_high": f"{float_ref['float_accuracy_ci_high']:.3f}",
        },
        {
            "dataset_size": total,
            "metric": "Float-vs-Q8.8 delta",
            "accuracy_percent": f"{delta:.3f}",
            "ci_low": "",
            "ci_high": "",
        },
    ]
    write_csv(
        RESULTS_DIR / "table1_accuracy_500.csv",
        table_rows,
        ["dataset_size", "metric", "accuracy_percent", "ci_low", "ci_high"],
    )
    write_csv(
        RESULTS_DIR / "float_reference_accuracy.csv",
        [float_ref],
        ["dataset_size", "float_accuracy", "float_accuracy_ci_low", "float_accuracy_ci_high"],
    )
    write_manifest(uart_path, benchmark_rows, summary)

    speedup_rows = []
    for path in sorted(RESULTS_DIR.glob("renode_summary_mpc_*.csv")):
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            speedup_rows.extend(reader)
    if speedup_rows:
        speedup_rows.sort(key=lambda row: int(row["macs_per_cycle"]))
        write_csv(RESULTS_DIR / "speedup_sweep.csv", speedup_rows, list(speedup_rows[0].keys()))
        write_csv(RESULTS_DIR / "renode_sweep.csv", speedup_rows, list(speedup_rows[0].keys()))
        render_pdf_and_png(speedup_rows)

    print(f"Source UART: {uart_path}")
    print(f"Wrote {RESULTS_DIR / 'benchmark_500.csv'}")
    print(f"Wrote {BENCHMARK_MANIFEST_PATH}")
    print(f"Wrote {RESULTS_DIR / 'table1_accuracy_500.csv'}")
    print(f"Wrote {RESULTS_DIR / 'float_reference_accuracy.csv'}")
    if speedup_rows:
        print(f"Wrote {RESULTS_DIR / 'speedup_sweep.csv'}")
        print(f"Wrote {FIGS_DIR / 'speedup_sweep.pdf'}")


if __name__ == "__main__":
    main()
