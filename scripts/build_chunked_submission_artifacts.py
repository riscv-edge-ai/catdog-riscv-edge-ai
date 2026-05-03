#!/usr/bin/env python3
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
CHUNK_METADATA_GLOB = "chunked/renode_direct_uart_offset_*_count_*_mpc_4.txt.json"
BENCHMARK_CSV_PATH = RESULTS_DIR / "benchmark_500.csv"
BENCHMARK_MANIFEST_PATH = RESULTS_DIR / "benchmark_500_manifest.json"
MODEL_PATH = PROJECT_ROOT / "best_catdog.pth"
CHUNKED_ARTIFACT_PATHS = [
    BENCHMARK_CSV_PATH,
    BENCHMARK_MANIFEST_PATH,
    RESULTS_DIR / "table1_accuracy_500.csv",
    RESULTS_DIR / "float_reference_accuracy.csv",
]

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


def parse_uart_runs(text: str):
    rows = []
    current = None
    mode = None
    pending_prediction = None
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
            current["expected_label"] = m.group(1)
            if mode == "accel":
                current["accel_prediction"] = pending_prediction
            elif mode == "sw":
                current["sw_prediction"] = pending_prediction
            pending_prediction = None

    return rows, summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    phat = correct / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / total) + ((z * z) / (4.0 * total * total)))
    return (center - margin, center + margin)


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


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def remove_existing_chunked_artifacts():
    for path in CHUNKED_ARTIFACT_PATHS:
        if path.exists():
            path.unlink()


def main():
    remove_existing_chunked_artifacts()

    metadata_paths = sorted(RESULTS_DIR.glob(CHUNK_METADATA_GLOB))
    if not metadata_paths:
        raise SystemExit(f"No chunk metadata files found matching results/{CHUNK_METADATA_GLOB}")

    combined_rows = []
    chunk_entries = []
    expected_next_index = 0

    for meta_path in metadata_paths:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        uart_path = PROJECT_ROOT / meta["uart_log"]
        text = uart_path.read_text(encoding="utf-8", errors="ignore")
        rows, summary = parse_uart_runs(text)
        if len(rows) != meta["dataset_count"]:
            raise SystemExit(
                f"Chunk row count mismatch for {uart_path}: parsed={len(rows)} expected={meta['dataset_count']}"
            )
        if summary.get("images") not in (None, meta["dataset_count"]):
            raise SystemExit(
                f"Chunk summary image mismatch for {uart_path}: summary={summary.get('images')} "
                f"expected={meta['dataset_count']}"
            )

        if meta["dataset_offset"] != expected_next_index:
            raise SystemExit(
                f"Chunk ordering mismatch: expected offset {expected_next_index}, found {meta['dataset_offset']} in {meta_path}"
            )

        for row in rows:
            global_index = meta["dataset_offset"] + row["image_index"]
            accel_ok = row["accel_prediction"] == row["expected_label"]
            sw_ok = row["sw_prediction"] == row["expected_label"]
            combined_rows.append(
                {
                    "image_index": global_index,
                    "prediction_accel": row["accel_prediction"],
                    "prediction_sw": row["sw_prediction"],
                    "expected_label": row["expected_label"],
                    "accel_cycles": row["accel_cycles"],
                    "sw_cycles": row["sw_cycles"],
                    "accel_correct": int(accel_ok),
                    "sw_correct": int(sw_ok),
                }
            )

        chunk_entries.append(
            {
                "uart_log": str(uart_path.relative_to(PROJECT_ROOT)),
                "uart_sha256": sha256_file(uart_path),
                "metadata_json": str(meta_path.relative_to(PROJECT_ROOT)),
                "dataset_count": meta["dataset_count"],
                "dataset_offset": meta["dataset_offset"],
                "macs_per_cycle": meta["macs_per_cycle"],
                "avg_accel_cycles_from_rows": round(sum(row["accel_cycles"] for row in combined_rows[-len(rows):]) / len(rows), 3),
                "avg_sw_cycles_from_rows": round(sum(row["sw_cycles"] for row in combined_rows[-len(rows):]) / len(rows), 3),
            }
        )
        expected_next_index += meta["dataset_count"]

    combined_rows.sort(key=lambda row: row["image_index"])
    total = len(combined_rows)
    if total != expected_next_index:
        raise SystemExit(f"Combined row count mismatch: rows={total} expected={expected_next_index}")

    write_csv(
        BENCHMARK_CSV_PATH,
        combined_rows,
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

    accel_correct = sum(row["accel_correct"] for row in combined_rows)
    sw_correct = sum(row["sw_correct"] for row in combined_rows)
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

    manifest = {
        "benchmark_csv": str(BENCHMARK_CSV_PATH.relative_to(PROJECT_ROOT)),
        "source_uart_logs": chunk_entries,
        "row_count": total,
        "avg_accel_cycles_from_rows": round(sum(row["accel_cycles"] for row in combined_rows) / total, 3),
        "avg_sw_cycles_from_rows": round(sum(row["sw_cycles"] for row in combined_rows) / total, 3),
        "notes": [
            "benchmark_500.csv is aggregated from multiple chunked Renode UART runs",
            "each chunk is a real rerun with preserved UART and explicit dataset offset metadata",
        ],
    }
    BENCHMARK_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {BENCHMARK_CSV_PATH}")
    print(f"Wrote {BENCHMARK_MANIFEST_PATH}")
    print(f"Wrote {RESULTS_DIR / 'table1_accuracy_500.csv'}")
    print(f"Wrote {RESULTS_DIR / 'float_reference_accuracy.csv'}")


if __name__ == "__main__":
    main()
