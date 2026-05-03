#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RENODE_DIR = PROJECT_ROOT / "renode"
RESC = RENODE_DIR / "run_headless_autostop.resc"
UART_OUT = Path("/tmp/ai_classification_renode_uart_output.txt")
RENODE_CONSOLE_OUT = Path("/tmp/ai_classification_renode_console.txt")
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "best_catdog.pth"
DATASET_EXPORT = PROJECT_ROOT / "scripts" / "export_eval_dataset.py"
FIRMWARE_DIR = PROJECT_ROOT / "firmware"
BENCHMARK_RESULT_PATTERNS = [
    "renode_runs_mpc_*.csv",
    "renode_summary_mpc_*.csv",
    "renode_sweep.csv",
    "speedup_sweep.csv",
    "float_reference_accuracy.csv",
    "renode_speedup_vs_mpc.png",
]

ACCEL_TOTAL_RE = re.compile(r"\[ACCEL_SIM\] Cycles:\s*(\d+)")
SW_TOTAL_RE = re.compile(r"\[SW_ONLY\]\s*Cycles:\s*(\d+)")
MODELED_RE = re.compile(r"Modeled accelerator cycles:\s*(\d+)")
PRED_RE = re.compile(r"Prediction:\s*(CAT|DOG)")
EXP_RE = re.compile(r"Expected:\s*(CAT|DOG)")
IMAGE_RE = re.compile(r"=== Image\s+(\d+)")
STAGE_RE = re.compile(r"\s+(conv1|relu1|pool1|conv2|relu2|pool2|conv3|relu3|pool3|flatten|fc|sigmoid):\s*(\d+)")
CPU_HZ = 25_000_000


def get_torch_modules():
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms

    return torch, nn, torchvision, transforms


def create_model(nn):
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

    return TinyCatDogNet()


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    phat = correct / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / total) + ((z * z) / (4.0 * total * total)))
    return (center - margin, center + margin)


def run_cmd(cmd, env=None, cwd=PROJECT_ROOT):
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def monitor_path(path: Path) -> str:
    return str(path).replace(" ", "\\ ")


def ensure_dataset(count: int, offset: int):
    run_cmd([sys.executable, str(DATASET_EXPORT), "--count", str(count), "--offset", str(offset)])


def build_firmware(extra_cflags: str = ""):
    run_cmd(["make", "-C", str(FIRMWARE_DIR), "clean"])
    cmd = ["make", "-C", str(FIRMWARE_DIR)]
    if extra_cflags:
        cmd.append(f"EXTRA_CFLAGS={extra_cflags}")
    run_cmd(cmd)


def launch_renode(env, timeout_s: int):
    if UART_OUT.exists():
        UART_OUT.unlink()
    if RENODE_CONSOLE_OUT.exists():
        RENODE_CONSOLE_OUT.unlink()

    with RENODE_CONSOLE_OUT.open("wb") as console_log:
        proc = subprocess.Popen(
            ["renode", "--plain", "--disable-xwt", "--console", "-e", f"include @{monitor_path(RESC)}"],
            # Renode resolves relative paths inside included .resc files against the
            # process working directory, not against the .resc file location.
            # Run from renode/ so checked-in relative paths in run_headless*.resc
            # resolve to this repository, not to some other copy on disk.
            cwd=RENODE_DIR,
            env=env,
            stdout=console_log,
            stderr=subprocess.STDOUT,
            text=False,
        )

        deadline = time.time() + timeout_s
        try:
            while time.time() < deadline:
                if UART_OUT.exists():
                    text = UART_OUT.read_text(encoding="utf-8", errors="ignore")
                    if "Done." in text:
                        return text
                if proc.poll() is not None:
                    text = ""
                    if UART_OUT.exists():
                        text = UART_OUT.read_text(encoding="utf-8", errors="ignore")
                        if "Done." in text:
                            return text
                    console_tail = ""
                    if RENODE_CONSOLE_OUT.exists():
                        console_lines = RENODE_CONSOLE_OUT.read_text(encoding="utf-8", errors="ignore").splitlines()
                        console_tail = "\n".join(console_lines[-20:])
                    raise RuntimeError(
                        "Renode exited before benchmark completion.\n"
                        f"Console tail:\n{console_tail}"
                    )
                time.sleep(1.0)
            raise TimeoutError(f"Timed out after {timeout_s}s waiting for Renode benchmark completion")
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


def parse_runs(text: str):
    runs = []
    current = None
    mode = None
    pending_prediction = None
    pending_expected = None

    for raw_line in text.splitlines():
        line = raw_line.strip("\r")

        image_match = IMAGE_RE.match(line)
        if image_match:
            current = {
                "image_index": int(image_match.group(1)),
                "accel_total_cycles": None,
                "accel_modeled_cycles": None,
                "sw_total_cycles": None,
                "accel_prediction": None,
                "sw_prediction": None,
                "expected_label": None,
            }
            for stage in ["conv1", "relu1", "pool1", "conv2", "relu2", "pool2", "conv3", "relu3", "pool3", "flatten", "fc", "sigmoid"]:
                current[f"accel_{stage}_cycles"] = None
                current[f"sw_{stage}_cycles"] = None
            runs.append(current)
            mode = None
            pending_prediction = None
            pending_expected = None
            continue

        if current is None:
            continue

        if line.startswith("--- Renode-Modeled Accelerator"):
            mode = "accel"
            continue
        if line.startswith("--- Software Only"):
            mode = "sw"
            continue

        stage_match = STAGE_RE.match(line)
        if stage_match and mode in {"accel", "sw"}:
            current[f"{mode}_{stage_match.group(1)}_cycles"] = int(stage_match.group(2))
            continue

        total_match = ACCEL_TOTAL_RE.search(line)
        if total_match:
            current["accel_total_cycles"] = int(total_match.group(1))
            continue

        total_match = SW_TOTAL_RE.search(line)
        if total_match:
            current["sw_total_cycles"] = int(total_match.group(1))
            continue

        modeled_match = MODELED_RE.search(line)
        if modeled_match:
            current["accel_modeled_cycles"] = int(modeled_match.group(1))
            continue

        pred_match = PRED_RE.search(line)
        if pred_match:
            pending_prediction = pred_match.group(1)
            continue

        exp_match = EXP_RE.search(line)
        if exp_match:
            pending_expected = exp_match.group(1)
            if mode == "accel":
                current["accel_prediction"] = pending_prediction
            elif mode == "sw":
                current["sw_prediction"] = pending_prediction
            current["expected_label"] = pending_expected
            pending_prediction = None
            pending_expected = None

    return runs


def load_baseline_runs(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "image_index": int(row["image_index"]),
                    "sw_total_cycles": int(row["sw_cycles"]),
                    "sw_prediction": row["prediction_sw"],
                    "expected_label": row["expected_label"],
                }
            )
    return rows


def merge_sw_baseline(runs, baseline_runs):
    baseline_by_index = {row["image_index"]: row for row in baseline_runs}
    for row in runs:
        if row["sw_total_cycles"] is None:
            baseline = baseline_by_index.get(row["image_index"])
            if baseline is None:
                raise RuntimeError(f"Missing software baseline row for image {row['image_index']}")
            row["sw_total_cycles"] = baseline["sw_total_cycles"]
            row["sw_prediction"] = baseline["sw_prediction"]
            if row["expected_label"] is None:
                row["expected_label"] = baseline["expected_label"]
    return runs


def summarize_runs(runs):
    accel_avg = sum(row["accel_total_cycles"] for row in runs) / len(runs)
    sw_avg = sum(row["sw_total_cycles"] for row in runs) / len(runs)
    speedup = sw_avg / accel_avg
    accel_correct = sum(1 for row in runs if row["accel_prediction"] == row["expected_label"])
    sw_correct = sum(1 for row in runs if row["sw_prediction"] == row["expected_label"])
    accel_ci = wilson_interval(accel_correct, len(runs))
    sw_ci = wilson_interval(sw_correct, len(runs))
    return {
        "avg_cycles_accel": accel_avg,
        "avg_cycles_sw": sw_avg,
        "speedup": speedup,
        "accuracy": 100.0 * accel_correct / len(runs),
        "accuracy_sw": 100.0 * sw_correct / len(runs),
        "accuracy_ci_low": 100.0 * accel_ci[0],
        "accuracy_ci_high": 100.0 * accel_ci[1],
        "accuracy_sw_ci_low": 100.0 * sw_ci[0],
        "accuracy_sw_ci_high": 100.0 * sw_ci[1],
    }


def validate_runs(runs, require_sw: bool = True):
    if not runs:
        raise RuntimeError("No Renode runs were parsed from uart_output.txt")
    for row in runs:
        if row["accel_total_cycles"] is None or (require_sw and row["sw_total_cycles"] is None):
            raise RuntimeError(f"Incomplete timing row for image {row['image_index']}")
        if row["accel_modeled_cycles"] in (None, 0):
            raise RuntimeError(f"Missing modeled accelerator cycles for image {row['image_index']}")
    conv1 = [row["accel_conv1_cycles"] for row in runs if row["accel_conv1_cycles"] is not None]
    conv2 = [row["accel_conv2_cycles"] for row in runs if row["accel_conv2_cycles"] is not None]
    conv3 = [row["accel_conv3_cycles"] for row in runs if row["accel_conv3_cycles"] is not None]
    if conv1 and conv2 and conv3 and min(sum(conv1) / len(conv1), sum(conv2) / len(conv2), sum(conv3) / len(conv3)) < 1000:
        raise RuntimeError(
            "Accelerated convolution stages are still implausibly small; modeled timing likely regressed"
        )


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def remove_existing_benchmark_outputs(results_dir: Path):
    for pattern in BENCHMARK_RESULT_PATTERNS:
        for path in results_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def get_eval_subset(count: int):
    torch, _, torchvision, transforms = get_torch_modules()
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(DATA_ROOT), train=False, download=False, transform=transform
    )
    images = []
    labels = []
    for image, label in testset:
        if label not in [3, 5]:
            continue
        images.append(image)
        labels.append(0 if label == 3 else 1)
        if len(images) == count:
            break
    if not images:
        raise RuntimeError("No CIFAR-10 cat/dog evaluation images found for float reference accuracy")
    return torch.stack(images, dim=0), torch.tensor(labels)


def evaluate_float_reference(count: int):
    if not MODEL_PATH.exists():
        return None
    torch, nn, _, _ = get_torch_modules()
    model = create_model(nn)
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
        "dataset_size": int(len(labels)),
        "float_accuracy": 100.0 * correct / len(labels),
        "float_accuracy_ci_low": 100.0 * low,
        "float_accuracy_ci_high": 100.0 * high,
    }


def render_plot(csv_path: Path, output_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    x = [int(row["macs_per_cycle"]) for row in rows]
    y = [float(row["speedup"]) for row in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("MACS_PER_CYCLE")
    plt.ylabel("Simulated speedup vs software")
    plt.title("Renode-modeled accelerator sweep")
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the Renode-modeled accelerator benchmark sweep.")
    parser.add_argument("--dataset-count", type=int, default=500)
    parser.add_argument("--dataset-offset", type=int, default=0)
    parser.add_argument("--macs-per-cycle", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--accel-only", action="store_true")
    parser.add_argument("--reuse-sw-benchmark", type=Path, default=None)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    remove_existing_benchmark_outputs(args.results_dir)
    ensure_dataset(args.dataset_count, args.dataset_offset)
    build_firmware("-DSKIP_SW_BENCHMARK" if args.accel_only else "")

    baseline_runs = None
    if args.reuse_sw_benchmark is not None:
        baseline_runs = load_baseline_runs(args.reuse_sw_benchmark)

    sweep_rows = []
    detailed_paths = []

    for macs_per_cycle in args.macs_per_cycle:
        env = os.environ.copy()
        env["RENODE_MACS_PER_CYCLE"] = str(macs_per_cycle)
        text = launch_renode(env, timeout_s=args.timeout)
        runs = parse_runs(text)
        if baseline_runs is not None:
            runs = merge_sw_baseline(runs, baseline_runs)
        validate_runs(runs, require_sw=not args.accel_only)
        summary = summarize_runs(runs)

        detail_path = args.results_dir / f"renode_runs_mpc_{macs_per_cycle}.csv"
        detail_fields = list(runs[0].keys())
        write_csv(detail_path, runs, detail_fields)
        detailed_paths.append(detail_path)

        sweep_rows.append(
            {
                "dataset_size": args.dataset_count,
                "macs_per_cycle": macs_per_cycle,
                "avg_cycles_accel": f"{summary['avg_cycles_accel']:.3f}",
                "avg_cycles_sw": f"{summary['avg_cycles_sw']:.3f}",
                "speedup": f"{summary['speedup']:.6f}",
                "accuracy": f"{summary['accuracy']:.3f}",
                "accuracy_ci_low": f"{summary['accuracy_ci_low']:.3f}",
                "accuracy_ci_high": f"{summary['accuracy_ci_high']:.3f}",
                "accuracy_sw": f"{summary['accuracy_sw']:.3f}",
                "accuracy_sw_ci_low": f"{summary['accuracy_sw_ci_low']:.3f}",
                "accuracy_sw_ci_high": f"{summary['accuracy_sw_ci_high']:.3f}",
                "simulated_latency_ms_accel": f"{(summary['avg_cycles_accel'] / CPU_HZ) * 1000.0:.6f}",
                "simulated_latency_ms_sw": f"{(summary['avg_cycles_sw'] / CPU_HZ) * 1000.0:.6f}",
            }
        )

        write_csv(
            args.results_dir / f"renode_summary_mpc_{macs_per_cycle}.csv",
            [sweep_rows[-1]],
            [
                "dataset_size",
                "macs_per_cycle",
                "avg_cycles_accel",
                "avg_cycles_sw",
                "speedup",
                "accuracy",
                "accuracy_ci_low",
                "accuracy_ci_high",
                "accuracy_sw",
                "accuracy_sw_ci_low",
                "accuracy_sw_ci_high",
                "simulated_latency_ms_accel",
                "simulated_latency_ms_sw",
            ],
        )

    sweep_path = args.results_dir / "renode_sweep.csv"
    write_csv(
        sweep_path,
        sweep_rows,
        [
            "dataset_size",
            "macs_per_cycle",
            "avg_cycles_accel",
            "avg_cycles_sw",
            "speedup",
            "accuracy",
            "accuracy_ci_low",
            "accuracy_ci_high",
            "accuracy_sw",
            "accuracy_sw_ci_low",
            "accuracy_sw_ci_high",
            "simulated_latency_ms_accel",
            "simulated_latency_ms_sw",
        ],
    )

    float_reference = evaluate_float_reference(args.dataset_count)
    if float_reference is not None:
        write_csv(
            args.results_dir / "float_reference_accuracy.csv",
            [float_reference],
            ["dataset_size", "float_accuracy", "float_accuracy_ci_low", "float_accuracy_ci_high"],
        )

    plot_ok = render_plot(sweep_path, args.results_dir / "renode_speedup_vs_mpc.png")

    print("Renode-modeled benchmark sweep complete.")
    print(f"Sweep CSV: {sweep_path}")
    for path in detailed_paths:
        print(f"Detailed runs: {path}")
    if float_reference is not None:
        print(f"Float reference CSV: {args.results_dir / 'float_reference_accuracy.csv'}")
    if plot_ok:
        print(f"Plot: {args.results_dir / 'renode_speedup_vs_mpc.png'}")
    else:
        print("Plot skipped: matplotlib unavailable")


if __name__ == "__main__":
    main()
