import argparse
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

warnings.filterwarnings(
    "ignore",
    message=r".*align should be passed as.*",
    category=Warning,
)

# [span_1](start_span)تعريف هيكلية الشبكة كما في الدليل[span_1](end_span)
class TinyCatDogNet(nn.Module):
    def __init__(self):
        super(TinyCatDogNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 4 * 4, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_cpu_name() -> str:
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        with cpuinfo_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return "Unknown CPU"


def get_gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No CUDA GPU"


def get_cat_dog_testloader(data_root: Path, batch_size: int = 64):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    if not (data_root / "cifar-10-batches-py").exists():
        return None

    testset = torchvision.datasets.CIFAR10(
        root=str(data_root), train=False, download=False, transform=transform
    )

    indices = [i for i, label in enumerate(testset.targets) if label in [3, 5]]
    testset.targets = [0 if testset.targets[i] == 3 else 1 for i in indices]
    testset.data = testset.data[indices]
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )
    return testloader


def run_benchmark(device, model, iterations=1000, batch_size=1):
    model.to(device)
    model.eval()
    input_data = torch.randn(batch_size, 1, 32, 32, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(input_data)
        if device.type == "cuda":
            torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            model(input_data)
        if device.type == "cuda":
            torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    avg_latency = (total_time / iterations) * 1000.0
    fps = (iterations * batch_size) / total_time
    return avg_latency, fps


def evaluate_accuracy(device, model, testloader):
    if testloader is None:
        return None
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total if total else None


def get_cifar10_benchmark_samples(data_root: Path, count: int = 20):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    if not (data_root / "cifar-10-batches-py").exists():
        return []

    testset = torchvision.datasets.CIFAR10(
        root=str(data_root), train=False, download=False
    )

    samples = []
    class_names = {3: "Cat", 5: "Dog"}
    for idx, (image, label) in enumerate(testset):
        if label not in class_names:
            continue
        samples.append(
            {
                "name": f"cifar10_test_{idx}_{class_names[label].lower()}",
                "image": transform(image),
            }
        )
        if len(samples) == count:
            break
    return samples


def benchmark_images_on_device(device, model, benchmark_samples, iterations=100):
    model.to(device)
    model.eval()
    classes = ["Cat", "Dog"]
    results = []
    prepared_samples = [
        {"name": sample["name"], "image": sample["image"].unsqueeze(0).to(device)}
        for sample in benchmark_samples
    ]

    with torch.no_grad():
        for _ in range(10):
            for sample in prepared_samples:
                model(sample["image"])
        if device.type == "cuda":
            torch.cuda.synchronize()

        for sample in prepared_samples:
            start_time = time.perf_counter()
            for _ in range(iterations):
                logits = model(sample["image"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = ((time.perf_counter() - start_time) / iterations) * 1000.0
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            results.append(
                {
                    "name": sample["name"],
                    "prediction": classes[pred_idx],
                    "cat_prob": float(probs[0]),
                    "dog_prob": float(probs[1]),
                    "latency_ms": latency_ms,
                }
            )
    return results


def benchmark_batch_on_device(device, model, benchmark_samples, iterations=100):
    model.to(device)
    model.eval()
    batch_tensor = torch.stack([sample["image"] for sample in benchmark_samples], dim=0).to(device)

    with torch.no_grad():
        for _ in range(10):
            model(batch_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(iterations):
            model(batch_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    batch_latency_ms = (total_time / iterations) * 1000.0
    per_image_latency_ms = batch_latency_ms / len(benchmark_samples)
    fps = (iterations * len(benchmark_samples)) / total_time
    return batch_latency_ms, per_image_latency_ms, fps


def run_repeated_batch_benchmarks(device, model, benchmark_samples, iterations=100, rounds=5):
    round_results = []
    for round_idx in range(rounds):
        batch_ms, per_image_ms, fps = benchmark_batch_on_device(
            device, model, benchmark_samples, iterations=iterations
        )
        round_results.append(
            {
                "round": round_idx + 1,
                "batch_latency_ms": batch_ms,
                "per_image_latency_ms": per_image_ms,
                "fps": fps,
            }
        )
    return round_results


def summarize_rounds(round_results):
    avg_batch_ms = sum(row["batch_latency_ms"] for row in round_results) / len(round_results)
    avg_per_image_ms = sum(row["per_image_latency_ms"] for row in round_results) / len(round_results)
    avg_fps = sum(row["fps"] for row in round_results) / len(round_results)
    min_batch_ms = min(row["batch_latency_ms"] for row in round_results)
    max_batch_ms = max(row["batch_latency_ms"] for row in round_results)
    return avg_batch_ms, avg_per_image_ms, avg_fps, min_batch_ms, max_batch_ms


def infer_images(model, testloader=None, image_paths=None, image_samples=None, iterations=100):
    image_paths = image_paths or []
    image_samples = image_samples or []
    image_paths = [Path(p) for p in image_paths]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"[!] Image not found: {p}")
        return

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    benchmark_samples = []
    for p in image_paths:
        benchmark_samples.append(
            {
                "name": str(p),
                "image": transform(Image.open(p).convert("RGB")),
            }
        )
    benchmark_samples.extend(image_samples)

    if not benchmark_samples:
        print("[!] No images available for per-photo benchmarking.")
        return

    cpu_results = benchmark_images_on_device(
        torch.device("cpu"), model, benchmark_samples, iterations=iterations
    )

    gpu_results = None
    if torch.cuda.is_available():
        try:
            gpu_results = benchmark_images_on_device(
                torch.device("cuda"), model, benchmark_samples, iterations=iterations
            )
        except Exception as exc:
            print(f"[!] GPU per-photo benchmark unavailable: {exc}")

    print("\nBenchmark Table:")
    if gpu_results is not None:
        print(
            "Image                 | CPU Pred | CPU ms  | GPU Pred | GPU ms"
        )
        print(
            "----------------------|----------|---------|----------|--------"
        )
        for cpu_row, gpu_row in zip(cpu_results, gpu_results):
            print(
                f"{cpu_row['name'][:22]:<22} | "
                f"{cpu_row['prediction']:<8} | {cpu_row['latency_ms']:>7.4f} | "
                f"{gpu_row['prediction']:<8} | {gpu_row['latency_ms']:>6.4f}"
            )
    else:
        print("Image                 | CPU Pred | CPU ms")
        print("----------------------|----------|--------")
        for cpu_row in cpu_results:
            print(
                f"{cpu_row['name'][:22]:<22} | "
                f"{cpu_row['prediction']:<8} | {cpu_row['latency_ms']:>6.4f}"
            )

    cpu_avg_latency_ms = sum(row["latency_ms"] for row in cpu_results) / len(cpu_results)
    cpu_avg_fps = 1000.0 / cpu_avg_latency_ms if cpu_avg_latency_ms > 0 else float("inf")
    cpu_acc = evaluate_accuracy(torch.device("cpu"), model, testloader)

    print("\nTotal:")
    print("Platform | Images | Avg Latency (ms) | Throughput (FPS) | Accuracy (%)")
    print("---------|--------|------------------|------------------|-------------")
    cpu_acc_str = f"{cpu_acc:.2f}" if cpu_acc is not None else "N/A"
    print(f"CPU      | {len(cpu_results):>6} | {cpu_avg_latency_ms:>16.4f} | {cpu_avg_fps:>16.2f} | {cpu_acc_str:>11}")

    if gpu_results is not None:
        gpu_avg_latency_ms = sum(row["latency_ms"] for row in gpu_results) / len(gpu_results)
        gpu_avg_fps = 1000.0 / gpu_avg_latency_ms if gpu_avg_latency_ms > 0 else float("inf")
        gpu_acc = evaluate_accuracy(torch.device("cuda"), model, testloader)
        gpu_acc_str = f"{gpu_acc:.2f}" if gpu_acc is not None else "N/A"
        print(f"GPU      | {len(gpu_results):>6} | {gpu_avg_latency_ms:>16.4f} | {gpu_avg_fps:>16.2f} | {gpu_acc_str:>11}")

def main():
    parser = argparse.ArgumentParser(description="Cross-platform CNN benchmark")
    parser.add_argument(
        "--image",
        type=str,
        action="append",
        default=[],
        help="Optional image path for inference (repeatable)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for benchmark (default: 20)",
    )
    parser.add_argument(
        "--photo-iterations",
        type=int,
        default=100,
        help="Number of repeated runs per real image benchmark (default: 100)",
    )
    args = parser.parse_args()

    print("="*30)
    print(" CROSS-PLATFORM BENCHMARK ")
    print("="*30)

    repo_root = Path(__file__).resolve().parents[1]
    weights_path = repo_root / "best_catdog.pth"
    data_root = repo_root / "data"

    net = TinyCatDogNet()

    if weights_path.exists():
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"[*] Loaded weights from {weights_path}")
    else:
        print("[!] Weights not found, using random initialization.")

    print(f"[*] CPU: {get_cpu_name()}")
    print(f"[*] GPU: {get_gpu_name()}")
    print(f"[*] PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[*] CUDA: {torch.version.cuda}")
        try:
            _ = torch.cuda.get_device_name(0)
        except Exception as exc:
            print(f"[!] CUDA visible but unusable: {exc}")

    testloader = get_cat_dog_testloader(data_root)
    if testloader is None:
        print("[!] CIFAR-10 data not found. Accuracy will be skipped.")

    if args.image:
        infer_images(net, testloader=testloader, image_paths=args.image, iterations=args.photo_iterations)
    else:
        cifar_samples = get_cifar10_benchmark_samples(data_root, count=20)
        if cifar_samples:
            infer_images(net, testloader=testloader, image_samples=cifar_samples, iterations=args.photo_iterations)
        else:
            print("[!] CIFAR-10 samples not found for per-photo benchmark.")


if __name__ == "__main__":
    main()
