import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

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


def run_benchmark(device, model, iterations=1000):
    model.to(device)
    model.eval()
    input_data = torch.randn(1, 1, 32, 32, device=device)

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
    fps = iterations / total_time
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

def main():
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

    testloader = get_cat_dog_testloader(data_root)
    if testloader is None:
        print("[!] CIFAR-10 data not found. Accuracy will be skipped.")

    results = []

    cpu_device = torch.device("cpu")
    cpu_latency, cpu_fps = run_benchmark(cpu_device, net, iterations=100)
    cpu_acc = evaluate_accuracy(cpu_device, net, testloader)
    results.append(("CPU", cpu_latency, cpu_fps, cpu_acc))

    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_latency, gpu_fps = run_benchmark(gpu_device, net, iterations=1000)
        gpu_acc = evaluate_accuracy(gpu_device, net, testloader)
        results.append(("GPU", gpu_latency, gpu_fps, gpu_acc))
    else:
        print("[*] CUDA GPU not detected on this system.")

    print("\nResults:")
    print("Platform | Avg Latency (ms) | Throughput (FPS) | Accuracy (%)")
    print("---------|------------------|------------------|-------------")
    for platform, latency, fps, acc in results:
        acc_str = f"{acc:.2f}" if acc is not None else "N/A"
        print(f"{platform:<8}| {latency:>16.4f} | {fps:>16.2f} | {acc_str:>11}")


if __name__ == "__main__":
    main()
