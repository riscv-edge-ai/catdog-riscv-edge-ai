"""Train the tiny CIFAR-10 cat/dog CNN and export firmware assets."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "best_catdog.pth"
FIRMWARE_INC_DIR = PROJECT_ROOT / "firmware" / "include"
FIRMWARE_LUT_DIR = PROJECT_ROOT / "firmware" / "lut"

FIRMWARE_INC_DIR.mkdir(parents=True, exist_ok=True)
FIRMWARE_LUT_DIR.mkdir(parents=True, exist_ok=True)


class TinyCatDogNet(nn.Module):
    """Small grayscale CNN for binary cat/dog classification."""

    def __init__(self):
        super(TinyCatDogNet, self).__init__()
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
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(32 * 4 * 4, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32*4*4)
        x = self.classifier(x)
        return x


def get_cat_dog_dataloaders(data_dir=DEFAULT_DATA_DIR, batch_size=64, download=True):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=download, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=download, transform=transform
    )

    def filter_dataset(dataset):
        indices = [i for i, label in enumerate(dataset.targets) if label in [3, 5]]
        dataset.targets = [0 if dataset.targets[i] == 3 else 1 for i in indices]
        dataset.data = dataset.data[indices]
        return dataset

    trainset = filter_dataset(trainset)
    testset = filter_dataset(testset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)  # small batch for test images

    return trainloader, testloader


def fold_conv_bn(conv_w, bn_gamma, bn_beta, running_mean, running_var, eps):
    denom = torch.sqrt(running_var + eps)
    scale = bn_gamma / denom
    folded_weight = conv_w * scale.view(-1, 1, 1, 1)
    folded_bias = bn_beta - scale * running_mean
    return folded_weight, folded_bias


def quantize_to_q8_8(tensor):
    return torch.round(tensor * 256.0).clamp(-32768, 32767).to(torch.int16)


def write_c_array(handle, name, data):
    flat_data = data.flatten().cpu().numpy()
    handle.write(f"const int16_t {name}[{len(flat_data)}] = {{\n    ")
    for i, val in enumerate(flat_data):
        handle.write(f"{int(val)}, ")
        if (i + 1) % 16 == 0:
            handle.write("\n    ")
    handle.write("\n};\n\n")


def export_weights_h(model, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="ascii") as f:
        f.write("// [GENERATED] Q8.8 folded weights for firmware\n")
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        conv1_w, conv1_b = fold_conv_bn(
            model.features[0].weight.data,
            model.features[1].weight.data,
            model.features[1].bias.data,
            model.features[1].running_mean,
            model.features[1].running_var,
            model.features[1].eps,
        )
        conv2_w, conv2_b = fold_conv_bn(
            model.features[4].weight.data,
            model.features[5].weight.data,
            model.features[5].bias.data,
            model.features[5].running_mean,
            model.features[5].running_var,
            model.features[5].eps,
        )
        conv3_w, conv3_b = fold_conv_bn(
            model.features[8].weight.data,
            model.features[9].weight.data,
            model.features[9].bias.data,
            model.features[9].running_mean,
            model.features[9].running_var,
            model.features[9].eps,
        )

        write_c_array(f, "conv1_weight", quantize_to_q8_8(conv1_w))
        write_c_array(f, "conv1_bias", quantize_to_q8_8(conv1_b))
        write_c_array(f, "conv2_weight", quantize_to_q8_8(conv2_w))
        write_c_array(f, "conv2_bias", quantize_to_q8_8(conv2_b))
        write_c_array(f, "conv3_weight", quantize_to_q8_8(conv3_w))
        write_c_array(f, "conv3_bias", quantize_to_q8_8(conv3_b))
        write_c_array(f, "fc_weight", quantize_to_q8_8(model.classifier.weight.data))
        write_c_array(f, "fc_bias", quantize_to_q8_8(model.classifier.bias.data))
        f.write("#endif // WEIGHTS_H\n")
    print(f"[*] Saved weights to {filepath}")


def export_test_images_h(testloader, filepath):
    # Historical export path kept for reference; current Renode evaluation uses
    # scripts/export_eval_dataset.py to build an external image blob instead.
    images, labels = next(iter(testloader))
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="ascii") as f:
        f.write("// [GENERATED] Test images as C arrays\n")
        f.write("#ifndef TEST_IMAGES_H\n#define TEST_IMAGES_H\n\n")
        f.write("#include <stdint.h>\n\n")
        for i in range(10):
            img_q8 = (images[i].flatten() * 256).to(torch.int16).numpy()
            label_name = "Dog" if labels[i] == 1 else "Cat"
            f.write(f"// Image {i}: {label_name}\n")
            f.write(f"const int16_t test_img_{i}[1024] = {{\n    ")
            for j, val in enumerate(img_q8):
                f.write(f"{val}, ")
                if (j + 1) % 16 == 0:
                    f.write("\n    ")
            f.write("\n};\n")
            f.write(f"const int expected_label_{i} = {labels[i].item()};\n\n")
        f.write("#endif // TEST_IMAGES_H\n")
    print(f"[*] Saved test images to {filepath}")


def export_luts(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = 1024
    x_min, x_max = -8.0, 8.0
    x = np.linspace(x_min, x_max, entries)
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    relu_q8 = np.clip(np.round(relu * 256), -32768, 32767).astype(np.int16)
    sigmoid_q8 = np.clip(np.round(sigmoid * 256), -32768, 32767).astype(np.int16)

    def write_lut(filename, array_name, data):
        filepath = out_dir / filename
        with filepath.open("w", encoding="ascii") as f:
            f.write(f"// [GENERATED] {array_name} LUT (1024 entries, Q8.8)\n")
            f.write(f"// Range: {x_min} to {x_max}\n")
            f.write(f"#ifndef {array_name.upper()}_H\n#define {array_name.upper()}_H\n\n")
            f.write("#include <stdint.h>\n\n")
            f.write(f"const int16_t {array_name}[{entries}] = {{\n    ")
            for i, val in enumerate(data):
                f.write(f"{val}, ")
                if (i + 1) % 16 == 0:
                    f.write("\n    ")
            f.write("\n};\n\n")
            f.write(f"#endif // {array_name.upper()}_H\n")
        print(f"[*] Saved {filename} to {filepath}")

    write_lut("relu_lut.h", "relu_lut", relu_q8)
    write_lut("sigmoid_lut.h", "sigmoid_lut", sigmoid_q8)


def main():
    parser = argparse.ArgumentParser(
        description="Train TinyCatDogNet on CIFAR-10 cats vs dogs and export firmware weights."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--no-download", action="store_true", help="Use an existing CIFAR-10 download only.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    net = TinyCatDogNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    trainloader, testloader = get_cat_dog_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        download=not args.no_download,
    )
    best_acc = 0.0

    print("[*] Starting Training...")
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Loss: {running_loss / len(trainloader):.3f} - Val Accuracy: {acc:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), args.model_path)
            print(f"    --> Saved {args.model_path} (Accuracy: {best_acc:.2f}%)")

    print(f"[*] Training finished. Best Validation Accuracy: {best_acc:.2f}%")

    print("\n[*] Generating Firmware Files...")
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()
    export_weights_h(net, FIRMWARE_INC_DIR / "weights.h")
    export_luts(FIRMWARE_LUT_DIR)
    print("[*] Renode evaluation images are exported separately via scripts/export_eval_dataset.py")
    print("[*] All tasks completed successfully!")

if __name__ == "__main__":
    main()
