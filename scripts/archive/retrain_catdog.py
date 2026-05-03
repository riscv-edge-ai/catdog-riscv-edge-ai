import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

warnings.filterwarnings(
    "ignore",
    message=r".*align should be passed as.*",
    category=Warning,
)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_cat_dog(dataset):
    indices = [i for i, label in enumerate(dataset.targets) if label in [3, 5]]
    dataset.targets = [0 if dataset.targets[i] == 3 else 1 for i in indices]
    dataset.data = dataset.data[indices]
    return dataset


def get_cat_dog_dataloaders(data_root: Path, batch_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=str(data_root), train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(data_root), train=False, download=True, transform=eval_transform
    )

    trainset = filter_cat_dog(trainset)
    testset = filter_cat_dog(testset)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return trainloader, testloader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return loss_sum / total, (100.0 * correct / total)


def train(args):
    set_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"
    output_arg = Path(args.output)
    if output_arg.exists() and output_arg.is_dir():
        output_path = (output_arg / "best_catdog_retrained.pth").resolve()
    elif args.output.endswith("/"):
        output_path = (output_arg / "best_catdog_retrained.pth").resolve()
    else:
        output_path = output_arg.resolve()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] Data root: {data_root}")
    print(f"[*] Output: {output_path}")

    trainloader, testloader = get_cat_dog_dataloaders(data_root, args.batch_size)

    model = TinyCatDogNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for inputs, labels in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

        scheduler.step()
        train_loss = running_loss / total
        val_loss, val_acc = evaluate(model, testloader, device)
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.2f}% | lr={lr_now:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), output_path)
            print(f"    saved best model at epoch {best_epoch} with acc={best_acc:.2f}%")

    print(f"[*] Best validation accuracy: {best_acc:.2f}% at epoch {best_epoch}")


def main():
    parser = argparse.ArgumentParser(description="Retrain TinyCatDogNet on CIFAR-10 cats vs dogs")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default="/tmp/best_catdog_retrained.pth")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
