#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_BLOB = PROJECT_ROOT / "renode" / "test_images.bin"
DEFAULT_HEADER = PROJECT_ROOT / "firmware" / "include" / "eval_dataset_meta.h"
DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "eval_dataset_manifest.csv"


def load_catdog_samples(data_root: Path):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(data_root), train=False, download=False, transform=transform
    )

    samples = []
    for source_index, (image, label) in enumerate(testset):
        if label not in [3, 5]:
            continue
        mapped_label = 0 if label == 3 else 1
        image_q8 = torch.round(image.flatten() * 256.0).clamp(-32768, 32767).to(torch.int16)
        samples.append(
            {
                "source_index": source_index,
                "label": mapped_label,
                "image_q8": image_q8.numpy(),
            }
        )
    return samples


def write_blob(blob_path: Path, samples):
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    with blob_path.open("wb") as f:
        for sample in samples:
            f.write(sample["image_q8"].astype(np.int16).tobytes())


def write_header(header_path: Path, samples):
    header_path.parent.mkdir(parents=True, exist_ok=True)
    with header_path.open("w", encoding="ascii") as f:
        f.write("// [GENERATED] Evaluation dataset labels for Renode benchmarking\n")
        f.write("#ifndef EVAL_DATASET_META_H\n#define EVAL_DATASET_META_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define EVAL_DATASET_COUNT {len(samples)}\n\n")
        f.write("static const uint8_t eval_expected_labels[EVAL_DATASET_COUNT] = {\n    ")
        for idx, sample in enumerate(samples):
            f.write(f"{sample['label']}, ")
            if (idx + 1) % 32 == 0:
                f.write("\n    ")
        f.write("\n};\n\n")
        f.write("#endif\n")


def write_manifest(manifest_path: Path, samples):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset_index", "source_index", "label"])
        writer.writeheader()
        for dataset_index, sample in enumerate(samples):
            writer.writerow(
                {
                    "dataset_index": dataset_index,
                    "source_index": sample["source_index"],
                    "label": sample["label"],
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Export Renode evaluation images as an external binary blob.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--blob", type=Path, default=DEFAULT_BLOB)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    samples = load_catdog_samples(args.data_root)
    if not samples:
        raise SystemExit(f"No CIFAR-10 cat/dog samples found under {args.data_root}")

    if args.offset < 0:
        raise SystemExit("--offset must be non-negative")
    if args.offset >= len(samples):
        raise SystemExit(f"--offset {args.offset} is out of range for {len(samples)} available samples")

    remaining = len(samples) - args.offset
    count = min(args.count, remaining) if args.count > 0 else remaining
    selected = samples[args.offset : args.offset + count]

    write_blob(args.blob, selected)
    write_header(args.header, selected)
    write_manifest(args.manifest, selected)

    print(f"Exported {len(selected)} samples")
    print(f"Offset: {args.offset}")
    print(f"Blob: {args.blob}")
    print(f"Header: {args.header}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
