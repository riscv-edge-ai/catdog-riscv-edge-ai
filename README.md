# RISC-V Edge AI Cat/Dog Classifier

This project trains a small convolutional neural network (CNN) to classify CIFAR-10 cat and dog images, exports quantized weights for firmware, and runs the model in a Renode-based RISC-V simulation flow.

## What is included

- `train_catdog.py` trains `TinyCatDogNet` with PyTorch and exports firmware weight/LUT headers.
- `firmware/` contains the bare-metal RISC-V inference code.
- `scripts/export_eval_dataset.py` exports CIFAR-10 cat/dog test images for Renode.
- `scripts/run_renode_full_uart.py` runs the Renode benchmark and saves UART logs.
- `results/` contains packaged benchmark outputs from previous runs.
- `best_catdog.pth` is a saved trained PyTorch model checkpoint.

The local `venv/` and downloaded `data/` directory are intentionally ignored by Git because they are large and can be recreated.

## Requirements

- Python 3.10 or newer
- PyTorch and torchvision
- NumPy
- RISC-V GCC toolchain, available as `riscv32-unknown-elf-gcc`
- Renode, for the simulation benchmark

## Setup

Create a virtual environment and install the Python packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train the CNN

Run a quick smoke test with one epoch:

```bash
python train_catdog.py --epochs 1
```

Run the full training flow:

```bash
python train_catdog.py --epochs 100
```

The training script downloads CIFAR-10 into `data/`, filters it to cats and dogs, trains the CNN, saves `best_catdog.pth`, and regenerates:

- `firmware/include/weights.h`
- `firmware/lut/relu_lut.h`
- `firmware/lut/sigmoid_lut.h`

## Build the Firmware

After training or using the included generated headers, build the RISC-V firmware:

```bash
make -C firmware
```

This produces `firmware/catdog_inference.elf`.

## Export Test Images

Create the binary image blob and labels used by the firmware/Renode flow:

```bash
python scripts/export_eval_dataset.py --count 500
```

This writes:

- `renode/test_images.bin`
- `firmware/include/eval_dataset_meta.h`
- `results/eval_dataset_manifest.csv`

## Run the Renode Benchmark

For a full packaged 500-image benchmark, run:

```bash
./run_benchmark.sh
```

The benchmark is split into three Renode runs because the full 500-image UART run can be unstable on some PCs:

- offset `0`, count `200`
- offset `200`, count `200`
- offset `400`, count `100`

The script saves chunk logs, rebuilds `results/benchmark_500.csv`, and verifies consistency.

To run one smaller manual test:

```bash
python scripts/run_renode_full_uart.py --dataset-count 10 --dataset-offset 0 --macs-per-cycle 4 --timeout 1200
```

## Repository Notes

The repository does not include `venv/` or `data/` because the Python environment and CIFAR-10 dataset can be recreated locally.
