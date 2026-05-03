#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
WEIGHTS_HEADER = PROJECT_ROOT / "firmware" / "include" / "weights.h"


def parse_c_arrays(path: Path) -> dict[str, list[int]]:
    text = path.read_text(encoding="utf-8")
    arrays: dict[str, list[int]] = {}
    marker = "const int16_t "
    pos = 0
    while True:
        start = text.find(marker, pos)
        if start == -1:
            break
        name_start = start + len(marker)
        name_end = text.find("[", name_start)
        brace_start = text.find("{", name_end)
        brace_end = text.find("};", brace_start)
        name = text[name_start:name_end].strip()
        raw_values = text[brace_start + 1:brace_end].replace("\n", " ").split(",")
        values = [int(v.strip()) for v in raw_values if v.strip()]
        arrays[name] = values
        pos = brace_end + 2
    return arrays


def clamp_s16(v: int) -> int:
    if v > 32767:
        return 32767
    if v < -32768:
        return -32768
    return v


def to_s32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


def sw_sigmoid_q8(x_q8: int) -> int:
    if x_q8 <= -1024:
        return 0
    if x_q8 >= 1024:
        return 256
    linear = int(x_q8 / 8) + 128
    if linear < 0:
        linear = 0
    if linear > 256:
        linear = 256
    return linear


def legacy_accel_sigmoid_q8(x_q8: int) -> int:
    xf = x_q8 / 256.0
    y = int(round((1.0 / (1.0 + math.exp(-xf))) * 256))
    return clamp_s16(y)


def relu(x: int) -> int:
    return x if x > 0 else 0


def conv2d_q8_same(input_arr, ch_in, h, w, kernel, bias, ch_out, emulate_s32: bool):
    out = [0] * (ch_out * h * w)
    max_abs_acc = 0
    wraps = 0
    for oc in range(ch_out):
        b = bias[oc]
        for oh in range(h):
            for ow in range(w):
                s = 0
                for ic in range(ch_in):
                    for ky in range(3):
                        in_y = oh + ky - 1
                        if in_y < 0 or in_y >= h:
                            continue
                        for kx in range(3):
                            in_x = ow + kx - 1
                            if in_x < 0 or in_x >= w:
                                continue
                            in_idx = ic * h * w + in_y * w + in_x
                            k_idx = oc * ch_in * 9 + ic * 9 + ky * 3 + kx
                            prod = input_arr[in_idx] * kernel[k_idx]
                            if emulate_s32:
                                new_s = to_s32(s + prod)
                                if new_s != s + prod:
                                    wraps += 1
                                s = new_s
                            else:
                                s += prod
                max_abs_acc = max(max_abs_acc, abs(s))
                if emulate_s32:
                    s = to_s32((to_s32(s) >> 8) + b)
                else:
                    s = (s >> 8) + b
                out[oc * h * w + oh * w + ow] = clamp_s16(s)
    return out, max_abs_acc, wraps


def maxpool2d_q8(input_arr, ch, h, w):
    out_h = h // 2
    out_w = w // 2
    out = [0] * (ch * out_h * out_w)
    for c in range(ch):
        for oh in range(out_h):
            for ow in range(out_w):
                m = -32768
                for ky in range(2):
                    for kx in range(2):
                        idx = c * h * w + (oh * 2 + ky) * w + (ow * 2 + kx)
                        if input_arr[idx] > m:
                            m = input_arr[idx]
                out[c * out_h * out_w + oh * out_w + ow] = m
    return out


def fc_q8(input_arr, weights, bias, out_dim):
    out = [0] * out_dim
    max_abs_acc = 0
    wraps = 0
    in_dim = len(input_arr)
    for o in range(out_dim):
        s = 0
        for i in range(in_dim):
            prod = input_arr[i] * weights[o * in_dim + i]
            new_s = to_s32(s + prod)
            if new_s != s + prod:
                wraps += 1
            s = new_s
        max_abs_acc = max(max_abs_acc, abs(s))
        s = to_s32((to_s32(s) >> 8) + bias[o])
        out[o] = clamp_s16(s)
    return out, max_abs_acc, wraps


@dataclass
class PathConfig:
    name: str
    conv_emulate_s32: bool
    relu1_accel: bool
    relu23_accel: bool
    sigmoid_mode: str  # "sw" or "legacy_accel"


def apply_activation(data, func: str):
    if func == "relu":
        return [relu(v) for v in data]
    if func == "sw_sigmoid":
        return [sw_sigmoid_q8(v) for v in data]
    if func == "legacy_accel_sigmoid":
        return [legacy_accel_sigmoid_q8(v) for v in data]
    raise ValueError(func)


def run_path(image, arrays, config: PathConfig):
    stats = {"max_abs_acc": 0, "wraps": 0}
    layer_names = {}

    conv1, max_acc, wraps = conv2d_q8_same(image, 1, 32, 32, arrays["conv1_weight"], arrays["conv1_bias"], 8, config.conv_emulate_s32)
    stats["max_abs_acc"] = max(stats["max_abs_acc"], max_acc)
    stats["wraps"] += wraps
    layer_names["conv1"] = conv1
    relu1 = apply_activation(conv1, "relu")
    layer_names["relu1"] = relu1
    pool1 = maxpool2d_q8(relu1, 8, 32, 32)
    layer_names["pool1"] = pool1

    conv2, max_acc, wraps = conv2d_q8_same(pool1, 8, 16, 16, arrays["conv2_weight"], arrays["conv2_bias"], 16, config.conv_emulate_s32)
    stats["max_abs_acc"] = max(stats["max_abs_acc"], max_acc)
    stats["wraps"] += wraps
    layer_names["conv2"] = conv2
    relu2 = apply_activation(conv2, "relu")
    layer_names["relu2"] = relu2
    pool2 = maxpool2d_q8(relu2, 16, 16, 16)
    layer_names["pool2"] = pool2

    conv3, max_acc, wraps = conv2d_q8_same(pool2, 16, 8, 8, arrays["conv3_weight"], arrays["conv3_bias"], 32, config.conv_emulate_s32)
    stats["max_abs_acc"] = max(stats["max_abs_acc"], max_acc)
    stats["wraps"] += wraps
    layer_names["conv3"] = conv3
    relu3 = apply_activation(conv3, "relu")
    layer_names["relu3"] = relu3
    pool3 = maxpool2d_q8(relu3, 32, 8, 8)
    layer_names["pool3"] = pool3

    fc_out, max_acc, wraps = fc_q8(pool3[:512], arrays["fc_weight"], arrays["fc_bias"], 2)
    stats["max_abs_acc"] = max(stats["max_abs_acc"], max_acc)
    stats["wraps"] += wraps
    layer_names["fc"] = fc_out

    if config.sigmoid_mode == "sw":
        sigmoid = apply_activation(fc_out, "sw_sigmoid")
    elif config.sigmoid_mode == "legacy_accel":
        sigmoid = apply_activation(fc_out, "legacy_accel_sigmoid")
    else:
        raise ValueError(config.sigmoid_mode)
    layer_names["sigmoid"] = sigmoid
    layer_names["pred"] = 0 if sigmoid[0] > sigmoid[1] else 1
    return layer_names, stats


def compare_layers(lhs, rhs):
    summary = {}
    first_divergence = None
    for layer in ["conv1", "relu1", "pool1", "conv2", "relu2", "pool2", "conv3", "relu3", "pool3", "fc", "sigmoid"]:
        a = lhs[layer]
        b = rhs[layer]
        mismatches = sum(1 for x, y in zip(a, b) if x != y)
        max_abs_err = max((abs(x - y) for x, y in zip(a, b)), default=0)
        summary[layer] = {"mismatches": mismatches, "max_abs_err": max_abs_err}
        if first_divergence is None and mismatches:
            first_divergence = layer
    pred_match = lhs["pred"] == rhs["pred"]
    return summary, first_divergence, pred_match


def load_eval_images(count: int):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(root=str(DATA_ROOT), train=False, download=False, transform=transform)
    images = []
    labels = []
    for image, label in testset:
        if label not in [3, 5]:
            continue
        q8 = torch.round(image.flatten() * 256.0).clamp(-32768, 32767).to(torch.int16).tolist()
        images.append([int(v) for v in q8])
        labels.append(0 if label == 3 else 1)
        if len(images) == count:
            break
    return images, labels


def main():
    parser = argparse.ArgumentParser(description="Compare firmware software math against legacy and current accelerator math.")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    arrays = parse_c_arrays(WEIGHTS_HEADER)
    images, labels = load_eval_images(args.count)

    software = PathConfig("software", conv_emulate_s32=True, relu1_accel=False, relu23_accel=False, sigmoid_mode="sw")
    accel_legacy = PathConfig("accel_legacy", conv_emulate_s32=False, relu1_accel=False, relu23_accel=True, sigmoid_mode="legacy_accel")
    accel_current = PathConfig("accel_current", conv_emulate_s32=True, relu1_accel=False, relu23_accel=True, sigmoid_mode="sw")

    totals = {
        "legacy_pred_mismatches": 0,
        "current_pred_mismatches": 0,
        "legacy_first_divergence": {},
        "current_first_divergence": {},
        "legacy_layer_max_abs": {},
        "current_layer_max_abs": {},
        "legacy_layer_mismatches": {},
        "current_layer_mismatches": {},
        "legacy_wraps": 0,
        "current_wraps": 0,
        "software_wraps": 0,
        "legacy_max_abs_acc": 0,
        "current_max_abs_acc": 0,
        "software_max_abs_acc": 0,
    }

    for image, _label in zip(images, labels):
        sw_layers, sw_stats = run_path(image, arrays, software)
        legacy_layers, legacy_stats = run_path(image, arrays, accel_legacy)
        current_layers, current_stats = run_path(image, arrays, accel_current)

        legacy_summary, legacy_first, legacy_pred_match = compare_layers(sw_layers, legacy_layers)
        current_summary, current_first, current_pred_match = compare_layers(sw_layers, current_layers)

        if not legacy_pred_match:
            totals["legacy_pred_mismatches"] += 1
        if not current_pred_match:
            totals["current_pred_mismatches"] += 1

        if legacy_first is not None:
            totals["legacy_first_divergence"][legacy_first] = totals["legacy_first_divergence"].get(legacy_first, 0) + 1
        if current_first is not None:
            totals["current_first_divergence"][current_first] = totals["current_first_divergence"].get(current_first, 0) + 1

        for layer, values in legacy_summary.items():
            totals["legacy_layer_max_abs"][layer] = max(totals["legacy_layer_max_abs"].get(layer, 0), values["max_abs_err"])
            totals["legacy_layer_mismatches"][layer] = totals["legacy_layer_mismatches"].get(layer, 0) + values["mismatches"]
        for layer, values in current_summary.items():
            totals["current_layer_max_abs"][layer] = max(totals["current_layer_max_abs"].get(layer, 0), values["max_abs_err"])
            totals["current_layer_mismatches"][layer] = totals["current_layer_mismatches"].get(layer, 0) + values["mismatches"]

        totals["legacy_wraps"] += legacy_stats["wraps"]
        totals["current_wraps"] += current_stats["wraps"]
        totals["software_wraps"] += sw_stats["wraps"]
        totals["legacy_max_abs_acc"] = max(totals["legacy_max_abs_acc"], legacy_stats["max_abs_acc"])
        totals["current_max_abs_acc"] = max(totals["current_max_abs_acc"], current_stats["max_abs_acc"])
        totals["software_max_abs_acc"] = max(totals["software_max_abs_acc"], sw_stats["max_abs_acc"])

    print(f"Compared images: {len(images)}")
    print("Before fix vs software:")
    print(f"  prediction mismatches: {totals['legacy_pred_mismatches']}")
    print(f"  first divergence counts: {totals['legacy_first_divergence']}")
    print(f"  layer max abs error: {totals['legacy_layer_max_abs']}")
    print(f"  layer mismatch counts: {totals['legacy_layer_mismatches']}")
    print(f"  max abs accumulator magnitude: {totals['legacy_max_abs_acc']}")
    print(f"  observed 32-bit wrap events if modeled in software path: {totals['software_wraps']}")
    print("After fix vs software:")
    print(f"  prediction mismatches: {totals['current_pred_mismatches']}")
    print(f"  first divergence counts: {totals['current_first_divergence']}")
    print(f"  layer max abs error: {totals['current_layer_max_abs']}")
    print(f"  layer mismatch counts: {totals['current_layer_mismatches']}")
    print(f"  max abs accumulator magnitude: {totals['current_max_abs_acc']}")
    print(f"  current path wrap events: {totals['current_wraps']}")
    print("Software reference accumulator stats:")
    print(f"  max abs accumulator magnitude: {totals['software_max_abs_acc']}")
    print(f"  wrap events: {totals['software_wraps']}")


if __name__ == "__main__":
    main()
