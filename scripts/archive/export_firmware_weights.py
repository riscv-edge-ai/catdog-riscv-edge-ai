#!/usr/bin/env python3
import os
import torch
import torch.nn as nn

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FW_INC = os.path.join(PROJECT_ROOT, "firmware", "include")
MODEL_PATH = os.path.join(PROJECT_ROOT, "best_catdog.pth")

os.makedirs(FW_INC, exist_ok=True)

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
        x = x.view(-1, 32 * 4 * 4)
        x = self.classifier(x)
        return x


def fold_conv_bn(conv_w, bn_gamma, bn_beta, running_mean, running_var, eps):
    # conv_w: (out_c, in_c, k, k)
    denom = torch.sqrt(running_var + eps)
    scale = bn_gamma / denom
    w_fold = conv_w * scale.view(-1, 1, 1, 1)
    b_fold = bn_beta - scale * running_mean
    return w_fold, b_fold


def to_q8_8(t):
    q = torch.round(t * 256.0).clamp(-32768, 32767).to(torch.int16)
    return q


def write_array(f, name, data, ctype="int16_t"):
    flat = data.flatten().tolist()
    f.write(f"const {ctype} {name}[{len(flat)}] = {{\n    ")
    for i, v in enumerate(flat):
        f.write(f"{int(v)}, ")
        if (i + 1) % 16 == 0:
            f.write("\n    ")
    f.write("\n};\n\n")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}")

    net = TinyCatDogNet()
    state = torch.load(MODEL_PATH, map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    # Fold BN into conv layers
    conv1_w = net.features[0].weight.data
    bn1 = net.features[1]
    conv1_wf, conv1_bf = fold_conv_bn(
        conv1_w, bn1.weight.data, bn1.bias.data, bn1.running_mean, bn1.running_var, bn1.eps
    )

    conv2_w = net.features[4].weight.data
    bn2 = net.features[5]
    conv2_wf, conv2_bf = fold_conv_bn(
        conv2_w, bn2.weight.data, bn2.bias.data, bn2.running_mean, bn2.running_var, bn2.eps
    )

    conv3_w = net.features[8].weight.data
    bn3 = net.features[9]
    conv3_wf, conv3_bf = fold_conv_bn(
        conv3_w, bn3.weight.data, bn3.bias.data, bn3.running_mean, bn3.running_var, bn3.eps
    )

    fc_w = net.classifier.weight.data
    fc_b = net.classifier.bias.data

    # Quantize to Q8.8 int16
    conv1_w_q = to_q8_8(conv1_wf)
    conv1_b_q = to_q8_8(conv1_bf)
    conv2_w_q = to_q8_8(conv2_wf)
    conv2_b_q = to_q8_8(conv2_bf)
    conv3_w_q = to_q8_8(conv3_wf)
    conv3_b_q = to_q8_8(conv3_bf)
    fc_w_q = to_q8_8(fc_w)
    fc_b_q = to_q8_8(fc_b)

    out_path = os.path.join(FW_INC, "weights.h")
    with open(out_path, "w") as f:
        f.write("// [GENERATED] Q8.8 folded weights for firmware\n")
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        write_array(f, "conv1_weight", conv1_w_q)
        write_array(f, "conv1_bias", conv1_b_q)
        write_array(f, "conv2_weight", conv2_w_q)
        write_array(f, "conv2_bias", conv2_b_q)
        write_array(f, "conv3_weight", conv3_w_q)
        write_array(f, "conv3_bias", conv3_b_q)
        write_array(f, "fc_weight", fc_w_q)
        write_array(f, "fc_bias", fc_b_q)
        f.write("#endif // WEIGHTS_H\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
